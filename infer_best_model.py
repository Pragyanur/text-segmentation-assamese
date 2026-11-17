"""
End-to-end inference for AsParaSeg / AsParaSeg-H.
Uses `seg_data.make_text_collate` and `assamese_sent.py` directly so tokenization
and tensor shapes match `train.py` 100%.
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel

import assamese_sent
from model import ContextSegmenter, EndToEndSegmenter
from seg_data import make_text_collate


def resolve_run_paths(run_path: str):
    for suffix in (".result.json", ".best.pt"):
        if run_path.endswith(suffix):
            base = run_path[: -len(suffix)]
            break
    else:
        base = run_path

    res_path = f"{base}.result.json"
    ckpt_path = f"{base}.best.pt"
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Could not find result metadata: {res_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint weights: {ckpt_path}")
    return res_path, ckpt_path


def resolve_encoder_name(enc: str) -> str:
    """Maps short repo aliases like 'muril-base-cased' to valid HF IDs if not a local path."""
    if os.path.exists(enc) or "/" in enc:
        return enc
    aliases = {
        "muril-base-cased": "google/muril-base-cased",
        "muril-large-cased": "google/muril-large-cased",
        "indic-bert": "ai4bharat/indic-bert",
    }
    return aliases.get(enc, enc)


def split_assamese_sentences(text: str) -> list[str]:
    """Finds and calls the sentence splitting function in assamese_sent.py."""
    for fn_name in ("split_sentences", "sent_tokenize", "split_sents", "sentences", "split"):
        fn = getattr(assamese_sent, fn_name, None)
        if callable(fn):
            return [s.strip() for s in fn(text) if s and s.strip()]
    available = [k for k in dir(assamese_sent) if not k.startswith("_") and callable(getattr(assamese_sent, k))]
    raise AttributeError(
        f"Could not find sentence splitter in assamese_sent.py. Available functions: {available}"
    )


def get_best_thresholds(result_meta: dict) -> tuple[float, float]:
    best_ep = result_meta["best_epoch"]
    for entry in result_meta.get("history", []):
        if entry["epoch"] == best_ep:
            return float(entry["thr_a"]), float(entry.get("thr_b", 0.5))
    raise ValueError(f"best_epoch={best_ep} not found in result history.")


class AssameseSegmenterPredictor:
    def __init__(self, run_path: str, device: str | None = None, encoder_override: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        res_path, ckpt_path = resolve_run_paths(run_path)

        with open(res_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self.args = meta["args"]
        self.task = self.args["task"]
        self.mode = self.args["mode"]
        self.max_tokens = self.args.get("max_tokens", 64)
        self.thr_a, self.thr_b = get_best_thresholds(meta)

        raw_encoder = encoder_override or self.args["encoder"]
        hf_encoder = resolve_encoder_name(raw_encoder)

        n_heads = 2 if self.task == "hier" else 1
        ctx_layers = self.args.get("ctx_layers", 2)
        ctx_dim = self.args.get("ctx_dim", None)
        dropout = self.args.get("dropout", 0.1)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_encoder)
        self.collate_fn = make_text_collate(self.tokenizer, self.max_tokens)
        state_dict = torch.load(ckpt_path, map_location=self.device)

        if self.mode == "finetune":
            self.encoder = None
            # Try raw_encoder first in case EndToEndSegmenter resolves aliases internally;
            # fall back to hf_encoder if it passes directly to AutoModel
            try:
                self.model = EndToEndSegmenter(
                    raw_encoder,
                    n_heads=n_heads,
                    freeze_encoder=True,
                    n_layers=ctx_layers,
                    d_model=ctx_dim,
                    dropout=dropout,
                ).to(self.device)
            except OSError:
                self.model = EndToEndSegmenter(
                    hf_encoder,
                    n_heads=n_heads,
                    freeze_encoder=True,
                    n_layers=ctx_layers,
                    d_model=ctx_dim,
                    dropout=dropout,
                ).to(self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.encoder = AutoModel.from_pretrained(hf_encoder).to(self.device)
            self.encoder.eval()
            d_in = self.encoder.config.hidden_size
            self.model = ContextSegmenter(
                d_in,
                n_heads=n_heads,
                d_model=ctx_dim,
                n_layers=ctx_layers,
                dropout=dropout,
            ).to(self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    @torch.no_grad()
    def predict_sentences(self, sentences: list[str]) -> dict:
        n = len(sentences)
        if n == 0:
            return {"sentences": [], "labels": [], "probs_a": [], "probs_b": []}
        if n == 1:
            return {"sentences": sentences, "labels": [0], "probs_a": [0.0], "probs_b": [0.0]}

       # Build a single-document batch using the exact collate function from seg_data.py
        dummy_item = {
            "id": "infer_0",
            "sentences": sentences,
            "labels": [0] * n,
            "n": n,
        }
        batch = self.collate_fn([dummy_item])
        sent_mask = batch["sent_mask"].to(self.device)

        if self.mode == "finetune":
            outs = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                sent_mask,
            )
        else:
            enc = self.tokenizer(
                sentences, padding=True, truncation=True,
                max_length=self.max_tokens, return_tensors="pt"
            ).to(self.device)
            out = self.encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            sent_vecs = ((out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).unsqueeze(0)
            outs = self.model(sent_vecs, sent_mask)

        probs_a = torch.sigmoid(outs[0][0]).cpu().tolist()
        probs_b = torch.sigmoid(outs[1][0]).cpu().tolist() if self.task == "hier" else [0.0] * n

        probs_a[-1] = 0.0
        probs_b[-1] = 0.0

        labels = []
        for i in range(n):
            if i == n - 1:
                labels.append(0)
            elif self.task == "hier":
                if probs_a[i] >= self.thr_a and probs_b[i] >= self.thr_b:
                    labels.append(2)
                elif probs_a[i] >= self.thr_a:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(1 if probs_a[i] >= self.thr_a else 0)
        print(labels)
        return {
            "sentences": sentences,
            "labels": labels,
            "probs_a": [round(p, 4) for p in probs_a],
            "probs_b": [round(p, 4) for p in probs_b] if self.task == "hier" else None,
        }

    def segment_text(self, text: str, para_sep: str = "\n\n", sec_sep: str = "\n\n---\n\n") -> tuple[str, dict]:
        sentences = split_assamese_sentences(text)
        res = self.predict_sentences(sentences)

        chunks, cur_para = [], []
        for sent, lbl in zip(res["sentences"], res["labels"]):
            cur_para.append(sent)
            if lbl == 1:
                chunks.append(" ".join(cur_para))
                chunks.append(para_sep)
                cur_para = []
            elif lbl == 2:
                chunks.append(" ".join(cur_para))
                chunks.append(sec_sep)
                cur_para = []

        if cur_para:
            chunks.append(" ".join(cur_para))

        return "".join(chunks), res


def main():
    ap = argparse.ArgumentParser(description="Run inference with a trained AsParaSeg-H model.")
    ap.add_argument("--run", default="best_model/hier_ft_tl3_bs256_mbs800.result.json")
    ap.add_argument("--text", default=None)
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--out", default="output.txt")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.text is not None:
        raw_text = args.text
    elif args.input is not None and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    else:
        raw_text = sys.stdin.read()

    predictor = AssameseSegmenterPredictor(
        run_path=args.run,
        device=args.device,
        encoder_override=args.encoder,
    )
    segmented_text, details = predictor.segment_text(raw_text)

    output_str = json.dumps(details, ensure_ascii=False, indent=2) if args.json else segmented_text

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(output_str + "\n")
        print(f"Wrote segmented output to {args.out}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()