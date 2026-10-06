#!/usr/bin/env python3
"""
Run inference with a trained AsParaSeg / AsParaSeg-H segmenter.

Loads a checkpoint (<run_name>.best.pt) together with its companion
<run_name>.result.json (for the model's hyperparameters and the dev-tuned
decision thresholds), then segments one or more documents you supply as
plain text -- one sentence per line, a blank line between documents.

    python infer.py --result hier_runs/RUN.result.json --input doc.txt
    python infer.py --result fine_runs/RUN.result.json --input doc.txt --json out.json

Run this from the same directory as model.py (or put it on PYTHONPATH) --
it imports your actual model classes rather than reimplementing them.

Checkpoint path is derived from --result by default (same directory, same
run name, ".best.pt" instead of ".result.json") -- pass --checkpoint to
override if it's named differently.

ASSUMPTIONS -- built from model.py + train.py, which I have read in full.
I have NOT seen seg_data.py, so two things here are inferred, not verified:

  1. Input is raw text pre-split into sentences (one per line). Your
     Assamese sentence segmentation logic lives in seg_data.py/precompute.py,
     which I don't have -- so segmenting new text into sentences is on you;
     this script starts from sentences you've already split.

  2. A predicted label at sentence i is rendered here as "a boundary follows
     sentence i" (paragraph break after it for label>=1, section break for
     label==2 in the hier model). I inferred this direction from how Pk/
     WindowDiff treat boundary positions in your setup -- before trusting it
     on new text, run it on one dev document whose true labels you know and
     check the printed breaks land where the reference says they should.

  3. Documents longer than 1024 sentences are truncated: ContextSegmenter's
     position embedding table is nn.Embedding(1024, d).

This processes one document at a time (no batching) -- fine for inference
use; for scoring many documents at scale you'd want the same bucketing/
collate machinery train.py uses, which lives in seg_data.py.
"""
import argparse
import json
import os
import sys

import torch

from model import ContextSegmenter, EndToEndSegmenter, SentenceEncoder, decode_fine, decode_hier


def load_run(result_path, checkpoint_path=None, device="cpu"):
    with open(result_path) as f:
        result = json.load(f)
    a = result["args"]
    best_epoch = result["best_epoch"]
    thr_entry = next(h for h in result["history"] if h["epoch"] == best_epoch)
    thr_a, thr_b = thr_entry["thr_a"], thr_entry["thr_b"]

    if checkpoint_path is None:
        checkpoint_path = result_path.replace(".result.json", ".best.pt")
    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"checkpoint not found: {checkpoint_path}\n"
            f"pass --checkpoint explicitly if it's named differently")

    task = a["task"]
    mode = a["mode"]
    n_heads = 2 if task == "hier" else 1
    ctx_kw = dict(n_layers=a["ctx_layers"], d_model=a.get("ctx_dim"), dropout=a["dropout"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a["encoder"])

    if mode == "finetune":
        model = EndToEndSegmenter(a["encoder"], n_heads=n_heads,
                                  freeze_encoder=False, **ctx_kw)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        sent_encoder = None  # folded into `model` for finetune checkpoints
    else:
        sent_encoder = SentenceEncoder(a["encoder"], freeze=True).to(device).eval()
        model = ContextSegmenter(sent_encoder.hidden_size, n_heads=n_heads, **ctx_kw)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()

    return dict(model=model, sent_encoder=sent_encoder, tok=tok, task=task, mode=mode,
                thr_a=thr_a, thr_b=thr_b, max_tokens=a.get("max_tokens", 64),
                run_name=result["run_name"])


@torch.no_grad()
def segment(run, sentences, device="cpu"):
    if len(sentences) == 0:
        return []
    if len(sentences) > 1024:
        print(f"warning: {len(sentences)} sentences, truncating to 1024 "
              f"(ContextSegmenter's position-embedding limit)", file=sys.stderr)
        sentences = sentences[:1024]

    tok = run["tok"]
    enc = tok(sentences, padding="max_length", truncation=True,
               max_length=run["max_tokens"], return_tensors="pt")
    input_ids = enc["input_ids"].unsqueeze(0).to(device)         # (1, S, T)
    attention_mask = enc["attention_mask"].unsqueeze(0).to(device)
    sent_mask = torch.ones(1, len(sentences), device=device)

    if run["mode"] == "finetune":
        outs = run["model"](input_ids, attention_mask, sent_mask)
    else:
        sent_vecs = run["sent_encoder"](input_ids, attention_mask)
        outs = run["model"](sent_vecs, sent_mask)

    if run["task"] == "fine":
        labels = decode_fine(outs[0], thr=run["thr_a"]).squeeze(0).tolist()
    else:
        labels = decode_hier(outs[0], outs[1], thr_a=run["thr_a"],
                              thr_b=run["thr_b"]).squeeze(0).tolist()
    return labels


def read_documents(path):
    """Plain text: sentences one per line, a blank line separates documents."""
    docs, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if cur:
                    docs.append(cur)
                    cur = []
            else:
                cur.append(line)
    if cur:
        docs.append(cur)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="<run_name>.result.json")
    ap.add_argument("--checkpoint", default=None, help="override: <run_name>.best.pt")
    ap.add_argument("--input", required=True,
                    help="text file: one sentence per line, blank line between documents")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None, help="also write structured output here")
    args = ap.parse_args()

    run = load_run(args.result, args.checkpoint, device=args.device)
    print(f"loaded {run['run_name']}  (task={run['task']} mode={run['mode']}  "
          f"thr_a={run['thr_a']} thr_b={run['thr_b']})", file=sys.stderr)

    docs = read_documents(args.input)
    if not docs:
        raise SystemExit(f"no sentences found in {args.input}")

    all_out = []
    for di, sentences in enumerate(docs):
        labels = segment(run, sentences, device=args.device)
        all_out.append({"sentences": sentences, "labels": labels})

        print(f"\n=== document {di + 1} ===")
        for sent, lab in zip(sentences, labels):
            print(sent)
            if run["task"] == "fine" and lab >= 1:
                print("    --- paragraph break ---")
            elif run["task"] == "hier" and lab == 1:
                print("    --- paragraph break ---")
            elif run["task"] == "hier" and lab == 2:
                print("    === section break ===")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
