#!/usr/bin/env python3
"""
Train either segmenter.

    # fine model (section-level, binary) -- the paragraph-break model
    python precompute.py dataset      --task fine --out vecs_fine
    python train.py dataset --task fine --mode frozen --vectors vecs_fine

    # hierarchical model (article-level, 3-way)
    python precompute.py dataset_hier --task hier --out vecs_hier
    python train.py dataset_hier --task hier --mode frozen --vectors vecs_hier

    # end-to-end finetuning of the encoder (needs a GPU)
    python train.py dataset --task fine --mode finetune --epochs 3 --lr 2e-5

Default encoder is ai4bharat/indic-bert (ALBERT; needs `pip install
sentencepiece`). Swap with --encoder -- the code reads hidden_size off the
config, so any BERT-family encoder or a local folder path works.

Start with --mode frozen. It trains only the context transformer over cached
sentence vectors, runs in minutes, and settles the design questions before you
spend anything on finetuning.

Reported metrics are P/R/F1, Pk and WindowDiff -- never bare accuracy, which
the all-zero predictor already wins. The decision threshold is tuned on DEV
and then applied unchanged to TEST.
"""
import argparse
import hashlib
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader

from seg_data import (SegDataset, make_text_collate, make_vector_collate,
                      BucketBatchSampler, unpad)
from metrics import prf1_corpus, pk_wd_corpus, best_threshold, summarize
from model import (ContextSegmenter, EndToEndSegmenter, fine_loss,
                   hierarchical_loss, pos_weight_for)


def run_slug(args, lr):
    """A filename that states the configuration it came from.

    Every run used to land on runs/run1/result.json, so each one silently
    overwrote the last and results could not be told apart after the fact.
    The slug carries every setting that changes the numbers, plus a short
    hash of the FULL argument set so two runs differing only in something
    not spelled out below still cannot collide.
    """
    def clean(s):
        return re.sub(r"[^A-Za-z0-9._-]", "-", str(s))

    parts = [
        clean(os.path.basename(os.path.normpath(args.data))),
        args.task,
        args.mode,
        clean(os.path.basename(args.encoder.rstrip("/\\")) or "enc"),
        f"e{args.epochs}",
        f"L{args.ctx_layers}",
        f"d{args.dropout:g}",
        f"lr{lr:g}",
        f"bs{args.batch_size}",
        f"wd{args.weight_decay:g}",
        f"s{args.seed}",
    ]
    if args.task == "hier":
        parts.append(f"wb{args.w_b:g}")
    if args.ctx_dim:
        parts.append(f"dim{args.ctx_dim}")
    if args.max_sents:
        parts.append(f"ms{args.max_sents}")
    if args.no_pos_weight:
        parts.append("nopw")
    if args.patience:
        parts.append(f"pat{args.patience}")
    if args.mode == "finetune":
        parts.append("encALL" if args.train_layers < 0
                     else f"enc{args.train_layers}L")
        if args.head_lr:
            parts.append(f"hlr{args.head_lr:g}")
    if args.select_on != "f1":
        parts.append(f"sel-{args.select_on}")
    cfg = hashlib.sha1(
        json.dumps(vars(args), sort_keys=True, default=str).encode()
    ).hexdigest()[:6]
    return "_".join(parts) + "_" + cfg


def build_loader(ds, batch_size, collate, shuffle, seed=0, max_batch_sents=0):
    return DataLoader(ds, collate_fn=collate,
                      batch_sampler=BucketBatchSampler(
                          [it["n"] for it in ds], batch_size,
                          shuffle=shuffle, seed=seed,
                          max_batch_sents=max_batch_sents))


def gpu_report(device, tag=""):
    """Allocated vs reserved. A large and GROWING gap between the two is
    fragmentation, not a leak: the caching allocator is holding blocks it
    cannot reuse because batch shapes keep changing."""
    if not str(device).startswith("cuda"):
        return ""
    a = torch.cuda.max_memory_allocated() / 2**30
    r = torch.cuda.max_memory_reserved() / 2**30
    return f"{tag}peak {a:.2f} GB alloc / {r:.2f} GB reserved"


def run_epoch(model, loader, task, device, mode, opt=None,
              pw=None, w_b=1.0, sched=None, clip=1.0,
              amp_dtype=None, collect=True):
    """One pass. opt=None -> evaluation. Returns (mean_loss, refs, probs)."""
    train = opt is not None
    model.train(train)
    total, nb = 0.0, 0
    refs, probs_a, probs_b = [], [], []

    for batch in loader:
        sent_mask = batch["sent_mask"].to(device)
        labels = batch["labels"].to(device)
        try:
            with torch.set_grad_enabled(train), torch.autocast(
                    device_type="cuda", dtype=amp_dtype,
                    enabled=amp_dtype is not None and str(device).startswith("cuda")):
                if mode == "finetune":
                    outs = model(batch["input_ids"].to(device),
                                 batch["attention_mask"].to(device), sent_mask)
                else:
                    outs = model(batch["sent_vecs"].to(device), sent_mask)

                if task == "fine":
                    loss = fine_loss(outs[0], labels, sent_mask, pos_weight=pw[0])
                else:
                    loss = hierarchical_loss(outs[0], outs[1], labels, sent_mask,
                                             pos_weight_a=pw[0], pos_weight_b=pw[1],
                                             w_b=w_b)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                if sched is not None:
                    sched.step()
        except torch.cuda.OutOfMemoryError:
            # say exactly which batch shape killed it, instead of a bare OOM
            B = len(batch["lengths"])
            S = max(batch["lengths"])
            shape = f"{B} docs x {S} sentences = {B*S} sequences"
            raise SystemExit(
                f"\nCUDA OOM on a batch of {shape}.\n"
                f"  {gpu_report(device)}\n"
                f"  Peak memory scales with that PRODUCT, not with "
                f"--batch-size.\n"
                f"  Cap it with --max-batch-sents (try "
                f"{max(32, B*S//2)}), and/or add --amp bf16.\n")

        total += loss.item()
        nb += 1
        if collect:
            L = batch["lengths"]
            refs.extend(unpad(labels.cpu(), L))
            probs_a.extend(unpad(
                torch.sigmoid(outs[0].detach().float().cpu()), L))
            if task == "hier":
                probs_b.extend(unpad(
                    torch.sigmoid(outs[1].detach().float().cpu()), L))
        del outs, loss

    return total / max(nb, 1), refs, probs_a, probs_b


def evaluate(task, refs, probs_a, probs_b, thr_a, thr_b):
    """Fine: one report. Hierarchical: head A, head B, and joint 3-way."""
    out = {}
    ref_a = [[1 if l >= 1 else 0 for l in r] for r in refs]
    hyp_a = [[1 if p >= thr_a else 0 for p in d] for d in probs_a]
    out["head_a"] = summarize(ref_a, hyp_a, name="boundary>=1")

    if task == "hier":
        ref_b = [[1 if l >= 2 else 0 for l in r] for r in refs]
        # monotonic decode: a 2 requires head A to have fired too
        hyp_b = [[1 if (pa >= thr_a and pb >= thr_b) else 0
                  for pa, pb in zip(da, db)]
                 for da, db in zip(probs_a, probs_b)]
        out["head_b"] = summarize(ref_b, hyp_b, name="boundary>=2")
        exact = correct = 0
        for r, ha, hb in zip(refs, hyp_a, hyp_b):
            for li, a, b in zip(r, ha, hb):
                pred = 2 if b else (1 if a else 0)
                correct += pred == li
                exact += 1
        out["three_way_accuracy"] = round(correct / max(exact, 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--task", choices=["fine", "hier"], required=True)
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--vectors", default=None, help="dir from precompute.py")
    ap.add_argument("--encoder", default="muril-base-cased")
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-sents", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--w-b", type=float, default=1.0,
                    help="weight on head B's loss (hierarchical only)")
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--select-on", choices=["f1", "loss", "pk"], default="f1",
        help="checkpoint selection criterion. 'pk' selects on the metric you "
             "actually report and is usually right. 'loss' measures "
             "calibration, which can diverge from placement: in the top-2 "
             "finetune run the dev-loss minimum was 0.021 Pk worse on head B "
             "than the best epoch. 'f1' is the historical default but goes "
             "flat while the model is still overfitting")
    ap.add_argument(
        "--patience", type=int, default=0,
        help="stop after this many epochs with no improvement in the "
             "selection criterion (0 = train the full schedule)")
    ap.add_argument("--name", default=None,
                    help="override the auto-generated run name")
    ap.add_argument(
        "--train-layers", type=int, default=2,
        help="finetune mode: train only the TOP N encoder layers, freezing "
             "the embeddings and every layer below. 0 freezes the encoder "
             "entirely; -1 trains all of it. Default 2")
    ap.add_argument(
        "--max-batch-sents", type=int, default=800,
        help="cap on padded sentences per batch (batch_size x longest doc). "
             "Peak memory tracks this PRODUCT, not batch_size, so this is what "
             "stops one long article from OOMing a run mid-epoch. 0 = off")
    ap.add_argument(
        "--amp", choices=["off", "bf16", "fp16"], default="off",
        help="mixed precision. bf16 halves activation memory on Ampere and "
             "later (A4000 included) and needs no loss scaling")
    ap.add_argument(
        "--head-lr", type=float, default=None,
        help="learning rate for the context transformer and output heads. "
             "They are randomly initialised and need a far higher rate than "
             "the pretrained encoder, which uses --lr. Defaults to 1e-3 in "
             "finetune mode and to --lr in frozen mode")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    n_heads = 2 if args.task == "hier" else 1
    lr = args.lr if args.lr is not None else (2e-5 if args.mode == "finetune" else 1e-3)
    # the head is trained from scratch; at the encoder's 2e-5 it would barely
    # move, and finetuning would look worse than frozen for the wrong reason
    head_lr = args.head_lr if args.head_lr is not None else (
        1e-3 if args.mode == "finetune" else lr)

    # output files are named for the configuration, so runs never overwrite
    run_name = args.name or run_slug(args, lr)
    res_path = os.path.join(args.out, f"{run_name}.result.json")
    ckpt_path = os.path.join(args.out, f"{run_name}.best.pt")
    print(f"run: {run_name}")

    # ---------------- data
    vecs = {}
    if args.mode == "frozen":
        if not args.vectors:
            raise SystemExit("--mode frozen needs --vectors from precompute.py")
        for split in ("train", "dev", "test"):
            p = os.path.join(args.vectors, f"{split}.pt")
            if os.path.exists(p):
                vecs[split] = torch.load(p)

    splits = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(args.data, f"{split}.jsonl")
        if os.path.exists(path):
            splits[split] = SegDataset(path, task=args.task,
                                       max_sents=args.max_sents,
                                       vectors=vecs.get(split))
    if "train" not in splits:
        raise SystemExit(f"no train.jsonl in {args.data}")

    counts = splits["train"].label_counts()
    total = sum(counts.values())
    n_a, n_b = counts[1] + counts[2], counts[2]
    print(f"train labels 0/1/2 = {counts[0]}/{counts[1]}/{counts[2]}  "
          f"({total} positions)")
    print(f"  head A positives (>=1): {n_a} ({100*n_a/total:.2f}%)")
    if args.task == "hier":
        print(f"  head B positives (>=2): {n_b} ({100*n_b/total:.2f}%)")

    if args.no_pos_weight:
        pw = (None, None)
    else:
        pw = (pos_weight_for(n_a, total).to(args.device),
              pos_weight_for(n_b, total).to(args.device) if args.task == "hier" else None)
        print(f"  pos_weight_a = {pw[0].item():.2f}"
              + (f"   pos_weight_b = {pw[1].item():.2f}" if pw[1] is not None else ""))
        if args.task == "hier":
            print("  (without pos_weight_b, head B collapses to always-zero "
                  "and still scores well on accuracy)")

    # ---------------- model
    if args.mode == "finetune":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.encoder)
        collate = make_text_collate(tok, args.max_tokens)
        model = EndToEndSegmenter(args.encoder, n_heads=n_heads,
                                  freeze_encoder=False,
                                  n_layers=args.ctx_layers,
                                  d_model=args.ctx_dim,
                                  dropout=args.dropout).to(args.device)
        if args.train_layers >= 0:
            got = model.enc.set_trainable_top_layers(args.train_layers)
            if got is None:
                print("  WARNING: this encoder has no ordered layer stack "
                      "(ALBERT shares weights across depth), so --train-layers "
                      "cannot select a subset. The whole encoder is frozen; "
                      "use --train-layers -1 to train all of it.")
            else:
                n, tot = got
                tr, al = model.enc.trainable_report()
                print(f"  training top {n} of {tot} encoder layers: "
                      f"{tr/1e6:.1f}M of {al/1e6:.1f}M encoder params "
                      f"({100*tr/max(al,1):.1f}%)")
        else:
            print("  training the FULL encoder")
    else:
        d_in = next(iter(vecs["train"].values())).shape[-1]
        collate = None
        model = ContextSegmenter(d_in, n_heads=n_heads,
                                 d_model=args.ctx_dim,
                                 n_layers=args.ctx_layers,
                                 dropout=args.dropout).to(args.device)

    loaders = {}
    for split, ds in splits.items():
        c = collate if args.mode == "finetune" else make_vector_collate(vecs[split])
        loaders[split] = build_loader(ds, args.batch_size, c,
                                      shuffle=(split == "train"), seed=args.seed,
                                      max_batch_sents=args.max_batch_sents)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    # what the heaviest batch will actually be -- the number that OOMs you
    _lens = [it["n"] for it in splits["train"]]
    _worst = max((len(b) * max(_lens[j] for j in b))
                 for b in loaders["train"].batch_sampler)
    print(f"  heaviest batch: {_worst} padded sentences"
          + (f"  (cap {args.max_batch_sents})" if args.max_batch_sents else
             "  <- uncapped; use --max-batch-sents if this OOMs"))

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_par:,} of {n_all:,}")

    # two parameter groups: the pretrained encoder is nudged, the freshly
    # initialised context transformer and heads are trained properly
    if args.mode == "finetune":
        enc_p = [p for p in model.enc.parameters() if p.requires_grad]
        head_p = [p for p in model.ctx.parameters() if p.requires_grad]
        groups = []
        if enc_p:
            groups.append({"params": enc_p, "lr": lr})
        groups.append({"params": head_p, "lr": head_lr})
        print(f"  lr: encoder {lr:g}"
              + (" (nothing trainable)" if not enc_p else "")
              + f"   head {head_lr:g}")
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": lr}]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)
    steps = max(1, len(loaders["train"]) * args.epochs)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else max(0.0, (steps - s) / max(1, steps - warm)))

    # ---------------- train
    best, history, stale = None, [], 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tr_loss, *_ = run_epoch(model, loaders["train"], args.task, args.device,
                                args.mode, opt=opt, pw=pw, w_b=args.w_b,
                                sched=sched, amp_dtype=amp_dtype, collect=False)
        dv_loss, refs, pa, pb = run_epoch(model, loaders["dev"], args.task,
                                          args.device, args.mode, pw=pw,
                                          w_b=args.w_b, amp_dtype=amp_dtype)
        # thresholds are tuned on DEV only
        ref_a = [[1 if l >= 1 else 0 for l in r] for r in refs]
        thr_a, _ = best_threshold(ref_a, pa)
        thr_b = 0.5
        if args.task == "hier":
            ref_b = [[1 if l >= 2 else 0 for l in r] for r in refs]
            thr_b, _ = best_threshold(ref_b, pb)
        rep = evaluate(args.task, refs, pa, pb, thr_a, thr_b)
        f1 = rep["head_a"]["f1"]
        line = (f"epoch {ep:>2}  train {tr_loss:.4f}  dev {dv_loss:.4f}  "
                f"A: F1={f1:.4f} Pk={rep['head_a']['pk']:.4f} "
                f"WD={rep['head_a']['window_diff']:.4f} @t={thr_a}")
        if args.task == "hier":
            line += (f"  |  B: F1={rep['head_b']['f1']:.4f} "
                     f"Pk={rep['head_b']['pk']:.4f} @t={thr_b}")
        mem = gpu_report(args.device, "  ")
        print(line + f"   [{time.time()-t0:.0f}s]{mem}", flush=True)
        history.append({"epoch": ep, "train_loss": tr_loss, "dev_loss": dv_loss,
                        "thr_a": thr_a, "thr_b": thr_b, **rep})
        # every criterion is turned into higher-is-better; Pk and the loss are
        # error rates, so they are negated
        if args.select_on == "loss":
            score = -dv_loss
        elif args.select_on == "pk":
            score = -(rep["head_a"]["pk"] +
                      (rep["head_b"]["pk"] if args.task == "hier" else 0)) \
                / (2 if args.task == "hier" else 1)
        else:
            score = f1 + (rep["head_b"]["f1"] if args.task == "hier" else 0)
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": ep, "thr_a": thr_a, "thr_b": thr_b}
            torch.save(model.state_dict(), ckpt_path)
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early stop: no improvement in dev {args.select_on} for "
                      f"{stale} epochs (best was epoch {best['epoch']})")
                break

    # ---------------- test, at the DEV-selected threshold
    result = {"run_name": run_name, "args": vars(args),
              "select_on": args.select_on, "epochs_run": len(history),
              "history": history, "best_epoch": best["epoch"]}
    if "test" in loaders:
        model.load_state_dict(torch.load(ckpt_path))
        _, refs, pa, pb = run_epoch(model, loaders["test"], args.task,
                                    args.device, args.mode, pw=pw, w_b=args.w_b)
        result["test"] = evaluate(args.task, refs, pa, pb,
                                  best["thr_a"], best["thr_b"])
        print("\nTEST (epoch %d, thresholds from dev):" % best["epoch"])
        print(json.dumps(result["test"], indent=2))

    with open(res_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {res_path}\n      {ckpt_path}")


if __name__ == "__main__":
    main()
