#!/usr/bin/env python3
"""
Encode every sentence once with MuRIL and cache the vectors.

Run this BEFORE train.py --mode frozen. Stage 1 (the sentence encoder)
dominates the cost of this model, and when it is frozen its output never
changes, so encoding the corpus once and training only the context
transformer is both faster and cheaper -- on the article-level data it is the
difference between needing a large GPU and running on a modest one.

Do NOT use these caches for --mode finetune: there the encoder's weights move
every step, so its output must be recomputed every step.

    python precompute.py dataset_hier --out vecs_hier
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from seg_data import SegDataset, make_text_collate, BucketBatchSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="dataset dir with train/dev/test.jsonl")
    ap.add_argument("--out", default=None, help="default: <data>/vectors")
    ap.add_argument("--encoder", default="google/muril-base-cased")
    ap.add_argument("--task", choices=["fine", "hier"], default="fine")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = args.out or os.path.join(args.data, "vectors")
    os.makedirs(out, exist_ok=True)

    from transformers import AutoTokenizer
    from model import SentenceEncoder

    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = SentenceEncoder(args.encoder, freeze=True).to(args.device).eval()

    for split in ("train", "dev", "test"):
        path = os.path.join(args.data, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        ds = SegDataset(path, task=args.task)
        dl = DataLoader(
            ds, collate_fn=make_text_collate(tok, args.max_tokens),
            batch_sampler=BucketBatchSampler([it["n"] for it in ds],
                                             args.batch_size, shuffle=False),
        )
        cache = {}
        with torch.no_grad():
            for bi, b in enumerate(dl):
                v = enc(b["input_ids"].to(args.device),
                        b["attention_mask"].to(args.device)).cpu()
                for i, (doc_id, n) in enumerate(zip(b["ids"], b["lengths"])):
                    cache[doc_id] = v[i, :n].clone()
                if bi % 50 == 0:
                    print(f"  {split}: {len(cache)}/{len(ds)} docs", flush=True)
        torch.save(cache, os.path.join(out, f"{split}.pt"))
        print(f"{split}: cached {len(cache)} docs -> {out}/{split}.pt")


if __name__ == "__main__":
    main()
