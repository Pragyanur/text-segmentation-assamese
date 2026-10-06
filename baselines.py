#!/usr/bin/env python3
"""
Unsupervised tiling baselines for AsParaSeg.

    # lexical TextTiling (Hearst 1997) -- bag of words
    python baselines.py dataset --task fine

    # semantic tiling (Solbiati et al. 2021) -- the SAME algorithm over
    # MuRIL sentence embeddings instead of word counts
    python baselines.py dataset --task fine --vectors vecs_fine

    # both, on the article-level corpus with two boundary levels
    python baselines.py dataset_hier --task hier --vectors vecs_hier

ONE algorithm, TWO representations. Everything except the block
representation is held constant -- block size, depth scoring, thresholding
and hyperparameter tuning are identical -- so the difference between the two
rows isolates exactly one thing: what the semantic representation buys over
lexical overlap. That is the first question the Results section has to answer,
and a baseline tuned less carefully than the model would not answer it
honestly.

Hyperparameters (block size w, threshold coefficient c) are tuned on DEV by
grid search and applied unchanged to TEST -- the same protocol the neural
models get for their decision threshold.
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import summarize


# ------------------------------------------------------------------ tokenise
RE_TOK = re.compile(r"[^\s।॥৷.,;:!?\"'()\[\]{}—–\-]+")


def tokenize(text):
    """Whitespace/punctuation tokenisation. No stemming: no Assamese stemmer
    is available, which is precisely the weakness of the lexical baseline in a
    morphologically rich language -- one lemma appears in many surface forms
    and the overlap signal is diluted."""
    return RE_TOK.findall(text.lower())


# ------------------------------------------------------------------ similarity
def cosine_counters(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    if num == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb)


def sim_lexical(sentences, w):
    """Cosine between bag-of-words of the w sentences either side of each gap."""
    n = len(sentences)
    counts = [Counter(tokenize(s)) for s in sentences]
    sims = []
    for i in range(n - 1):
        left = Counter()
        for j in range(max(0, i - w + 1), i + 1):
            left.update(counts[j])
        right = Counter()
        for j in range(i + 1, min(n, i + 1 + w)):
            right.update(counts[j])
        sims.append(cosine_counters(left, right))
    return sims


def sim_semantic(vecs, w):
    """Cosine between mean-pooled sentence embeddings either side of each gap.
    vecs is an (n, d) tensor."""
    import torch
    n = vecs.shape[0]
    sims = []
    for i in range(n - 1):
        l = vecs[max(0, i - w + 1): i + 1].mean(0)
        r = vecs[i + 1: min(n, i + 1 + w)].mean(0)
        sims.append(float(torch.nn.functional.cosine_similarity(
            l.unsqueeze(0), r.unsqueeze(0)).item()))
    return sims


# ------------------------------------------------------------------ tiling
def depth_scores(sims):
    """Hearst's depth score: how deep a valley is relative to the peaks on
    either side of it. A shallow dip in a generally low-similarity region is
    not a boundary; a deep one in a high-similarity region is."""
    n = len(sims)
    out = [0.0] * n
    for i in range(n):
        j = i
        while j > 0 and sims[j - 1] >= sims[j]:
            j -= 1
        lpeak = sims[j]
        j = i
        while j < n - 1 and sims[j + 1] >= sims[j]:
            j += 1
        rpeak = sims[j]
        out[i] = (lpeak - sims[i]) + (rpeak - sims[i])
    return out


def boundaries_from_depth(depth, n_sent, c):
    """Boundaries at local maxima of the depth score above mean + c*std.

    Hearst's original cutoff is mean - std/2, i.e. c = -0.5; we tune c on dev
    instead of fixing it, so the baseline is not handicapped."""
    labels = [0] * n_sent
    if not depth:
        return labels
    mu = sum(depth) / len(depth)
    var = sum((d - mu) ** 2 for d in depth) / max(len(depth), 1)
    sd = math.sqrt(var)
    cutoff = mu + c * sd
    for i, d in enumerate(depth):
        if d <= cutoff:
            continue
        if i > 0 and depth[i - 1] > d:          # keep only local maxima, so
            continue                            # boundaries are not adjacent
        if i < len(depth) - 1 and depth[i + 1] > d:
            continue
        labels[i] = 1
    labels[-1] = 0                              # never after the last sentence
    return labels


def predict(doc_sims, n_sent, c):
    return boundaries_from_depth(depth_scores(doc_sims), n_sent, c)


# ------------------------------------------------------------------ driver
def load_split(path, task):
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            docs.append({"id": e["id"], "sentences": e["sentences"],
                         "labels": e["labels"], "n": len(e["sentences"])})
    return docs


def compute_sims(docs, w, vectors=None):
    sims = {}
    for d in docs:
        if d["n"] < 3:
            sims[d["id"]] = []
        elif vectors is not None:
            sims[d["id"]] = sim_semantic(vectors[d["id"]][:d["n"]], w)
        else:
            sims[d["id"]] = sim_lexical(d["sentences"], w)
    return sims


def score_for(docs, sims, c, level):
    refs = [[1 if l >= level else 0 for l in d["labels"]] for d in docs]
    hyps = [predict(sims[d["id"]], d["n"], c) for d in docs]
    return refs, hyps


def tune(docs, vectors, ws, cs, level, objective):
    """Grid search (w, c) on dev. Returns (best_w, best_c, report)."""
    best = None
    for w in ws:
        sims = compute_sims(docs, w, vectors)
        for c in cs:
            refs, hyps = score_for(docs, sims, c, level)
            rep = summarize(refs, hyps)
            val = -rep["pk"] if objective == "pk" else rep["f1"]
            if best is None or val > best[0]:
                best = (val, w, c, rep)
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--task", choices=["fine", "hier"], required=True)
    ap.add_argument("--vectors", default=None,
                    help="dir from precompute.py; enables the SEMANTIC tiling "
                         "baseline. Omit for lexical TextTiling")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tune-on", choices=["pk", "f1"], default="pk",
                    help="must match how the neural model's threshold was "
                         "chosen, or the comparison is not like for like")
    ap.add_argument("--block-sizes", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--c-min", type=float, default=-1.0)
    ap.add_argument("--c-max", type=float, default=2.0)
    ap.add_argument("--c-step", type=float, default=0.1)
    ap.add_argument("--random-baseline", action="store_true",
                    help="also report boundaries placed at random at the dev "
                         "boundary rate")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    kind = "semantic" if args.vectors else "lexical"
    name = f"{os.path.basename(os.path.normpath(args.data))}_{args.task}_" \
           f"tiling-{kind}_tune-{args.tune_on}"
    os.makedirs(args.out, exist_ok=True)
    print(f"run: {name}")

    vectors = {}
    if args.vectors:
        import torch
        for split in ("dev", "test"):
            p = os.path.join(args.vectors, f"{split}.pt")
            if os.path.exists(p):
                vectors[split] = torch.load(p)

    splits = {}
    for split in ("dev", "test"):
        p = os.path.join(args.data, f"{split}.jsonl")
        if os.path.exists(p):
            splits[split] = load_split(p, args.task)
    if "dev" not in splits:
        raise SystemExit(f"need dev.jsonl in {args.data}")

    cs = [round(args.c_min + i * args.c_step, 3)
          for i in range(int((args.c_max - args.c_min) / args.c_step) + 1)]
    levels = [1, 2] if args.task == "hier" else [1]
    result = {"run_name": name, "args": vars(args), "kind": kind, "dev": {},
              "test": {}, "hyperparameters": {}}

    for level in levels:
        head = "head_a" if level == 1 else "head_b"
        dv = splits["dev"]
        w, c, rep_dev = tune(dv, vectors.get("dev"), args.block_sizes, cs,
                             level, args.tune_on)
        result["hyperparameters"][head] = {"block_size": w, "c": c}
        result["dev"][head] = rep_dev
        print(f"  {head}: best on dev  w={w}  c={c:+.2f}  "
              f"F1={rep_dev['f1']:.4f} Pk={rep_dev['pk']:.4f} "
              f"WD={rep_dev['window_diff']:.4f}")

        if "test" in splits:
            te = splits["test"]
            sims = compute_sims(te, w, vectors.get("test"))
            refs, hyps = score_for(te, sims, c, level)
            if level == 2:
                # nesting: a section boundary must also be a paragraph one
                w1 = result["hyperparameters"]["head_a"]["block_size"]
                c1 = result["hyperparameters"]["head_a"]["c"]
                sims1 = compute_sims(te, w1, vectors.get("test"))
                _, hyps1 = score_for(te, sims1, c1, 1)
                hyps = [[a & b for a, b in zip(h1, h2)]
                        for h1, h2 in zip(hyps1, hyps)]
            rep = summarize(refs, hyps,
                            name=f"boundary>={level}")
            result["test"][head] = rep
            print(f"  {head}: TEST  P={rep['precision']:.4f} "
                  f"R={rep['recall']:.4f} F1={rep['f1']:.4f} "
                  f"Pk={rep['pk']:.4f} WD={rep['window_diff']:.4f} "
                  f"(k={rep['k']}, pred {rep['pred_boundaries']} vs ref "
                  f"{rep['ref_boundaries']})")

    # ---- trivial references, for the table's floor
    if "test" in splits:
        te = splits["test"]
        refs = [[1 if l >= 1 else 0 for l in d["labels"]] for d in te]
        zeros = [[0] * d["n"] for d in te]
        result["test"]["no_boundary"] = summarize(refs, zeros,
                                                  name="all-zero")
        print(f"  reference: no-boundary  F1={result['test']['no_boundary']['f1']:.4f} "
              f"Pk={result['test']['no_boundary']['pk']:.4f} "
              f"WD={result['test']['no_boundary']['window_diff']:.4f}")
        if args.random_baseline:
            rng = random.Random(args.seed)
            rate = sum(sum(r) for r in refs) / sum(len(r) for r in refs)
            rnd = [[1 if (i < len(r) - 1 and rng.random() < rate) else 0
                    for i in range(len(r))] for r in refs]
            result["test"]["random"] = summarize(refs, rnd, name="random")
            print(f"  reference: random       "
                  f"F1={result['test']['random']['f1']:.4f} "
                  f"Pk={result['test']['random']['pk']:.4f} "
                  f"WD={result['test']['random']['window_diff']:.4f}")

    path = os.path.join(args.out, f"{name}.result.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
