#!/usr/bin/env python3
"""Statistics and quality report for the Assamese segmentation dataset."""
import json
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assamese_sent import word_count


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def report(outdir):
    allex = []
    print(f"{'split':>6} {'docs':>7} {'sents':>8} {'bounds':>7} {'rate':>6} "
          f"{'sents/doc':>10} {'segs/doc':>9} {'seglen':>7}")
    print("-" * 70)
    for name in ("train", "dev", "test"):
        p = os.path.join(outdir, f"{name}.jsonl")
        if not os.path.exists(p):
            continue
        ex = load(p)
        allex += ex
        ns = sum(e["n_sentences"] for e in ex)
        nb = sum(sum(e["labels"]) for e in ex)
        seglens = [l for e in ex for l in e["segment_lengths"]]
        print(f"{name:>6} {len(ex):>7} {ns:>8} {nb:>7} {nb/ns:>6.3f} "
              f"{ns/len(ex):>10.1f} {st.mean(e['n_segments'] for e in ex):>9.2f} "
              f"{st.mean(seglens):>7.2f}")

    print()
    seglens = [l for e in allex for l in e["segment_lengths"]]
    words = [word_count(s) for e in allex for s in e["sentences"]]
    print(f"TOTAL documents      : {len(allex):,}")
    print(f"TOTAL sentences      : {sum(e['n_sentences'] for e in allex):,}")
    print(f"TOTAL boundaries     : {sum(sum(e['labels']) for e in allex):,}")
    print(f"TOTAL words          : {sum(words):,}")
    print(f"Distinct articles    : {len(set(e['article'] for e in allex)):,}")
    print()
    print(f"Segment length (sentences): mean={st.mean(seglens):.2f} "
          f"median={st.median(seglens)} p90={sorted(seglens)[int(.9*len(seglens))]} "
          f"max={max(seglens)}")
    print(f"Sentence length (words)   : mean={st.mean(words):.1f} "
          f"median={st.median(words)} p90={sorted(words)[int(.9*len(words))]}")
    print()
    print("Segment-length histogram (sentences per paragraph):")
    c = Counter(seglens)
    for k in sorted(c)[:12]:
        bar = "#" * int(60 * c[k] / max(c.values()))
        print(f"  {k:>3}: {c[k]:>6} {bar}")

    print()
    print("Documents by section type:")
    sec = Counter("lead" if e["section"] is None else "named" for e in allex)
    for k, v in sec.most_common():
        print(f"  {k:>6}: {v:,}")

    print()
    print("Most common section headings:")
    for k, v in Counter(
        e["section"] for e in allex if e["section"]
    ).most_common(12):
        print(f"  {v:>5}  {k}")

    # ---- label-confidence strata
    print()
    print("Confidence strata (structural reliability of the silver labels):")
    conf = Counter(e.get("confidence", "?") for e in allex)
    cb = Counter()
    for e in allex:
        cb[e.get("confidence", "?")] += sum(e["labels"])
    nb_all = sum(cb.values()) or 1
    print(f"  {'stratum':<8} {'docs':>7} {'%docs':>7} {'bounds':>8} {'%bounds':>8}")
    for k in ("high", "medium", "low", "?"):
        if not conf[k]:
            continue
        print(f"  {k:<8} {conf[k]:>7} {100*conf[k]/len(allex):>6.1f}% "
              f"{cb[k]:>8} {100*cb[k]/nb_all:>7.1f}%")

    neg = [e for e in allex if e.get("is_negative")]
    if neg:
        print()
        print(f"Boundary-free (negative) documents: {len(neg):,} "
              f"({100*len(neg)/len(allex):.1f}% of documents)")
        print(f"  their sentences: {sum(e['n_sentences'] for e in neg):,} "
              f"(all labelled 0)")

    # ---- trivial-baseline sanity check: is the boundary signal learnable at all?
    print()
    print("Baselines (accuracy of trivial predictors):")
    tot = sum(len(e["labels"]) for e in allex)
    pos = sum(sum(e["labels"]) for e in allex)
    print(f"  predict-all-zero accuracy : {(tot-pos)/tot:.4f}")
    print(f"  boundary base rate        : {pos/tot:.4f}")
    print(f"  -> a model must beat {(tot-pos)/tot:.4f} accuracy to be useful")
    print("  NOTE: report P/R/F1 and Pk or WindowDiff, never raw accuracy --")
    print("        the majority class alone already scores the number above.")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "dataset")
