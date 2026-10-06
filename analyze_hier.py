#!/usr/bin/env python3
"""Statistics for AsParaSeg-H (article-level, 3-way labels).

Also prints the pos_weight values to hand straight to hierarchical_loss(),
and the two baselines each head has to beat.
"""
import json
import os
import statistics as st
import sys
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def report(outdir):
    allex, per = [], {}
    for name in ("train", "dev", "test"):
        p = os.path.join(outdir, f"{name}.jsonl")
        if os.path.exists(p):
            per[name] = load(p)
            allex += per[name]
    if not allex:
        print(f"no jsonl files in {outdir}")
        return

    print(f"{'split':>6} {'articles':>9} {'sents':>8} {'label 0':>9} "
          f"{'label 1':>9} {'label 2':>8} {'rate 1':>7} {'rate 2':>7}")
    print("-" * 72)
    for name, ex in per.items():
        c = Counter(l for e in ex for l in e["labels"])
        t = sum(c.values())
        print(f"{name:>6} {len(ex):>9} {t:>8} {c[0]:>9} {c[1]:>9} {c[2]:>8} "
              f"{c[1]/t:>7.4f} {c[2]/t:>7.4f}")

    c = Counter(l for e in allex for l in e["labels"])
    tot = sum(c.values())
    print()
    print(f"TOTAL articles   : {len(allex):,}")
    print(f"TOTAL sentences  : {tot:,}")
    print(f"  label 0 (none)     : {c[0]:>8,}  {100*c[0]/tot:>5.2f}%")
    print(f"  label 1 (paragraph): {c[1]:>8,}  {100*c[1]/tot:>5.2f}%")
    print(f"  label 2 (section)  : {c[2]:>8,}  {100*c[2]/tot:>5.2f}%")

    # ---- cumulative targets: what the two heads actually see
    n_a = c[1] + c[2]          # label >= 1
    n_b = c[2]                 # label >= 2
    print()
    print("Cumulative targets (what the heads are trained on):")
    print(f"  head A  y=(label>=1) positives: {n_a:>8,}  {100*n_a/tot:>5.2f}%")
    print(f"  head B  y=(label>=2) positives: {n_b:>8,}  {100*n_b/tot:>5.2f}%")
    print()
    print("Class imbalance -> pass these to hierarchical_loss():")
    print(f"  pos_weight_a = {(tot-n_a)/max(n_a,1):.2f}")
    print(f"  pos_weight_b = {(tot-n_b)/max(n_b,1):.2f}")
    print()
    print("Baselines each head must beat (all-negative predictor):")
    print(f"  head A all-zero accuracy: {(tot-n_a)/tot:.4f}")
    print(f"  head B all-zero accuracy: {(tot-n_b)/tot:.4f}")
    print("  head B's baseline is the dangerous one -- without pos_weight it")
    print("  collapses to always-zero and still scores that number.")

    # ---- structure
    secs = [e["n_sections"] for e in allex]
    paras = [e["n_paragraphs"] for e in allex]
    sents = [e["n_sentences"] for e in allex]
    print()
    print(f"sections per article  : mean={st.mean(secs):.2f} median={st.median(secs)} "
          f"p90={sorted(secs)[int(.9*len(secs))]} max={max(secs)}")
    print(f"paragraphs per article: mean={st.mean(paras):.2f} median={st.median(paras)}")
    print(f"sentences per article : mean={st.mean(sents):.1f} median={st.median(sents)} "
          f"p90={sorted(sents)[int(.9*len(sents))]} max={max(sents)}")

    single = sum(1 for e in allex if e["n_sections"] == 1)
    print(f"single-section articles (head-B negatives): {single:,} "
          f"({100*single/len(allex):.1f}%)")

    print()
    print("Sections per article histogram:")
    h = Counter(secs)
    mx = max(h.values())
    for k in sorted(h)[:12]:
        print(f"  {k:>3}: {h[k]:>6} {'#'*int(50*h[k]/mx)}")

    # ---- segment sizes at the two granularities
    para_len, sec_len = [], []
    for e in allex:
        pc = Counter(e["paragraph_ids"]); para_len += list(pc.values())
        sc = Counter(e["section_ids"]);   sec_len  += list(sc.values())
    print()
    print(f"paragraph length (sentences): mean={st.mean(para_len):.2f} "
          f"median={st.median(para_len)}")
    print(f"section length   (sentences): mean={st.mean(sec_len):.2f} "
          f"median={st.median(sec_len)}")

    # ---- data hygiene
    sp = sum(1 for e in allex if e.get("spliced"))
    tr = sum(1 for e in allex if e.get("truncated"))
    print()
    print(f"spliced articles (middle section removed): {sp:,} "
          f"({100*sp/len(allex):.1f}%)  -- rebuild with --drop-spliced to exclude")
    print(f"truncated articles (hit the sentence cap): {tr:,} "
          f"({100*tr/len(allex):.1f}%)")

    # ---- the enwiki-inheritance check
    print()
    print("Most common section-heading sequences (translation-inheritance check):")
    seqs = Counter(" > ".join(e["section_titles"][:4]) for e in allex
                   if e["n_sections"] >= 2)
    for s, v in seqs.most_common(8):
        print(f"  {v:>5}  {s}")
    print("  If these read as literal translations of English Wikipedia's")
    print("  standard headings, the section level reflects enwiki editorial")
    print("  convention rather than Assamese discourse. Verify before claiming")
    print("  otherwise.")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "dataset_hier")
