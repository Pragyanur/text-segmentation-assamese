#!/usr/bin/env python3
"""
Build AsParaSeg-H -- the ARTICLE-level, three-way hierarchical dataset.

This is a separate corpus from the section-level paragraph dataset built by
build_dataset.py. Different unit, different labels, different model.

  build_dataset.py  unit = one SECTION   labels = {0, 1}     fine model
  build_hier.py     unit = one ARTICLE   labels = {0, 1, 2}  hierarchical model

Labels are ORDINAL and NESTED:

    0  no boundary after this sentence
    1  paragraph boundary
    2  section boundary  (which is ALSO a paragraph boundary -- a section
                          never starts mid-paragraph, so 2 implies 1)

Train cumulative targets from them: y_a = (label >= 1), y_b = (label >= 2).

WHY ALL PROSE SECTIONS ARE KEPT
build_dataset.py discards single-paragraph and short sections, which is
harmless when each section is its own document. Here it is NOT: dropping a
middle section splices two non-adjacent sections together and invents a
section boundary the text no longer supports. So every prose section is kept,
and length filters apply to the assembled article instead. Articles where a
boilerplate section still had to be dropped from the middle are flagged
`spliced: true` so you can exclude them.
"""
import argparse
import bz2
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wikiclean import clean_wikitext, parse_sections, is_boilerplate_heading
from assamese_sent import split_sentences, assamese_ratio, word_count
from build_dataset import iter_pages, RE_TITLE_BAD

# ---- sentence-level filters (identical to the fine dataset) ---------------
MIN_SENT_WORDS = 3
MAX_SENT_WORDS = 120

# ---- article-level filters ------------------------------------------------
ART_MIN_SENTS = 8       # article-level documents are longer than sections
ART_MAX_SENTS = 300     # cap on transformer positions; truncated at a SECTION
                        # edge so the document never ends mid-section
MIN_AS_RATIO = 0.70
MIN_SECTIONS = 1        # 1-section articles are legitimate head-B negatives


def sentences_of(paras):
    """[paragraph text] -> [[sentence, ...], ...], dropping empty paragraphs."""
    out = []
    for p in paras:
        ss = [s for s in split_sentences(p)
              if MIN_SENT_WORDS <= word_count(s) <= MAX_SENT_WORDS]
        if ss:
            out.append(ss)
    return out


def article_to_example(title, sections):
    """Assemble one article into a 3-way labelled document, or None."""
    kept, spliced = [], False
    dropped_since_kept = False
    for sec_title, _lvl, paras in sections:
        if is_boilerplate_heading(sec_title):
            if kept:
                dropped_since_kept = True     # only matters if more follows
            continue
        ps = sentences_of(paras)
        if not ps:
            if kept:
                dropped_since_kept = True
            continue
        if dropped_since_kept:
            spliced = True                    # a gap sits before this section
            dropped_since_kept = False
        kept.append((sec_title, ps))

    if len(kept) < MIN_SECTIONS:
        return None

    # truncate at a SECTION edge so the document never stops mid-section
    total, fitted = 0, []
    for sec_title, ps in kept:
        n = sum(len(p) for p in ps)
        if total + n > ART_MAX_SENTS:
            break
        fitted.append((sec_title, ps))
        total += n
    if not fitted or total < ART_MIN_SENTS:
        return None
    truncated = len(fitted) < len(kept)

    sentences, labels, para_ids, sec_ids = [], [], [], []
    gpara = 0
    for si, (_sec_title, ps) in enumerate(fitted):
        last_sec = si == len(fitted) - 1
        for pi, sents in enumerate(ps):
            last_para = pi == len(ps) - 1
            for k, s in enumerate(sents):
                sentences.append(s)
                para_ids.append(gpara)
                sec_ids.append(si)
                end_para = k == len(sents) - 1
                end_sec = end_para and last_para
                # 2 implies 1: a section end is also a paragraph end
                labels.append(2 if (end_sec and not last_sec)
                              else 1 if end_para else 0)
            gpara += 1
    labels[-1] = 0            # never a boundary after the final sentence

    joined = " ".join(sentences)
    if assamese_ratio(joined) < MIN_AS_RATIO:
        return None

    return {
        "id": hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16],
        "article": title,
        "sentences": sentences,
        "labels": labels,
        "paragraph_ids": para_ids,
        "section_ids": sec_ids,
        # heading sequence -- diff this against English Wikipedia to test how
        # much of aswiki's section structure is inherited by translation
        "section_titles": [t for t, _ in fitted],
        "n_sentences": len(sentences),
        "n_sections": len(fitted),
        "n_paragraphs": gpara,
        "n_section_boundaries": sum(1 for l in labels if l == 2),
        "n_paragraph_boundaries": sum(1 for l in labels if l == 1),
        "spliced": spliced,
        "truncated": truncated,
    }


def validate(ex):
    """Invariants the training code will silently rely on."""
    L, sec, par = ex["labels"], ex["section_ids"], ex["paragraph_ids"]
    n = len(ex["sentences"])
    assert len(L) == len(sec) == len(par) == n, "ragged fields"
    assert L[-1] == 0, "boundary on final sentence"
    assert set(L) <= {0, 1, 2}, "bad label value"
    for i in range(n - 1):
        # every section change must be marked 2, and only there
        assert (sec[i] != sec[i + 1]) == (L[i] == 2), "section id / label 2 mismatch"
        # 2 implies 1: a label-2 position must also end a paragraph
        if L[i] == 2:
            assert par[i] != par[i + 1], "label 2 not at a paragraph end"
        if L[i] == 1:
            assert par[i] != par[i + 1], "label 1 not at a paragraph end"
        if L[i] == 0:
            assert par[i] == par[i + 1], "label 0 inside a paragraph break"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="dataset_hier")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop-spliced", action="store_true",
                    help="exclude articles where a middle section was removed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats = Counter()
    docs, seen = [], set()

    for title, wikitext in iter_pages(args.dump):
        stats["pages"] += 1
        if args.limit and stats["pages"] > args.limit:
            break
        if RE_TITLE_BAD.search(title or ""):
            stats["skip_title"] += 1
            continue
        try:
            sections = parse_sections(clean_wikitext(wikitext))
        except Exception:
            stats["parse_error"] += 1
            continue

        ex = article_to_example(title, sections)
        if ex is None:
            stats["articles_filtered"] += 1
            continue
        if ex["id"] in seen:
            stats["dup"] += 1
            continue
        if args.drop_spliced and ex["spliced"]:
            stats["dropped_spliced"] += 1
            continue
        validate(ex)
        seen.add(ex["id"])
        docs.append(ex)
        stats["spliced"] += ex["spliced"]
        stats["truncated"] += ex["truncated"]

        if stats["pages"] % 5000 == 0:
            print(f"  {stats['pages']:>6} pages -> {len(docs):>6} articles",
                  flush=True)

    stats["documents"] = len(docs)
    rng = random.Random(args.seed)
    rng.shuffle(docs)                       # unit IS the article: no leakage
    n = len(docs)
    n_test = max(1, int(0.10 * n))
    n_dev = max(1, int(0.10 * n))
    splits = {"test": docs[:n_test],
              "dev": docs[n_test:n_test + n_dev],
              "train": docs[n_test + n_dev:]}

    summary = {"unit": "article", "labels": {"0": "none", "1": "paragraph",
                                            "2": "section"},
               "filters": {"art_min_sents": ART_MIN_SENTS,
                           "art_max_sents": ART_MAX_SENTS,
                           "min_sent_words": MIN_SENT_WORDS,
                           "max_sent_words": MAX_SENT_WORDS,
                           "min_assamese_ratio": MIN_AS_RATIO},
               "counts": dict(stats), "splits": {}}

    for name, group in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        c = Counter()
        with open(path, "w", encoding="utf-8") as fh:
            for ex in group:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                for l in ex["labels"]:
                    c[l] += 1
        tot = sum(c.values())
        summary["splits"][name] = {
            "documents": len(group), "sentences": tot,
            "label_0": c[0], "label_1": c[1], "label_2": c[2],
            "rate_1": round(c[1] / tot, 4) if tot else 0,
            "rate_2": round(c[2] / tot, 4) if tot else 0,
        }
        print(f"{name:>5}: {len(group):>6} articles {tot:>7} sents  "
              f"0/1/2 = {c[0]}/{c[1]}/{c[2]}")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
