#!/usr/bin/env python3
"""
Build an Assamese linear text segmentation dataset from the aswiki dump.

Task framing: PARAGRAPH segmentation, not topic segmentation. Boundaries are
paragraph breaks inside a single section, so they mark soft semantic shifts
within one continuous topic -- the Multi-Wiki90k label scheme, not the
Wiki-727K / WikiSection heading scheme.

Output unit: one "document" = one article section, flattened to a sentence
sequence with paragraph breaks removed. The model must put them back.
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
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wikiclean import clean_wikitext, parse_sections, is_boilerplate_heading
from assamese_sent import split_sentences, assamese_ratio, word_count

RE_NS = re.compile(r"^\{[^}]*\}")  # export-0.10 / 0.11 / ... -- strip whatever it is

# ------------------------------------------------------------------ filters
MIN_SENT_WORDS = 3          # a "sentence" shorter than this is fragment/noise
MAX_SENT_WORDS = 120
MIN_PARAS = 2               # need >=2 paragraphs for >=1 boundary
MAX_PARAS = 25
MIN_SENTS = 4               # per section document
MAX_SENTS = 150
MIN_AS_RATIO = 0.70         # section must be genuinely Assamese
MIN_PARA_SENTS = 1

# --- negative (boundary-free) documents -----------------------------------
# A corpus made only of multi-paragraph sections teaches the model that a
# boundary ALWAYS exists somewhere. Real inputs -- a research-article
# introduction, an arbitrary passage -- carry no such guarantee. So a sampled
# fraction of single-paragraph sections is admitted as all-zero documents.
NEG_MIN_SENTS = 4
NEG_LONG = 12               # an unbroken paragraph this long is a suspected
                            # MISSED break, so it is flagged low-confidence


def document_confidence(segment_lengths, labels):
    """Structural confidence in a document's silver labels.

    Wikipedia paragraph breaks are an editor's stylistic choice as much as a
    discourse fact. The doubtful cases are structurally identifiable:

      low    - adjacent boundaries (a one-sentence paragraph stranded between
               two others, usually a formatting accident), or every segment
               <= 2 sentences (a shredded section).
      medium - a one-sentence segment at the start or end. A trailing
               one-sentence paragraph is very often a later addendum.
      high   - no one-sentence segments.

    Train on everything; evaluate on `high` if you want a clean number, and
    report the stratum mix either way rather than silently filtering.
    """
    if len(segment_lengths) == 1:                    # negative document
        return "low" if segment_lengths[0] >= NEG_LONG else "high"
    if any(a == 1 and b == 1 for a, b in zip(labels, labels[1:])):
        return "low"
    if all(l <= 2 for l in segment_lengths):
        return "low"
    if segment_lengths[0] == 1 or segment_lengths[-1] == 1:
        return "medium"
    return "high"

RE_TITLE_BAD = re.compile(
    r"(তালিকা|list of|disambiguation|নিৰদেশক|বছৰ|\(পৃথকীকৰণ\))", re.I
)


def iter_pages(path):
    """Stream (title, text) for main-namespace, non-redirect pages.

    Works with BOTH dump flavours:
      * pages-articles  -- one <revision> per page (what you want)
      * pages-meta-history -- every historical revision per page; we keep only
        the last <text> seen inside each <page>, i.e. the current revision.

    Memory is bounded: <revision> subtrees are cleared as they close, so an
    8 GB history dump with thousands of revisions per page still runs in a few
    hundred MB of RAM. Nothing is ever decompressed to disk.
    """
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as fh:
        title = ns = text = None
        redirect = False
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            tag = RE_NS.sub("", elem.tag)
            if event == "start":
                if tag == "page":
                    title = ns = text = None
                    redirect = False
                continue
            if tag == "title" and title is None:
                title = elem.text          # page title, not a revision field
            elif tag == "ns":
                ns = elem.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                if elem.text:
                    text = elem.text       # last one wins = current revision
            elif tag == "revision":
                elem.clear()               # drop contributor/timestamp/sha1/comment
            elif tag == "page":
                if ns == "0" and not redirect and title and text:
                    yield title, text
                elem.clear()


def section_to_example(article, sec_title, sec_index, paras, negative=False):
    """Turn one section into a segmentation example, or None.

    negative=False -> require >=2 paragraphs (a document WITH boundaries)
    negative=True  -> require exactly 1 paragraph (an all-zero document)
    """
    if is_boilerplate_heading(sec_title):
        return None

    para_sents = []
    for p in paras:
        sents = [
            s for s in split_sentences(p)
            if MIN_SENT_WORDS <= word_count(s) <= MAX_SENT_WORDS
        ]
        if len(sents) >= MIN_PARA_SENTS:
            para_sents.append(sents)

    if negative:
        if len(para_sents) != 1:
            return None
    elif not (MIN_PARAS <= len(para_sents) <= MAX_PARAS):
        return None

    sentences, labels, para_ids = [], [], []
    for pi, sents in enumerate(para_sents):
        for si, s in enumerate(sents):
            sentences.append(s)
            para_ids.append(pi)
            # label 1 = a paragraph break follows this sentence
            labels.append(1 if si == len(sents) - 1 else 0)
    labels[-1] = 0  # no boundary after the final sentence of the document

    lo = NEG_MIN_SENTS if negative else MIN_SENTS
    if not (lo <= len(sentences) <= MAX_SENTS):
        return None

    joined = " ".join(sentences)
    if assamese_ratio(joined) < MIN_AS_RATIO:
        return None

    seglens = [len(s) for s in para_sents]
    return {
        "id": hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16],
        "article": article,
        "section": None if sec_title == "__LEAD__" else sec_title,
        "section_index": sec_index,
        "sentences": sentences,
        "labels": labels,
        "paragraph_ids": para_ids,
        "n_sentences": len(sentences),
        "n_segments": len(para_sents),
        "segment_lengths": seglens,
        "is_negative": bool(negative),
        "confidence": document_confidence(seglens, labels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--neg-ratio", type=float, default=0.15,
        help="boundary-free documents to add, as a fraction of the number of "
             "positive documents (0 disables them)")
    ap.add_argument(
        "--neg-pool", type=int, default=20000,
        help="reservoir size for negative candidates; keeps memory bounded on "
             "a full dump instead of holding every single-paragraph section")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats = Counter()
    by_article = {}
    seen = set()
    rng = random.Random(args.seed)
    neg_pool, neg_seen = [], 0        # reservoir sample (Algorithm R)

    for i, (title, wikitext) in enumerate(iter_pages(args.dump)):
        stats["pages"] += 1
        if args.limit and stats["pages"] > args.limit:
            break
        if RE_TITLE_BAD.search(title or ""):
            stats["skip_title"] += 1
            continue
        try:
            cleaned = clean_wikitext(wikitext)
            sections = parse_sections(cleaned)
        except Exception:
            stats["parse_error"] += 1
            continue

        kept = []
        for si, (sec_title, _lvl, paras) in enumerate(sections):
            stats["sections_seen"] += 1
            ex = section_to_example(title, sec_title, si, paras)
            if ex is None:
                stats["sections_filtered"] += 1
                # a single-paragraph section is a candidate NEGATIVE
                if args.neg_ratio > 0:
                    neg = section_to_example(title, sec_title, si, paras,
                                             negative=True)
                    if neg is not None and neg["id"] not in seen:
                        neg_seen += 1
                        if len(neg_pool) < args.neg_pool:
                            neg_pool.append(neg)
                        else:                      # reservoir replacement
                            j = rng.randrange(neg_seen)
                            if j < args.neg_pool:
                                neg_pool[j] = neg
                continue
            if ex["id"] in seen:
                stats["dup"] += 1
                continue
            seen.add(ex["id"])
            kept.append(ex)
        if kept:
            by_article[title] = kept
            stats["articles_kept"] += 1
            stats["examples"] += len(kept)

        if stats["pages"] % 5000 == 0:
            print(
                f"  {stats['pages']:>6} pages -> {stats['examples']:>6} examples",
                flush=True,
            )

    # ---- add the sampled boundary-free documents ---------------------------
    stats["neg_candidates"] = neg_seen
    if args.neg_ratio > 0 and neg_pool:
        target = int(round(args.neg_ratio * stats["examples"]))
        chosen = rng.sample(neg_pool, min(target, len(neg_pool)))
        for neg in chosen:
            if neg["id"] in seen:
                continue
            seen.add(neg["id"])
            by_article.setdefault(neg["article"], []).append(neg)
            stats["negatives"] += 1
        if len(neg_pool) < target:
            print(f"  note: wanted {target} negatives, pool held only "
                  f"{len(neg_pool)} (raise --neg-pool)")

    # split by ARTICLE so no article leaks across splits
    arts = sorted(by_article)
    rng.shuffle(arts)
    n = len(arts)
    n_test = max(1, int(0.10 * n))
    n_dev = max(1, int(0.10 * n))
    splits = {
        "test": arts[:n_test],
        "dev": arts[n_test : n_test + n_dev],
        "train": arts[n_test + n_dev :],
    }

    summary = {"filters": {
        "min_sent_words": MIN_SENT_WORDS, "max_sent_words": MAX_SENT_WORDS,
        "min_paras": MIN_PARAS, "max_paras": MAX_PARAS,
        "min_sents": MIN_SENTS, "max_sents": MAX_SENTS,
        "min_assamese_ratio": MIN_AS_RATIO,
    }, "counts": dict(stats), "splits": {}}

    for name, titles in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        ns = nb = nsent = nneg = 0
        conf = Counter()
        with open(path, "w", encoding="utf-8") as fh:
            for t in titles:
                for ex in by_article[t]:
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    ns += 1
                    nb += sum(ex["labels"])
                    nsent += ex["n_sentences"]
                    nneg += ex["is_negative"]
                    conf[ex["confidence"]] += 1
        summary["splits"][name] = {
            "articles": len(titles), "documents": ns,
            "sentences": nsent, "boundaries": nb,
            "boundary_rate": round(nb / nsent, 4) if nsent else 0,
            "negatives": nneg,
            "confidence": {k: conf[k] for k in ("high", "medium", "low")},
        }
        print(f"{name:>5}: {ns:>6} docs ({nneg:>5} neg)  {nsent:>7} sents  "
              f"{nb:>6} bounds  conf H/M/L="
              f"{conf['high']}/{conf['medium']}/{conf['low']}")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
