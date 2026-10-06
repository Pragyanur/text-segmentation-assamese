# Hierarchical Linear Text Segmentation for Assamese

Code, data-building scripts and paper for linear text segmentation in Assamese:
splitting a document into contiguous segments of closely related sentences.

The repository introduces two silver-annotated corpora built from the Assamese
Wikipedia dump, and two supervised segmenters built on MuRIL sentence
representations.

## Datasets

| Corpus | Unit of input | Labels |
|---|---|---|
| **AsParaSeg** | one article section | paragraph boundaries (binary) |
| **AsParaSeg-H** | one whole article | nested: 0 = none, 1 = paragraph, 2 = section |

- Source: `aswiki-latest-pages-articles` (Assamese Wikipedia).
- Boundaries come from the article's own paragraph breaks and section headings,
  so the annotation is free but silver.
- Every AsParaSeg document carries a label-confidence stratum
  (high / medium / low) based on structural signs of unreliable breaks.
- Both corpora are split 80/10/10 **by article**, so no article crosses splits.

| | Documents | Sentences | Boundaries |
|---|---|---|---|
| AsParaSeg | 38,204 (incl. 4,978 negatives) | 413,618 | 72,757 |
| AsParaSeg-H | 21,401 | 599,521 | 140,825 (74,873 paragraph + 65,952 section) |

## Models

Both models share the same pipeline:

1. **Sentence encoder** — MuRIL base cased, mean-pooled over non-padding tokens,
   64 wordpieces max. Used frozen or with its top 1–3 layers fine-tuned.
2. **Context transformer** — 2 pre-norm layers over the sequence of sentence
   vectors, with learned position embeddings over sentence index.
3. **Output heads**
   - *Fine-grained segmenter*: one head, paragraph boundaries.
   - *Hierarchical segmenter*: two cumulative (ordinal) heads predicting
     P(ℓ ≥ 1) and P(ℓ ≥ 2), decoded monotonically.

Decision thresholds are swept on the development split and applied unchanged
to test. Baselines are a no-boundary predictor, lexical TextTiling, and
TextTiling over frozen MuRIL embeddings.

## Repository layout

```
build_dataset.py     build AsParaSeg from the Wikipedia dump
build_hier.py        build AsParaSeg-H
seg_data.py          data loading and label conventions
precompute.py        cache frozen MuRIL sentence vectors
model.py             context transformer, output heads, gap modes
train.py             training and evaluation
gap_ablation.py      sweep and tabulate the --gap-mode ablation
notes/               design notes (e.g. gap-representation.md)
paper/               LaTeX source, tables and bibliography
```

## Usage

Example: the gap-representation ablation on cached frozen vectors.

```bash
python gap_ablation.py run --data dataset --vectors vecs_fine \
    --task fine --out runs/gapabl --seeds 13 17 23
python gap_ablation.py table --out runs/gapabl                  # dev
python gap_ablation.py table --out runs/gapabl --split test --latex
```

`train.py --task {fine,hier}` selects the model; see each script's `--help`
for the full set of options.

## Evaluation

Results are reported with P_k and WindowDiff (error rates, lower is better),
plus exact-match boundary precision, recall, F1 and the ratio of predicted to
reference boundaries. Accuracy over positions is not used, since a
no-boundary predictor already scores about 80% on it.

## Status

Work in progress. Reported numbers are single runs at seed 13, and training
was capped before development F1 had converged, so current results should be
treated as preliminary.

## License and citation

To be added. The dataset and code will be released publicly with the paper.
