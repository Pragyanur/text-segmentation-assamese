# Hierarchical Linear Text Segmentation for Assamese

Code, data-building scripts and paper for text segmentation in Assamese:
splitting a document into contiguous segments of closely related sentences.

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

## Evaluation

Results are reported with P_k and WindowDiff (error rates, lower is better),
plus exact-match boundary precision, recall, F1 and the ratio of predicted to
reference boundaries. Accuracy over positions is not used, since a
no-boundary predictor already scores about 80% on it.
