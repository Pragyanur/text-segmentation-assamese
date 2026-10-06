"""
Segmentation metrics: P/R/F1, Pk and WindowDiff.

Convention used everywhere in this project:
    labels[i] == 1  means a boundary falls AFTER sentence i,
                    i.e. between sentence i and sentence i+1.
    labels[-1] is always 0 (no boundary after the final sentence).

Why Pk and WindowDiff and not accuracy: boundaries are ~19% of positions, so
predicting all-zero already scores ~81% accuracy. Pk and WindowDiff are
error rates -- LOWER IS BETTER, 0.0 is perfect -- and they give partial credit
for a boundary predicted near the right place instead of exactly on it, which
plain F1 does not.
"""
from collections import Counter


# --------------------------------------------------------------- boundary F1
def prf1(ref, hyp):
    """Exact-position precision / recall / F1 over boundary positions."""
    tp = sum(1 for r, h in zip(ref, hyp) if r == 1 and h == 1)
    fp = sum(1 for r, h in zip(ref, hyp) if r == 0 and h == 1)
    fn = sum(1 for r, h in zip(ref, hyp) if r == 1 and h == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def prf1_corpus(refs, hyps):
    """Micro-averaged over a corpus of documents."""
    tp = fp = fn = 0
    for ref, hyp in zip(refs, hyps):
        for r, h in zip(ref, hyp):
            tp += r == 1 and h == 1
            fp += r == 0 and h == 1
            fn += r == 1 and h == 0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ------------------------------------------------------------- Pk / WindowDiff
def _segment_ids(labels):
    """boundary-after-i labels -> a segment id per sentence."""
    ids, c = [], 0
    for b in labels:
        ids.append(c)
        if b:
            c += 1
    return ids


def default_k(refs):
    """Standard choice: half the mean reference segment length, in sentences."""
    n_sent = sum(len(r) for r in refs)
    n_seg = sum(sum(r) + 1 for r in refs)
    if n_seg == 0:
        return 2
    return max(2, int(round(n_sent / (2.0 * n_seg))))


def pk(ref, hyp, k):
    """Beeferman et al. Pk. Lower is better; 0.0 is perfect.

    Slides a k-sentence window; counts positions where reference and
    hypothesis disagree about whether the two ends sit in the same segment.
    """
    n = len(ref)
    if n <= k:
        return 0.0
    sr, sh = _segment_ids(ref), _segment_ids(hyp)
    err = 0
    for i in range(n - k):
        same_r = sr[i] == sr[i + k]
        same_h = sh[i] == sh[i + k]
        err += same_r != same_h
    return err / (n - k)


def window_diff(ref, hyp, k):
    """Pevzner & Hearst WindowDiff. Lower is better; 0.0 is perfect.

    Compares the NUMBER of boundaries inside each window, so it penalises
    both missed and spurious boundaries and is not fooled by segment-size
    distribution the way Pk can be.
    """
    n = len(ref)
    if n <= k:
        return 0.0
    err = 0
    for i in range(n - k):
        b_r = sum(ref[i:i + k])
        b_h = sum(hyp[i:i + k])
        err += b_r != b_h
    return err / (n - k)


def pk_wd_corpus(refs, hyps, k=None):
    """Document-length-weighted Pk and WindowDiff over a corpus."""
    if k is None:
        k = default_k(refs)
    num_pk = num_wd = den = 0.0
    for ref, hyp in zip(refs, hyps):
        n = len(ref)
        if n <= k:
            continue
        w = n - k
        num_pk += pk(ref, hyp, k) * w
        num_wd += window_diff(ref, hyp, k) * w
        den += w
    if den == 0:
        return 0.0, 0.0, k
    return num_pk / den, num_wd / den, k


# ------------------------------------------------------------ threshold search
def best_threshold(refs, probs, lo=0.05, hi=0.95, step=0.01):
    """Pick the decision threshold that maximises boundary F1 on dev.

    Never tune this on test. With a 19% positive rate the F1-optimal
    threshold is usually well below 0.5, and leaving it at 0.5 costs real
    recall.
    """
    best = (0.5, 0.0)
    t = lo
    while t <= hi + 1e-9:
        hyps = [[1 if p >= t else 0 for p in doc] for doc in probs]
        _, _, f = prf1_corpus(refs, hyps)
        if f > best[1]:
            best = (round(t, 4), f)
        t += step
    return best


def summarize(refs, hyps, probs=None, name=""):
    """One-line evaluation record."""
    p, r, f = prf1_corpus(refs, hyps)
    p_k, wd, k = pk_wd_corpus(refs, hyps)
    out = {"name": name, "precision": round(p, 4), "recall": round(r, 4),
           "f1": round(f, 4), "pk": round(p_k, 4), "window_diff": round(wd, 4),
           "k": k}
    n_ref = sum(sum(x) for x in refs)
    n_hyp = sum(sum(x) for x in hyps)
    out["ref_boundaries"] = n_ref
    out["pred_boundaries"] = n_hyp
    return out
