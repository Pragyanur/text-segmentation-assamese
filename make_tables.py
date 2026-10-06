#!/usr/bin/env python3
"""
Turn result JSONs into the LaTeX table bodies.

    python make_tables.py runs --task fine
    python make_tables.py runs --task hier

Reads every *.result.json in the directory, works out whether each is a
baseline (from baselines.py) or a model run (from train.py), and prints rows
ready to paste between the \\midrule lines of tables.tex.

Transcribing twenty numbers by hand is how a paper ends up with a figure that
does not match its own JSON. Regenerate instead of retyping.
"""
import argparse
import glob
import json
import os

CTX_PARAMS = 14.9        # context transformer, millions
PER_LAYER = 7.09         # one MuRIL base encoder layer, millions


def classify(d):
    """(sort_key, label, trained_params_string) for one result file."""
    if "kind" in d:                                   # from baselines.py
        lab = ("TextTiling (lexical)" if d["kind"] == "lexical"
               else "Semantic tiling (frozen MuRIL)")
        return (1 if d["kind"] == "lexical" else 2), lab, "---"
    a = d.get("args", {})
    if a.get("mode") == "frozen":
        return 10, "Frozen encoder", f"{CTX_PARAMS:.1f}M"
    n = a.get("train_layers", -1)
    if n < 0:
        return 99, "Finetune, all layers", f"{CTX_PARAMS + 12*PER_LAYER:.1f}M"
    plural = "" if n == 1 else "s"
    return (10 + n, f"Finetune, top {n} layer{plural}",
            f"{CTX_PARAMS + n * PER_LAYER:.1f}M")


def fmt(x, best=False, nd=4):
    if x is None:
        return "---"
    s = f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if best else s


def collect(runs_dir, task):
    rows = []
    zero = None
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.result.json"))):
        d = json.load(open(path, encoding="utf-8"))
        a = d.get("args", {})
        # keep only files for this task
        if a.get("task") != task:
            continue
        test = d.get("test") or {}
        if "head_a" not in test:
            continue
        if zero is None and "no_boundary" in test:
            zero = test["no_boundary"]
        key, label, params = classify(d)
        rows.append({"key": key, "label": label, "params": params,
                     "a": test["head_a"], "b": test.get("head_b"),
                     "acc": test.get("three_way_accuracy"),
                     "file": os.path.basename(path)})
    rows.sort(key=lambda r: r["key"])
    return rows, zero


def best_of(rows, get, lower_is_better):
    vals = [get(r) for r in rows if get(r) is not None]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--task", choices=["fine", "hier"], required=True)
    args = ap.parse_args()

    rows, zero = collect(args.runs, args.task)
    if not rows:
        raise SystemExit(f"no {args.task} result files in {args.runs}")

    print(f"% generated from {args.runs} -- {len(rows)} runs")
    for r in rows:
        print(f"%   {r['label']:<34} {r['file']}")
    print()

    # best values, over real systems only (the all-zero row is a reference,
    # not a competitor -- if it beats everything on Pk, that is a finding to
    # discuss, not a number to bold)
    ba = {k: best_of(rows, lambda r: r["a"].get(k), k in ("pk", "window_diff"))
          for k in ("pk", "window_diff", "precision", "recall", "f1")}
    bb = {k: best_of(rows, lambda r: (r["b"] or {}).get(k),
                     k in ("pk", "window_diff"))
          for k in ("pk", "window_diff", "f1")}
    bacc = best_of(rows, lambda r: r["acc"], False)

    def ratio(m):
        return m["pred_boundaries"] / max(m["ref_boundaries"], 1)

    if args.task == "fine":
        if zero:
            print(f"No boundary                & ---   & {fmt(zero['pk'])} & "
                  f"{fmt(zero['window_diff'])} & ---  & 0.00 & 0.00 & 0.00 \\\\")
        for r in rows:
            m = r["a"]
            print(f"{r['label']:<26} & {r['params']:<5} & "
                  f"{fmt(m['pk'], m['pk']==ba['pk'])} & "
                  f"{fmt(m['window_diff'], m['window_diff']==ba['window_diff'])} & "
                  f"{fmt(m['precision'], m['precision']==ba['precision'])} & "
                  f"{fmt(m['recall'], m['recall']==ba['recall'])} & "
                  f"{fmt(m['f1'], m['f1']==ba['f1'])} & "
                  f"{ratio(m):.2f} \\\\")
    else:
        if zero:
            print(f"No boundary                & ---   & {fmt(zero['pk'])} & "
                  f"{fmt(zero['window_diff'])} & 0.00 & --- & --- & 0.00 & --- \\\\")
        for r in rows:
            m, s = r["a"], r["b"] or {}
            acc = f"{r['acc']:.4f}" if r["acc"] is not None else "---"
            if r["acc"] is not None and r["acc"] == bacc:
                acc = f"\\textbf{{{acc}}}"
            print(f"{r['label']:<26} & {r['params']:<5} & "
                  f"{fmt(m['pk'], m['pk']==ba['pk'])} & "
                  f"{fmt(m['window_diff'], m['window_diff']==ba['window_diff'])} & "
                  f"{fmt(m['f1'], m['f1']==ba['f1'])} & "
                  f"{fmt(s.get('pk'), s.get('pk')==bb['pk'])} & "
                  f"{fmt(s.get('window_diff'), s.get('window_diff')==bb['window_diff'])} & "
                  f"{fmt(s.get('f1'), s.get('f1')==bb['f1'])} & {acc} \\\\")

    # ---- sanity warnings, so a bad row is caught before it reaches the paper
    print()
    if zero:
        for r in rows:
            if r["a"]["pk"] >= zero["pk"]:
                print(f"% WARNING: {r['label']} has Pk {r['a']['pk']:.4f} >= "
                      f"all-zero {zero['pk']:.4f} -- worse than predicting "
                      f"nothing")
    for r in rows:
        rr = ratio(r["a"])
        if rr < 0.5 or rr > 1.8:
            print(f"% WARNING: {r['label']} predicts {rr:.2f}x the reference "
                  f"boundaries -- badly calibrated")


if __name__ == "__main__":
    main()
