#!/usr/bin/env python3
"""
Runs the full ablation queue on a single GPU, sequentially, and keeps going
even if one run fails. Pure Python (subprocess) -- no bash, no tmux needed.

BEFORE YOU WALK AWAY, CHECK THESE TWO LINES -- I could not see your repo,
so these are my best guess from the result JSON you shared, not verified:
  1. SCRIPT   -- path to your training entrypoint (guessed: train.py)
  2. FT_MODE  -- the --mode value your script expects for fine-tuning
                 (guessed: "finetune"; your frozen runs used "frozen")

Launch it so it survives you logging out -- see the chat message for the
exact command (nohup is the one that doesn't need bash or tmux).
"""
import subprocess
import sys
import datetime
from pathlib import Path

SCRIPT = "train.py"          # <-- CHECK: your actual training script path
FT_MODE = "finetune"         # <-- CHECK: value your script expects for finetuning

ENCODER = "muril-base-cased"
EPOCHS = 10                  # matches the frozen runs you already have (25 epochs)
CTX_LAYERS = 2
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
WARMUP = 0.1
MAX_TOKENS = 64
W_B = 1.0
SEED = 13
SELECT_ON = "f1"
PATIENCE = 0
AMP = "bf16"
BATCH_SIZE = 256               # settled value from your GPU-utilization tuning
MAX_BATCH_SENTS = 800         # ~90% util at 800, OOM at 1024 -- 768 leaves margin
FROZEN_LR = "1e-3"
FT_LR = "2e-5"                 # encoder LR when fine-tuning (per paper.tex)
FT_HEAD_LR = "1e-3"            # context-transformer/head LR when fine-tuning

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
FAIL_LOG = LOG_DIR / "failures.log"
FAIL_LOG.write_text("")

TASKS = {
    "hier": ["dataset_hier", "--task", "hier", "--vectors", "vecs_hier", "--out", "hier_runs"],
    "fine": ["dataset",      "--task", "fine", "--vectors", "vecs_fine", "--out", "fine_runs"],
}

COMMON = [
    "--encoder", ENCODER,
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--max-batch-sents", str(MAX_BATCH_SENTS),
    "--weight-decay", str(WEIGHT_DECAY),
    "--warmup", str(WARMUP),
    "--ctx-layers", str(CTX_LAYERS),
    "--dropout", str(DROPOUT),
    "--max-tokens", str(MAX_TOKENS),
    "--w-b", str(W_B),
    "--seed", str(SEED),
    "--select-on", SELECT_ON,
    "--patience", str(PATIENCE),
    "--amp", AMP,
]


def build_jobs():
    jobs = []

    for task in ("hier", "fine"):
        jobs.append((
            f"{task}_frozen",
            TASKS[task] + ["--mode", "frozen", "--lr", FROZEN_LR] + COMMON +
            ["--name", f"{task}_frozen_bs{BATCH_SIZE}_mbs{MAX_BATCH_SENTS}"],
        ))

    for task in ("hier", "fine"):
        for tl in (1, 2, 3):
            jobs.append((
                f"{task}_finetune_tl{tl}",
                TASKS[task] + [
                    "--mode", FT_MODE, "--train-layers", str(tl),
                    "--lr", FT_LR, "--head-lr", FT_HEAD_LR,
                ] + COMMON + ["--name", f"{task}_ft_tl{tl}_bs{BATCH_SIZE}_mbs{MAX_BATCH_SENTS}"],
            ))

    return jobs


def run_job(name, args):
    ts = lambda: datetime.datetime.now().strftime("%F %T")
    print(f"=== [{ts()}] START  {name} ===", flush=True)
    log_path = LOG_DIR / f"{name}.log"
    cmd = [sys.executable, SCRIPT] + args
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        print(f"=== [{ts()}] DONE   {name} ===", flush=True)
    else:
        print(f"=== [{ts()}] FAILED {name} (see {log_path}) ===", flush=True)
        with open(FAIL_LOG, "a") as f:
            f.write(name + "\n")


def main():
    for name, args in build_jobs():
        run_job(name, args)

    print("=== queue finished. failures (if any) listed in", FAIL_LOG, "===")
    failures = FAIL_LOG.read_text().strip()
    if failures:
        print("FAILED RUNS:\n" + failures)


if __name__ == "__main__":
    main()
