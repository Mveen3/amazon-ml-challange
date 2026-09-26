"""Run the two cross-fitted cross-encoder halves concurrently, one process per GPU.

The halves are independent (half h trains on the S1 entities of ``folds.neural_halves[h]`` and scores the others), so
with two GPUs each half gets a whole GPU. This avoids ``DataParallel``'s per-step model copies and gather, and lets
the two halves overlap their CPU-side tokenisation too. Any failure (a worker exits non-zero, a GPU runs out of
memory, ...) is reported to the caller, which then falls back to the sequential single-process path.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..utils import gpu_cap, log


def n_gpus() -> int:
    try:
        import torch

        n = torch.cuda.device_count()
        return min(n, gpu_cap()) if gpu_cap() > 0 else n
    except Exception:  # noqa: BLE001
        return 0


def usable(cfg) -> bool:
    cc = cfg.ce
    return bool(cc.get("parallel_halves", True)) and n_gpus() >= int(cc.get("parallel_min_gpus", 2)) and "_cli" in cfg


def _spawn(cfg, gpu: int, args: list[str]) -> subprocess.Popen:
    src = str(Path(__file__).resolve().parents[2])  # .../src
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join([src] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    cmd = [sys.executable, "-u", "-m", "ber.models.ce_worker", "--config", cfg["_cli"]["config"]]
    for o in cfg["_cli"]["overrides"]:
        cmd += ["--set", o]
    return subprocess.Popen(cmd + args, env=env)


def _run(cfg, jobs: list[tuple[int, list[str]]], what: str) -> bool:
    """Start one worker per (half, args) job (half h on GPU h % n_gpus) and wait for all of them.

    While they run, the parts they have finished are pushed to the checkpoint every ``ce.sync_minutes``, so a
    session that ends mid-way resumes from them (the workers themselves never talk to Hugging Face).
    """
    from ..checkpoint import sync_now

    g = max(1, n_gpus())
    procs = [(_spawn(cfg, h % g, args), h) for h, args in jobs]
    log().info("  CE %s: %d workers running concurrently (GPU %s)", what, len(procs), ", ".join(str(h % g) for _, h in procs))
    every, last = 60.0 * float(cfg.ce.get("sync_minutes", 15)), time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(5)
        if time.time() - last >= every:
            sync_now(f"cross-encoder {what}: parts finished so far")
            last = time.time()
    codes = [(p.wait(), h) for p, h in procs]
    bad = [f"half {h} exit {c}" for c, h in codes if c != 0]
    if bad:
        log().warning("  CE %s: worker failure (%s) -> falling back to the sequential path", what, "; ".join(bad))
    return not bad


def train_halves(cfg, halves: list[int]) -> bool:
    return _run(cfg, [(h, ["--op", "train", "--half", str(h)]) for h in halves], "training")


def infer_halves(cfg, split: str) -> bool:
    from .cross_encoder import merge_halves

    if not _run(cfg, [(h, ["--op", "infer", "--half", str(h), "--split", split]) for h in (0, 1)], f"scoring ({split})"):
        return False  # finished halves / parts are kept (checked against their plans): the fallback resumes from them
    merge_halves(cfg, split)
    return True
