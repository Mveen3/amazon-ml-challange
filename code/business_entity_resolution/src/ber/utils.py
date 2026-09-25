"""Shared helpers: logging, timing, seeding, process pools, work-dir layout."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_LOG = None
# deterministic cuBLAS GEMMs (GPU kNN / neural models) for reproducible audits
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def log() -> logging.Logger:
    global _LOG
    if _LOG is None:
        _LOG = logging.getLogger("ber")
        if not _LOG.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
            _LOG.addHandler(h)
        _LOG.setLevel(logging.INFO)
        _LOG.propagate = False
    return _LOG


@contextmanager
def timer(msg: str):
    t0 = time.time()
    log().info("▶ %s", msg)
    yield
    log().info("✔ %s (%.1fs)", msg, time.time() - t0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def inference_only(cfg) -> bool:
    """Audit mode: test split only, every model/threshold loaded from ``work_dir/models``."""
    return bool(cfg.run.get("inference_only", False))


def n_workers(cfg) -> int:
    n = int(cfg.run.n_workers)
    return (os.cpu_count() or 1) if n <= 0 else n


# --------------------------------------------------------------------------- paths
def work_dir(cfg, split: str | None = None, *parts: str) -> Path:
    p = Path(cfg.paths.work_dir)
    if split:
        p = p / split
    for part in parts:
        p = p / part
    return p


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_done(cfg, stage: str, split: str | None = None) -> bool:
    return (Path(cfg.paths.work_dir) / "_markers" / f"{stage}.{split or 'all'}.done").exists()


def mark_done(cfg, stage: str, split: str | None = None, info: dict | None = None) -> None:
    d = ensure_dir(Path(cfg.paths.work_dir) / "_markers")
    (d / f"{stage}.{split or 'all'}.done").write_text(json.dumps(info or {}, indent=1, default=str))


def save_json(obj, path: Path) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- parallelism
_THREAD_ENV = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")


def pmap(fn, tasks, workers: int, initializer=None, initargs=(), desc: str = "tasks"):
    """Ordered parallel map over ``tasks`` using spawned processes.

    Child processes are pinned to one thread each for polars/BLAS/numba so the
    pool uses exactly ``workers`` cores. ``workers <= 1`` runs inline (debuggable).
    """
    tasks = list(tasks)
    if not tasks:
        return []
    if workers <= 1 or len(tasks) == 1:
        if initializer:
            initializer(*initargs)
        return [fn(t) for t in tasks]
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    try:
        ctx = mp.get_context("spawn")
        out = []
        step = max(1, len(tasks) // 10)
        with ctx.Pool(processes=min(workers, len(tasks)), initializer=initializer, initargs=initargs) as pool:
            for i, res in enumerate(pool.imap(fn, tasks, chunksize=1)):
                out.append(res)
                if (i + 1) % step == 0 or i + 1 == len(tasks):
                    log().info("   %s: %d/%d", desc, i + 1, len(tasks))
        return out
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def chunks(n: int, size: int):
    """Yield (start, stop) index ranges covering ``range(n)``."""
    for s in range(0, n, size):
        yield s, min(n, s + size)


def gpu_cap() -> int:
    """Global cap on how many GPUs any stage may use (``run.max_gpus`` -> env BER_MAX_GPUS); 0 = use them all."""
    try:
        return max(0, int(os.environ.get("BER_MAX_GPUS", "0")))
    except ValueError:
        return 0


def torch_device(pref: str = "auto"):
    import torch

    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
