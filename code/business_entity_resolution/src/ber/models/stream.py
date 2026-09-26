"""Out-of-core model helpers: train on a sample of entities, predict shard by shard.

Full-scale feature tables (tens of millions of pairs x ~150 features) do not fit
in RAM on a Kaggle box (~29 GB). Instead of materialising one dense matrix:
  * training loads only the rows of a random sample of S1 entities (whole
    entities, so candidate lists stay intact); fold models are fit on that sample;
  * OOF / test predictions stream over the feature shards one file at a time.
Every train pair still gets an OOF score (from the fold model that never saw its
entity), so downstream stages see the full candidate graph.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import polars as pl

from ..utils import gpu_cap, log, work_dir
from .gbdt import GBDT, predict_by_fold, to_matrix, xgb_gpu_available

KEYS = ["s1_uid", "rec_uid"]


def r1_paths(cfg, split: str) -> list[Path]:
    return sorted(work_dir(cfg, split, "feats", "r1").glob("*.parquet"))


def columns_of(paths: list[Path]) -> list[str]:
    return list(pl.read_parquet_schema(paths[0]).keys()) if paths else []


def entities(paths: list[Path]) -> pl.DataFrame:
    """One row per S1 entity present in the shards: (s1_uid, fold)."""
    return (pl.scan_parquet([str(p) for p in paths]).select(["s1_uid", "fold"])
            .unique("s1_uid").collect().sort("s1_uid"))


def sample_entities(ents: pl.DataFrame, n: int, seed: int) -> pl.Series:
    if ents.height <= n:
        return ents["s1_uid"]
    return ents["s1_uid"].sample(n, seed=seed).sort()


def training_sample(paths: list[Path], ents: pl.DataFrame, n: int, max_rows: int, seed: int, tag: str) -> pl.Series:
    """Entities to train on, capped by rows as well as entities: rows = entities x candidates per S1, and a fixed
    entity count can exceed RAM when the candidate lists are long (entities = min(n, max_rows / pairs-per-S1))."""
    n_rows = pl.scan_parquet([str(p) for p in paths]).select(pl.len()).collect().item() if paths else 0
    per_s1 = max(1.0, n_rows / max(1, ents.height))
    n_ent = min(int(n), int(max_rows / per_s1))
    if n_ent < min(int(n), ents.height):
        log().info("   %s: training on %d entities (not %d): %.1f pairs/S1 x %d would exceed max_train_rows=%d",
                   tag, n_ent, min(int(n), ents.height), per_s1, min(int(n), ents.height), max_rows)
    return sample_entities(ents, n_ent, seed)


def load_rows(paths: list[Path], columns: list[str], s1_keep: pl.Series | None = None) -> pl.DataFrame:
    """Rows of the kept entities only, read shard by shard (peak memory = output size)."""
    keep = s1_keep.implode() if s1_keep is not None else None
    avail = set(columns_of(paths))
    cols = [c for c in dict.fromkeys(columns) if c in avail]
    parts = []
    for p in paths:
        df = pl.read_parquet(p, columns=cols)
        if keep is not None:
            df = df.filter(pl.col("s1_uid").is_in(keep))
        if df.height:
            parts.append(df)
    out = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame(schema={c: pl.Float32 for c in cols})
    log().info("   loaded %d training rows (%d entities)", out.height,
               out["s1_uid"].n_unique() if out.height else 0)
    return out.sort(KEYS) if out.height else out


def predict_shards(paths: list[Path], models: list[GBDT], features: list[str], keep_cols: list[str],
                   transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None) -> pl.DataFrame:
    """Score every row shard by shard. Train rows use their own fold's model (OOF), fold -1 rows the mean.

    ``transform`` may add columns (e.g. round-2 extras) before scoring; features missing from the
    shard itself are expected to come from it.
    """
    avail = set(columns_of(paths))
    read = [c for c in dict.fromkeys(list(keep_cols) + list(features) + ["fold"]) if c in avail]
    sets = _device_model_sets(models)
    done = [0]

    def score(i: int, ms: list[GBDT]) -> pl.DataFrame | None:
        df = pl.read_parquet(paths[i], columns=read)
        if transform is not None:
            df = transform(df)
        if df.height == 0:
            return None
        pred = predict_by_fold(ms, to_matrix(df, features), df["fold"].to_numpy())
        done[0] += 1
        if done[0] % max(1, len(paths) // 10) == 0:
            log().info("   scored %d/%d shards", done[0], len(paths))
        return df.select([c for c in keep_cols if c in df.columns]).with_columns(pl.Series("pred", pred))

    if len(sets) == 1:
        out = [score(i, models) for i in range(len(paths))]
    else:  # shards split across GPUs: one thread and one model copy per GPU (xgboost releases the GIL)
        from concurrent.futures import ThreadPoolExecutor

        log().info("   scoring %d shards on %d devices at once", len(paths), len(sets))
        with ThreadPoolExecutor(len(sets)) as ex:
            futs = [ex.submit(lambda g: [(i, score(i, sets[g])) for i in range(g, len(paths), len(sets))], g)
                    for g in range(len(sets))]
            res = dict(pair for f in futs for pair in f.result())
        out = [res[i] for i in range(len(paths))]
    return pl.concat([o for o in out if o is not None]).sort(KEYS)


def _device_model_sets(models: list[GBDT]) -> list[list[GBDT]]:
    """One copy of the fold models per GPU for parallel scoring (XGBoost with >= 2 GPUs); otherwise [models].

    ``BER_SCORE_DEVICES`` (e.g. ``cpu,cpu``) forces a split on other devices (used to test the threaded path).
    """
    forced = os.environ.get("BER_SCORE_DEVICES")
    if forced:
        devices = [d.strip() for d in forced.split(",") if d.strip()]
    else:
        if not models or models[0].backend != "xgboost" or not xgb_gpu_available():
            return [models]
        import torch

        n = torch.cuda.device_count()
        if gpu_cap() > 0:
            n = min(n, gpu_cap())
        devices = [f"cuda:{g}" for g in range(n)]
    if len(devices) < 2 or models[0].backend != "xgboost":
        return [models]
    return [[m.on_device(d) for m in models] for d in devices]
