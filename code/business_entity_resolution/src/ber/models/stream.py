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

from pathlib import Path
from typing import Callable

import polars as pl

from ..utils import log, work_dir
from .gbdt import GBDT, predict_by_fold, to_matrix

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
    out = []
    for i, p in enumerate(paths):
        df = pl.read_parquet(p, columns=read)
        if transform is not None:
            df = transform(df)
        if df.height == 0:
            continue
        pred = predict_by_fold(models, to_matrix(df, features), df["fold"].to_numpy())
        out.append(df.select([c for c in keep_cols if c in df.columns]).with_columns(pl.Series("pred", pred)))
        if (i + 1) % max(1, len(paths) // 10) == 0:
            log().info("   scored %d/%d shards", i + 1, len(paths))
    return pl.concat(out).sort(KEYS)
