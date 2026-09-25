"""TSV ingest into Parquet and submission writers.

Every split (train/test) gets one unified ``records.parquet``: S1 rows first,
then S2, then S3, with a dense int64 ``uid`` used by every later stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .normalize.profiles import profile_name
from .utils import ensure_dir, log, work_dir

SOURCES = (1, 2, 3)


def _read_tsv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     has_header=True, truncate_ragged_lines=True)
    return df.with_columns([pl.col(c).fill_null("") for c in df.columns])


def ingest(cfg, split: str) -> None:
    data = Path(cfg.paths.data_dir) / split
    frac = float(cfg.get("ingest", {}).get("train_s1_frac", 1.0)) if split == "train" else 1.0
    frames = []
    for src in SOURCES:
        df = _read_tsv(data / f"{split}_source{src}.tsv")
        df = df.select([
            pl.col("entity_id").str.strip_chars().alias("eid"),
            pl.col("business_name").alias("name_raw"),
            pl.col("business_address").alias("addr_raw"),
            pl.col("country").str.strip_chars().alias("country"),
        ]).with_columns(pl.lit(src, dtype=pl.Int8).alias("src"))
        frames.append(df)
        log().info("  %s S%d: %d rows", split, src, df.height)
    if frac < 1.0:
        frames = _subsample_train(frames, data / "train_ground_truth.tsv", frac, int(cfg.run.seed))
    rec = pl.concat(frames).with_row_index("uid").with_columns(pl.col("uid").cast(pl.Int64))
    prof_map = {c: profile_name(c) for c in rec["country"].unique().to_list()}
    rec = rec.with_columns(pl.col("country").replace_strict(prof_map, return_dtype=pl.Utf8).alias("prof"))
    out = ensure_dir(work_dir(cfg, split))
    rec.write_parquet(out / "records.parquet")

    s1 = rec.filter(pl.col("src") == 1).select(["uid", "eid", "country", "prof"]).rename({"uid": "s1_uid"})
    if split == "train":
        truth = _read_truth(data / "train_ground_truth.tsv", rec)
        truth.write_parquet(out / "truth.parquet")
        n_true = truth.group_by("s1_uid").agg(pl.len().alias("n_true"))
        s1 = s1.join(n_true, on="s1_uid", how="left").with_columns(pl.col("n_true").fill_null(0).cast(pl.Int32))
        rng = np.random.default_rng(int(cfg.run.seed))
        s1 = s1.with_columns(pl.Series("fold", rng.integers(0, int(cfg.folds.n_folds), s1.height), dtype=pl.Int8))
    else:
        s1 = s1.with_columns([pl.lit(None, dtype=pl.Int32).alias("n_true"), pl.lit(-1, dtype=pl.Int8).alias("fold")])
    s1.write_parquet(out / "s1.parquet")


def _subsample_train(frames: list[pl.DataFrame], gt_path: Path, frac: float, seed: int) -> list[pl.DataFrame]:
    """Keep a fraction of S1 clusters (with all their records) and the same fraction of unmatched records.

    Preserves cluster sizes and the unmatched-record rate; only the number of look-alike S1 siblings
    shrinks. A time-saving valve for small machines -- leave at 1.0 for the real run if you can.
    """
    gt = _read_tsv(gt_path)
    s1_ids = frames[0]["eid"]
    keep_s1 = s1_ids.sample(fraction=frac, seed=seed)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select([pl.col("source1_entity_id").str.strip_chars().alias("s1"),
                      pl.col("matched_entity_ids").str.strip_chars().alias("eid")])
             .filter(pl.col("eid").str.len_chars() > 0))
    keep_rec = pairs.filter(pl.col("s1").is_in(keep_s1.implode()))["eid"]
    matched = pairs["eid"]
    out = [frames[0].filter(pl.col("eid").is_in(keep_s1.implode()))]
    for i, df in enumerate(frames[1:]):
        orphan = ~pl.col("eid").is_in(matched.implode())
        orphans = df.filter(orphan).sample(fraction=frac, seed=seed + i + 1)
        out.append(pl.concat([df.filter(pl.col("eid").is_in(keep_rec.implode())), orphans]))
    log().info("  train subsampled to %.0f%% of S1 clusters: %d S1, %d records", 100 * frac, out[0].height,
               sum(f.height for f in out[1:]))
    return out


def _read_truth(path: Path, rec: pl.DataFrame) -> pl.DataFrame:
    gt = _read_tsv(path)
    ids = rec.select(["eid", "uid"])
    pairs = (gt.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .with_columns(pl.col("matched_entity_ids").str.strip_chars())
             .filter(pl.col("matched_entity_ids").str.len_chars() > 0))
    pairs = (pairs.join(ids.rename({"eid": "source1_entity_id", "uid": "s1_uid"}), on="source1_entity_id")
             .join(ids.rename({"eid": "matched_entity_ids", "uid": "rec_uid"}), on="matched_entity_ids")
             .select(["s1_uid", "rec_uid"]).unique())
    log().info("  truth pairs: %d", pairs.height)
    return pairs


def load(cfg, split: str, name: str, columns=None) -> pl.DataFrame:
    return pl.read_parquet(work_dir(cfg, split, f"{name}.parquet"), columns=columns)


def write_id_lists(s1: pl.DataFrame, pairs: pl.DataFrame, rec: pl.DataFrame, path: Path, header: list[str],
                   order_col: str | None = None) -> None:
    """Write one row per S1 (file order) with a comma-joined S2/S3 id list."""
    ids = rec.select([pl.col("uid").alias("rec_uid"), pl.col("eid").alias("rec_eid")])
    p = pairs.join(ids, on="rec_uid", how="inner")
    if order_col and order_col in p.columns:
        p = p.sort(["s1_uid", order_col], descending=[False, True])
    lists = p.group_by("s1_uid", maintain_order=True).agg(pl.col("rec_eid").unique(maintain_order=True).str.join(","))
    out = (s1.select(["s1_uid", "eid"]).join(lists, on="s1_uid", how="left", maintain_order="left")
           .select([pl.col("eid").alias(header[0]), pl.col("rec_eid").fill_null("").alias(header[1])]))
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
