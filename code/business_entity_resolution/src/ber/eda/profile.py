"""Recompute the data facts of docs/Pipeline_Architecture.md §2 from the ingested data."""
from __future__ import annotations

import polars as pl

from ..utils import log, save_json, work_dir


def _noise(rec: pl.DataFrame) -> pl.DataFrame:
    return (rec.group_by(["src", "country"]).agg([
        pl.len().alias("n"),
        pl.col("name_raw").str.contains("[ऀ-෿]").mean().alias("indic_name"),
        pl.col("name_raw").str.to_lowercase().str.contains(r"(\.com|www\.|\.in\b|\.fr\b|\.net|\.org)").mean()
        .alias("domain_name"),
        (pl.col("addr_raw").str.strip_chars() == "").mean().alias("empty_addr"),
        pl.col("addr_raw").str.to_lowercase().str.contains(r"\b(null|n/a|none)\b").mean().alias("null_token_addr"),
    ]).sort(["src", "country"]))


def run_eda(cfg) -> dict:
    out = {}
    for split in ("train", "test"):
        rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"))
        dens = (rec.group_by("country").agg([(pl.col("src") == 1).sum().alias("s1"), (pl.col("src") != 1).sum().alias("rec")])
                .with_columns((pl.col("rec") / pl.col("s1")).alias("rec_per_s1")))
        out[split] = {"density": dens.to_dicts(), "noise": _noise(rec).to_dicts()}
        if split == "train":
            truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
            s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
            src = rec.select(["uid", "src", "country"])
            t = truth.join(src.rename({"uid": "rec_uid"}), on="rec_uid")
            per = t.group_by("s1_uid").agg([(pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")])
            out[split]["owners_per_record_max"] = int(truth.group_by("rec_uid").len()["len"].max())
            out[split]["cluster_size"] = s1.group_by("n_true").len().sort("n_true").to_dicts()
            out[split]["max_s2_per_s1"] = int(per["n2"].max())
            out[split]["max_s3_per_s1"] = int(per["n3"].max())
            out[split]["singleton_rate"] = s1.group_by("country").agg((pl.col("n_true") == 0).mean()).to_dicts()
            xc = t.join(s1.select(["s1_uid", pl.col("country").alias("c1")]), on="s1_uid")
            out[split]["cross_country_pairs"] = int((xc["country"] != xc["c1"]).sum())
            out[split]["unmatched_records"] = int(rec.filter(pl.col("src") != 1).height - truth["rec_uid"].n_unique())
    save_json(out, work_dir(cfg, None, "eda.json"))
    log().info("  EDA written to %s", work_dir(cfg, None, "eda.json"))
    return out
