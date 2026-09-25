"""Round-2 features: group consensus and one-owner competition on OOF round-1 scores.

For a pair (e, r) with confident members M(e) = {r' != r : p1(e, r') >= tau}:
  cons_n, name/address similarity of r to M(e) (max, mean),
  agreement of r with the members' majority house number / PIN,
  S1-number hit rate among members.
Competition (``c1_*``): r's best score with any other S1, margins and shares.
All inputs are OOF on train, so confidence is realistic (no self-reinforcement).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ..utils import chunks, log, n_workers, work_dir
from .context import context_names, score_context

CONS_FEATURES = ["cons_n", "cons_name_max", "cons_name_mean", "cons_addr_max", "cons_addr_mean",
                 "cons_numhit_rate", "cons_num_agree", "cons_pin_agree", "cons_is_member"]
R2_EXTRA = ["p1", "ce", "ce_scored"] + context_names("c1") + CONS_FEATURES


def _consensus(df: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    rc = cfg.r2
    tau = float(rc.conf_tau)
    norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"),
                           columns=["uid", "name_core", "addr_latin", "num_primary", "nums", "pin"])
    members = (df.filter((pl.col("p1") >= tau) & (pl.col("c1_s1_rank") <= int(rc.top_members)))
               .select(["s1_uid", pl.col("rec_uid").alias("m"), pl.col("num_hit").alias("m_hit")]))
    # member-level majority values per S1
    mv = members.join(norm.select([pl.col("uid").alias("m"), pl.col("num_primary").alias("m_num"),
                                   pl.col("pin").alias("m_pin")]), on="m", how="left")
    maj = mv.group_by("s1_uid").agg([
        pl.len().alias("_n"), pl.col("m_hit").sum().alias("_hits"),
        pl.col("m_num").filter(pl.col("m_num") != "").mode().sort().first().alias("maj_num"),
        pl.col("m_pin").filter(pl.col("m_pin") != "").mode().sort().first().alias("maj_pin"),
    ])
    base = df.select(["s1_uid", "rec_uid", "p1", "num_hit"]).join(maj, on="s1_uid", how="left")
    base = base.join(norm.select([pl.col("uid").alias("rec_uid"), "nums", "pin"]), on="rec_uid", how="left")
    is_mem = (pl.col("p1") >= tau).cast(pl.Int8)
    base = base.with_columns([
        is_mem.alias("cons_is_member"),
        (pl.col("_n").fill_null(0) - is_mem).alias("cons_n"),
        pl.when(pl.col("_n").fill_null(0) - is_mem > 0)
        .then((pl.col("_hits").fill_null(0) - is_mem * pl.col("num_hit").fill_null(0)) / (pl.col("_n") - is_mem))
        .otherwise(None).alias("cons_numhit_rate"),
        pl.when(pl.col("maj_num").is_not_null() & (pl.col("nums").list.len() > 0))
        .then(pl.col("nums").list.contains(pl.col("maj_num")).cast(pl.Float32)).otherwise(None).alias("cons_num_agree"),
        pl.when(pl.col("maj_pin").is_not_null() & (pl.col("pin") != ""))
        .then((pl.col("pin") == pl.col("maj_pin")).cast(pl.Float32)).otherwise(None).alias("cons_pin_agree"),
    ])
    # similarity of r to the other confident members
    txt = norm.select(["uid", "name_core", "addr_latin"])
    sims = []
    s1_ids = df["s1_uid"].unique().sort()
    workers = n_workers(cfg)
    for s, e in chunks(len(s1_ids), int(rc.chunk_s1)):
        ids = s1_ids.slice(s, e - s)
        pr = df.filter(pl.col("s1_uid").is_in(ids)).select(["s1_uid", "rec_uid"])
        x = (pr.join(members.select(["s1_uid", "m"]), on="s1_uid").filter(pl.col("m") != pl.col("rec_uid"))
             .sort(["s1_uid", "rec_uid", "m"]))
        if x.height == 0:
            continue
        x = (x.join(txt.rename({"uid": "rec_uid", "name_core": "rn", "addr_latin": "ra"}), on="rec_uid", how="left",
                    maintain_order="left")
             .join(txt.rename({"uid": "m", "name_core": "mn", "addr_latin": "ma"}), on="m", how="left",
                   maintain_order="left"))
        ns = cpdist(x["rn"].to_list(), x["mn"].to_list(), scorer=fuzz.token_set_ratio, workers=workers,
                    dtype=np.float32) / 100.0
        ad = cpdist(x["ra"].to_list(), x["ma"].to_list(), scorer=fuzz.token_set_ratio, workers=workers,
                    dtype=np.float32) / 100.0
        sims.append(x.select(["s1_uid", "rec_uid"]).with_columns([pl.Series("ns", ns), pl.Series("as", ad)])
                    .group_by(["s1_uid", "rec_uid"], maintain_order=True).agg([
                        pl.col("ns").max().alias("cons_name_max"), pl.col("ns").mean().alias("cons_name_mean"),
                        pl.col("as").max().alias("cons_addr_max"), pl.col("as").mean().alias("cons_addr_mean")]))
    base = base.select(["s1_uid", "rec_uid", "cons_is_member", "cons_n", "cons_numhit_rate", "cons_num_agree",
                        "cons_pin_agree"])
    if sims:
        base = base.join(pl.concat(sims), on=["s1_uid", "rec_uid"], how="left")
    else:
        base = base.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in
                                  ("cons_name_max", "cons_name_mean", "cons_addr_max", "cons_addr_mean")])
    return base


def build_r2_extras(cfg, split: str, p1: pl.DataFrame, ce: pl.DataFrame | None) -> pl.DataFrame:
    """p1: (s1_uid, rec_uid, p1, num_hit). Returns (s1_uid, rec_uid, *R2_EXTRA)."""
    df = p1
    if ce is not None:
        df = df.join(ce, on=["s1_uid", "rec_uid"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias("ce"))
    df = df.with_columns(pl.col("ce").is_not_null().cast(pl.Int8).alias("ce_scored"))
    df = score_context(df.sort(["s1_uid", "rec_uid"]), "p1", "c1")
    cons = _consensus(df, cfg, split)
    out = df.join(cons, on=["s1_uid", "rec_uid"], how="left")
    log().info("  %s r2 extras: %d rows", split, out.height)
    return out.select(["s1_uid", "rec_uid"] + R2_EXTRA).sort(["s1_uid", "rec_uid"])
