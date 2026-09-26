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
from .admin import load_admin_vocab, strip_latin
from .context import context_names, density_normalize, score_context

CONS_FEATURES = ["cons_n", "cons_name_max", "cons_name_mean", "cons_addr_max", "cons_addr_mean",
                 "cons_numhit_rate", "cons_num_agree", "cons_pin_agree", "cons_is_member"]
# Competing-cluster state (improvement.md, the dominant loss): about 70% of missed true pairs are records with no
# address whose name is shared by several S1 entities (same business name at different addresses). Pairwise they
# are indistinguishable; what differs is the state of each S1's cluster. An S1 with no confident member almost
# surely owns a record somewhere (only 5.6% are singletons); one that already holds 5 records rarely takes another.
# These features let round 2 compare this S1 with the record's best competing S1.
COMP_FEATURES = ["cm_own_src_n", "cm_own_cap_left", "cm_own_empty", "cm_comp_has", "cm_comp_n", "cm_comp_src_n",
                 "cm_comp_cap_left", "cm_comp_empty", "cm_n_close", "cm_n_diff"]
R2_EXTRA = ["p1", "ce", "ce_scored"] + context_names("c1") + CONS_FEATURES + COMP_FEATURES
DENSITY_COLS_R2 = ["c1_s1_n", "c1_rec_n"]  # list sizes: divided by the country median (features.density_norm)


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
    # similarity of r to the other confident members (admin level removed for profiles without labels, as in r1)
    admin = load_admin_vocab(cfg, split)
    if admin:
        comps = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=["uid", "addr_comps"])
        country = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "country"])
        norm = strip_latin(norm.join(comps, on="uid", how="left"), admin, country).drop("addr_comps")
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


def _competition(df: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    """Competing-cluster state per pair (s, r): see COMP_FEATURES.

    Members of an S1 are its confident candidates (p1 >= conf_tau and within its top ``top_members``), as in the
    consensus features. For record r, the competitor of (s, r) is r's best-scoring *other* S1.
    """
    rc = cfg.r2
    tau, close = float(rc.conf_tau), float(rc.get("close_gap", 0.1))
    caps = {int(k): int(v) for k, v in cfg.decision.caps.items()}
    src = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src"]).rename({"uid": "rec_uid"})
    d = df.select(["s1_uid", "rec_uid", "p1", "c1_s1_rank"]).join(src, on="rec_uid", how="left")
    d = d.with_columns(((pl.col("p1") >= tau) & (pl.col("c1_s1_rank") <= int(rc.top_members))).alias("_mem"))
    cnt = d.filter("_mem").group_by("s1_uid").agg([pl.len().alias("_n"), (pl.col("src") == 2).sum().alias("_n2"),
                                                   (pl.col("src") == 3).sum().alias("_n3")])
    cap_of = pl.when(pl.col("src") == 2).then(caps.get(2, 99)).otherwise(caps.get(3, 99))
    # the record's two best S1s -> competitor of each pair
    top = (d.sort(["rec_uid", "p1", "s1_uid"], descending=[False, True, False]).group_by("rec_uid", maintain_order=True)
           .agg([pl.col("s1_uid").first().alias("_t1"), pl.col("s1_uid").get(1, null_on_oob=True).alias("_t2"),
                 pl.col("p1").max().alias("_pmax"), pl.len().alias("_nr")]))
    d = d.join(top, on="rec_uid", how="left").with_columns(
        pl.when(pl.col("s1_uid") == pl.col("_t1")).then(pl.col("_t2")).otherwise(pl.col("_t1")).alias("_comp"))
    close_n = d.group_by("rec_uid").agg((pl.col("p1") >= pl.col("_pmax") - close).sum().alias("cm_n_close"))
    comp_mem = d.filter("_mem").select([pl.col("s1_uid").alias("_comp"), "rec_uid", pl.lit(True).alias("_cmem")])
    d = (d.join(cnt, on="s1_uid", how="left")
         .join(cnt.rename({"s1_uid": "_comp", "_n": "_cn", "_n2": "_cn2", "_n3": "_cn3"}), on="_comp", how="left")
         .join(comp_mem, on=["_comp", "rec_uid"], how="left").join(close_n, on="rec_uid", how="left")
         .with_columns([pl.col(c).fill_null(0) for c in ("_n", "_n2", "_n3", "_cn", "_cn2", "_cn3")]))
    own_src = pl.when(pl.col("src") == 2).then(pl.col("_n2")).otherwise(pl.col("_n3")) - pl.col("_mem").cast(pl.Int32)
    comp_src = (pl.when(pl.col("src") == 2).then(pl.col("_cn2")).otherwise(pl.col("_cn3"))
                - pl.col("_cmem").fill_null(False).cast(pl.Int32))
    comp_n = pl.col("_cn") - pl.col("_cmem").fill_null(False).cast(pl.Int32)
    has = pl.col("_comp").is_not_null()
    own_n = pl.col("_n") - pl.col("_mem").cast(pl.Int32)
    return d.select([
        "s1_uid", "rec_uid",
        own_src.cast(pl.Float32).alias("cm_own_src_n"),
        (cap_of - own_src).cast(pl.Float32).alias("cm_own_cap_left"),
        (own_n == 0).cast(pl.Int8).alias("cm_own_empty"),
        has.cast(pl.Int8).alias("cm_comp_has"),
        pl.when(has).then(comp_n).otherwise(None).cast(pl.Float32).alias("cm_comp_n"),
        pl.when(has).then(comp_src).otherwise(None).cast(pl.Float32).alias("cm_comp_src_n"),
        pl.when(has).then(cap_of - comp_src).otherwise(None).cast(pl.Float32).alias("cm_comp_cap_left"),
        pl.when(has).then((comp_n == 0).cast(pl.Int8)).otherwise(None).alias("cm_comp_empty"),
        pl.col("cm_n_close").cast(pl.Float32),
        pl.when(has).then(own_n - comp_n).otherwise(None).cast(pl.Float32).alias("cm_n_diff"),
    ])


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
    comp = _competition(df, cfg, split)
    out = df.join(cons, on=["s1_uid", "rec_uid"], how="left").join(comp, on=["s1_uid", "rec_uid"], how="left")
    if bool(cfg.r2.get("density_norm", True)):
        prof = pl.read_parquet(work_dir(cfg, split, "s1.parquet"), columns=["s1_uid", "prof"])
        out = density_normalize(out.join(prof, on="s1_uid", how="left"), DENSITY_COLS_R2, by="prof").drop("prof")
    log().info("  %s r2 extras: %d rows", split, out.height)
    return out.select(["s1_uid", "rec_uid"] + R2_EXTRA).sort(["s1_uid", "rec_uid"])
