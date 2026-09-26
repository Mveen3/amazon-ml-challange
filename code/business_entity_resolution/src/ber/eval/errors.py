"""Error analysis of the train out-of-fold predictions (Track 0 of improvement.md).

Re-applies the tuned decision to the train OOF scores, then attributes every missed true pair (false negative)
and every wrong match (false positive) to a cause, profiles them (missing address, script, name / address
similarity, house-number hit, cluster size, source), and splits the macro-F0.5 loss by cause. It also compares
the score distributions of train OOF and test per country (train/test shift, improvement.md C4/C6).
Writes ``models/error_analysis.md`` (+ ``.json``) and prints the report to the log. CPU only.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ..blocking.prerank import s1_stats
from ..decision.tune import apply, load_arrays
from ..models.gate import ownership
from ..utils import load_json, log, n_workers, save_json, work_dir
from .metric import f05_vec

KEYS = ["s1_uid", "rec_uid"]


def _md(df: pl.DataFrame, floats: int = 4) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        rows.append("| " + " | ".join(f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def _rec_info(cfg, split: str) -> pl.DataFrame:
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src"])
    norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=["uid", "addr_missing", "script", "is_domain"])
    return rec.filter(pl.col("src") != 1).join(norm, on="uid").rename({"uid": "rec_uid"}).drop("src")


def _profile(df: pl.DataFrame, n_all: int) -> pl.DataFrame:
    """One row per cause: share of all true pairs (or of all predictions) and the typical pair profile."""
    return df.group_by("cause").agg([
        pl.len().alias("pairs"),
        (pl.len() / n_all).alias("share"),
        pl.col("addr_missing").cast(pl.Float64).mean().alias("rec_addr_missing"),
        (pl.col("script") > 0).cast(pl.Float64).mean().alias("rec_indic_script"),
        pl.col("is_domain").cast(pl.Float64).mean().alias("rec_domain_name"),
        pl.col("nm_tset").fill_nan(None).mean().alias("name_sim"),
        pl.col("ad_tset").fill_nan(None).mean().alias("addr_sim"),
        pl.col("num_hit").cast(pl.Float64).fill_nan(None).mean().alias("num_hit"),
        (pl.col("s1_name_freq") > 1).cast(pl.Float64).mean().alias("s1_name_shared"),
        pl.col("q").fill_nan(None).mean().alias("q_mean"),
    ]).sort("pairs", descending=True)


def run_errors(cfg) -> dict:
    th = load_json(work_dir(cfg, None, "models", "thresholds.json"))
    arr = load_arrays(cfg, "train")
    pred = apply(cfg, arr, th).select(KEYS)
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet")).select(KEYS)
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "prof", "n_true"]).join(
        s1_stats(cfg, "train"), on="s1_uid", how="left")
    r2 = pl.read_parquet(work_dir(cfg, "train", "preds", "r2.parquet"),
                         columns=KEYS + ["src", "nm_tset", "ad_tset", "num_hit", "q"])
    r2 = ownership(r2.sort(KEYS)).select(KEYS + ["src", "nm_tset", "ad_tset", "num_hit", "q", "owner"])
    rec = _rec_info(cfg, "train")
    n_pred_s1 = pred.group_by("s1_uid").len().rename({"len": "n_pred"})

    # ---------------------------------------------------------------- missed true pairs (false negatives)
    fn = truth.join(pred, on=KEYS, how="anti")
    fn = (fn.join(r2, on=KEYS, how="left").join(n_pred_s1, on="s1_uid", how="left")
          .join(s1.select(["s1_uid", "n_true", "s1_name_freq"]), on="s1_uid", how="left")
          .join(rec, on="rec_uid", how="left"))
    fn = fn.with_columns(
        pl.when(pl.col("q").is_null()).then(pl.lit("not a candidate"))
        .when(~pl.col("owner")).then(pl.lit("record owned by another S1"))
        .when(pl.col("n_pred").is_null()).then(pl.lit("entity predicted empty"))
        .otherwise(pl.lit("candidate, below threshold")).alias("cause"))
    lost = fn.filter(pl.col("cause") == "not a candidate")
    if lost.height:  # similarities for pairs that never got features: names/addresses from the normalised table
        nm = pl.read_parquet(work_dir(cfg, "train", "norm.parquet"), columns=["uid", "name_core", "addr_latin"])
        lx = (lost.select(KEYS).join(nm.rename({"uid": "s1_uid", "name_core": "sn", "addr_latin": "sa"}), on="s1_uid")
              .join(nm.rename({"uid": "rec_uid", "name_core": "rn", "addr_latin": "ra"}), on="rec_uid"))
        w = n_workers(cfg)
        lx = lx.with_columns([
            pl.Series("nm_tset_x", cpdist(lx["sn"].to_list(), lx["rn"].to_list(), scorer=fuzz.token_set_ratio,
                                          workers=w, dtype=np.float32) / 100.0),
            pl.Series("ad_tset_x", cpdist(lx["sa"].to_list(), lx["ra"].to_list(), scorer=fuzz.token_set_ratio,
                                          workers=w, dtype=np.float32) / 100.0)]).select(KEYS + ["nm_tset_x", "ad_tset_x"])
        fn = fn.join(lx, on=KEYS, how="left").with_columns([
            pl.coalesce("nm_tset", "nm_tset_x").alias("nm_tset"), pl.coalesce("ad_tset", "ad_tset_x").alias("ad_tset")
        ]).drop(["nm_tset_x", "ad_tset_x"])
    tp = (truth.join(pred, on=KEYS, how="semi").join(r2, on=KEYS, how="left")
          .join(s1.select(["s1_uid", "n_true", "s1_name_freq"]), on="s1_uid", how="left")
          .join(rec, on="rec_uid", how="left").with_columns(pl.lit("(found: true positives)").alias("cause")))
    fn_tab = _profile(pl.concat([fn.drop("n_pred"), tp], how="diagonal_relaxed"), truth.height)

    # ---------------------------------------------------------------- wrong matches (false positives)
    fp = (pred.join(truth, on=KEYS, how="anti").join(r2, on=KEYS, how="left")
          .join(s1.select(["s1_uid", "n_true", "s1_name_freq"]), on="s1_uid", how="left")
          .join(rec, on="rec_uid", how="left"))
    owned_in_truth = truth.select(pl.col("rec_uid").unique()).with_columns(pl.lit(True).alias("_true_rec"))
    fp = fp.join(owned_in_truth, on="rec_uid", how="left").with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("S1 is a singleton"))
        .when(pl.col("_true_rec").is_not_null()).then(pl.lit("record belongs to another S1"))
        .otherwise(pl.lit("record is a distractor (no owner)")).alias("cause")).drop("_true_rec")
    fp_tab = _profile(fp, pred.height)

    # ---------------------------------------------------------------- macro-F0.5 loss by entity error pattern
    ent = (s1.select(["s1_uid", "prof", "n_true"])
           .join(pred.join(truth, on=KEYS, how="semi").group_by("s1_uid").len().rename({"len": "tp"}), on="s1_uid",
                 how="left")
           .join(n_pred_s1, on="s1_uid", how="left").with_columns([pl.col("tp").fill_null(0),
                                                                   pl.col("n_pred").fill_null(0)]))
    f = f05_vec(ent["tp"].to_numpy(), ent["n_pred"].to_numpy(), ent["n_true"].to_numpy())
    ent = ent.with_columns(pl.Series("loss", 1.0 - f)).with_columns(
        pl.when(pl.col("loss") <= 0).then(pl.lit("correct"))
        .when((pl.col("n_true") == 0)).then(pl.lit("singleton given matches"))
        .when(pl.col("n_pred") == 0).then(pl.lit("has matches, predicted empty"))
        .when(pl.col("tp") == 0).then(pl.lit("only wrong records"))
        .when((pl.col("n_pred") > pl.col("tp")) & (pl.col("tp") < pl.col("n_true"))).then(pl.lit("missed + wrong"))
        .when(pl.col("n_pred") > pl.col("tp")).then(pl.lit("all found + wrong extra"))
        .otherwise(pl.lit("some missed, none wrong")).alias("pattern"))
    n = ent.height
    loss_tab = (ent.group_by("pattern").agg([pl.len().alias("entities"), (pl.len() / n).alias("share"),
                                             (pl.col("loss").sum() / n).alias("macro_loss")])
                .sort("macro_loss", descending=True))
    size_tab = (ent.with_columns(pl.col("n_true").clip(0, 7).alias("n_true"))
                .group_by(["n_true", "pattern"]).agg((pl.col("loss").sum() / n).alias("macro_loss"))
                .filter(pl.col("macro_loss") > 0).sort(["n_true", "macro_loss"], descending=[False, True]))

    # ---------------------------------------------------------------- train OOF vs test score distributions
    rows = []
    for split in ("train", "test"):
        p = pl.read_parquet(work_dir(cfg, split, "preds", "r2.parquet"), columns=KEYS + ["prof", "q"])
        top = p.group_by(["s1_uid", "prof"]).agg(pl.col("q").max().alias("top"), pl.len().alias("ncand"))
        for prof, g in top.group_by("prof"):
            t = g["top"].to_numpy()
            rows.append({"split": split, "profile": prof[0], "S1": g.height,
                         "cands_per_S1": float(g["ncand"].mean()),
                         "q_in_0.3_0.9": float(p.filter(pl.col("prof") == prof[0])["q"].is_between(0.3, 0.9).mean()),
                         "top_q_p10": float(np.quantile(t, 0.10)), "top_q_p50": float(np.quantile(t, 0.5)),
                         "top1_uncertain_0.2_0.8": float(((t > 0.2) & (t < 0.8)).mean())})
    dist_tab = pl.DataFrame(rows).sort(["profile", "split"])

    rep = ["# Error analysis (train out-of-fold, tuned thresholds)", "",
           f"True pairs {truth.height:,}, predicted pairs {pred.height:,}, S1 entities {n:,}.", "",
           "## Missed true pairs by cause (share = of all true pairs)", "", _md(fn_tab), "",
           "## Wrong matches by cause (share = of all predicted pairs)", "", _md(fp_tab), "",
           "## Macro-F0.5 loss by entity error pattern", "", _md(loss_tab, 5), "",
           "## Loss by true cluster size and pattern", "", _md(size_tab, 5), "",
           "## Score distributions: train out-of-fold vs test, per country profile", "", _md(dist_tab), ""]
    text = "\n".join(rep)
    out = work_dir(cfg, None, "models", "error_analysis.md")
    out.write_text(text)
    save_json({"fn": fn_tab.to_dicts(), "fp": fp_tab.to_dicts(), "loss": loss_tab.to_dicts(),
               "dist": dist_tab.to_dicts()}, work_dir(cfg, None, "models", "error_analysis.json"))
    log().info("\n%s", text)
    return {"report": str(out)}
