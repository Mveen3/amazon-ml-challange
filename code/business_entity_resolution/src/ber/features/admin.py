"""Administrative address components for profiles without training labels (e.g. France), learned from the data.

Why: for the trained profiles (US, India) the mined component-equivalence tables reconcile the state level of an
address ("tx" = "texas", Hindi state names). A profile that has no labelled pairs gets no such tables. In the
France test data every S1 address carries its *region* ("Nouvelle-Aquitaine"), while its S2/S3 records carry the
region only 1/3 of the time, a *department* ("Gironde") 1/3 of the time and neither 1/3 of the time, so the
address similarities of true pairs are pushed down by an unreconciled admin mismatch the models never saw.

What: for such profiles the admin level is detected from the split's own addresses, with no gazetteer:
  * S1 admin level:   components that end >= ``s1_last_share`` of the country's S1 addresses (regions/states);
  * record-only level: digit-free components found in >= ``rec_share`` of the country's S2/S3 addresses that
    (almost) never occur in S1 addresses (departments/counties used only by the other sources).
Pair features then compare addresses without these components (``strip_pairs``); cities and streets are kept.
Profiles with training labels are never touched, so their features stay byte-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..utils import load_json, log, save_json, work_dir


def labelled_profiles(cfg) -> set[str]:
    """Profiles that had training labels: from the tuned thresholds (inference-only runs) or the train split."""
    th = work_dir(cfg, None, "models", "thresholds.json")
    if th.exists():
        return {k for k in load_json(th) if not k.startswith("_")}
    s1 = work_dir(cfg, "train", "s1.parquet")
    if s1.exists():
        return set(pl.read_parquet(s1, columns=["prof"])["prof"].unique().to_list())
    return set()


def unlabelled_countries(cfg, split: str) -> set[str]:
    """Countries of ``split`` whose profile had no training labels (they get the adaptations in this module)."""
    trained = labelled_profiles(cfg)
    rec = pl.scan_parquet(work_dir(cfg, split, "records.parquet")).select(["country", "prof"]).unique().collect()
    return {c for c, p in rec.iter_rows() if p not in trained}


def density_ref(cfg) -> dict | None:
    """Train-scale medians of the r1 list-size counts (``models/density_ref.json``; shipped with the models, so an
    inference-only run on new data uses the same reference). Computed from the train candidates when missing."""
    path = work_dir(cfg, None, "models", "density_ref.json")
    if path.exists():
        return load_json(path)
    final = work_dir(cfg, "train", "cands", "final.parquet")
    if not final.exists():
        return None
    f = pl.read_parquet(final, columns=["s1_uid", "rec_uid", "s1_ncand", "rec_ncand"])
    f = f.with_columns([pl.len().over("s1_uid").alias("pre_s1_n"), pl.len().over("rec_uid").alias("pre_rec_n")])
    ref = {c: max(1.0, float(f[c].median() or 1.0)) for c in ("s1_ncand", "rec_ncand", "pre_s1_n", "pre_rec_n")}
    save_json(ref, path)
    log().info("  density reference (train medians of list-size counts): %s", ref)
    return ref


def density_match(pairs: pl.DataFrame, cols: list[str], ref: dict | None) -> pl.DataFrame:
    """Rescale list-size counts of one country so their median equals the train median (candidate lists of an
    unseen country can be much longer, e.g. France test 20.4 per S1 vs 10.8 in train: the r1 model was trained on
    the train scale). A no-op without a reference."""
    if not ref:
        return pairs
    present = [c for c in cols if c in pairs.columns and c in ref]
    med = {c: max(1.0, float(pairs[c].median() or 1.0)) for c in present}
    if present:
        log().info("   density match (profile without labels): %s", {c: f"{med[c]:.1f} -> {ref[c]:.1f}" for c in present})
    return pairs.with_columns([(pl.col(c) * (ref[c] / med[c])).cast(pl.Float32).alias(c) for c in present])


def _path(cfg, split: str) -> Path:
    return work_dir(cfg, split, "admin_vocab.json")


def build_admin_vocab(cfg, split: str) -> dict[str, list[str]]:
    """country -> admin components to ignore when comparing addresses (only for profiles without labels)."""
    ac = cfg.features.get("admin_strip") or {}
    if not ac.get("enabled", True):
        save_json({}, _path(cfg, split))
        return {}
    trained = labelled_profiles(cfg)
    rec = pl.scan_parquet(work_dir(cfg, split, "records.parquet")).select(["uid", "src", "country", "prof"])
    norm = pl.scan_parquet(work_dir(cfg, split, "norm.parquet")).select(["uid", "addr_comps"])
    out = {}
    for country, prof in rec.select(["country", "prof"]).unique().collect().sort("country").iter_rows():
        if prof in trained:
            continue
        d = (rec.filter(pl.col("country") == country).join(norm, on="uid")
             .select(["src", "addr_comps"]).collect())
        s1 = d.filter(pl.col("src") == 1)
        r = d.filter(pl.col("src") != 1)
        if s1.height == 0 or r.height == 0:
            continue
        last = (s1.filter(pl.col("addr_comps").list.len() > 0).select(pl.col("addr_comps").list.last().alias("c"))
                .group_by("c").len())
        s1_level = set(last.filter(pl.col("len") >= float(ac.get("s1_last_share", 0.02)) * s1.height)["c"].to_list())
        s1_df = s1.select(pl.col("addr_comps").list.unique().alias("c")).explode("c").group_by("c").len()
        r_df = r.select(pl.col("addr_comps").list.unique().alias("c")).explode("c").group_by("c").len()
        j = r_df.join(s1_df, on="c", how="left", suffix="_s1").with_columns(pl.col("len_s1").fill_null(0))
        rec_level = set(j.filter((pl.col("len") >= float(ac.get("rec_share", 0.005)) * r.height)
                                 & ~pl.col("c").str.contains(r"\d")
                                 & (pl.col("len_s1") / s1.height < float(ac.get("max_s1_ratio", 0.05))
                                    * pl.col("len") / r.height))["c"].to_list())
        vocab = sorted(c for c in s1_level | rec_level if c)
        if vocab:
            out[country] = vocab
            log().info("  %s/%s (no training labels): address admin level ignored in pair features: %s", split,
                       country, ", ".join(vocab[:12]) + (" ..." if len(vocab) > 12 else ""))
    save_json(out, _path(cfg, split))
    return out


def load_admin_vocab(cfg, split: str) -> dict[str, list[str]]:
    p = _path(cfg, split)
    return json.loads(p.read_text()) if p.exists() else build_admin_vocab(cfg, split)


def strip_exprs(prefix: str, vocab: list[str]) -> list[pl.Expr]:
    """Recompute ``{prefix}addr_comps/latin/tokens/missing`` without the admin components."""
    comps = pl.col(f"{prefix}addr_comps").list.eval(pl.element().filter(~pl.element().is_in(vocab)))
    return [comps.alias(f"{prefix}addr_comps")]


def strip_pairs(df: pl.DataFrame, vocab: list[str], prefixes=("s_", "r_")) -> pl.DataFrame:
    """Pair frame with ``s_``/``r_`` address columns -> the same columns with the admin level removed."""
    if not vocab:
        return df
    for pre in prefixes:
        if f"{pre}addr_comps" not in df.columns:
            continue
        df = df.with_columns(strip_exprs(pre, vocab))
        c = pl.col(f"{pre}addr_comps")
        df = df.with_columns([
            c.list.join(", ").alias(f"{pre}addr_latin"),
            c.list.join(" ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias(f"{pre}addr_tokens"),
            (c.list.len() == 0).alias(f"{pre}addr_missing"),
        ])
    return df


def strip_latin(norm: pl.DataFrame, vocab_by_country: dict[str, list[str]], country_of: pl.DataFrame) -> pl.DataFrame:
    """``norm`` (uid, addr_comps, addr_latin, ...) with addr_latin rebuilt without admin components for the
    countries in ``vocab_by_country``; ``country_of`` maps uid -> country."""
    if not vocab_by_country:
        return norm
    out = norm.join(country_of, on="uid", how="left")
    parts = []
    for country, g in out.group_by("country", maintain_order=True):
        v = vocab_by_country.get(country[0])
        if v:
            g = g.with_columns(pl.col("addr_comps").list.eval(pl.element().filter(~pl.element().is_in(v)))
                               .list.join(", ").alias("addr_latin"))
        parts.append(g)
    return pl.concat(parts).drop("country")
