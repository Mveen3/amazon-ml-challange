"""Competition / context features derived from a score over the candidate graph."""
from __future__ import annotations

import polars as pl


def score_context(df: pl.DataFrame, score: str, prefix: str) -> pl.DataFrame:
    """Ranks, gaps, runner-up and share of ``score`` from both the S1 and the record side.

    ``{prefix}_rec_comp`` is the best score this record has with any *other* S1,
    the key signal for the one-owner competition.
    """
    s = pl.col(score)
    df = df.with_columns([
        s.rank("ordinal", descending=True).over("s1_uid").cast(pl.Int16).alias(f"{prefix}_s1_rank"),
        s.rank("ordinal", descending=True).over("rec_uid").cast(pl.Int16).alias(f"{prefix}_rec_rank"),
    ])
    s1_top2 = s.filter(pl.col(f"{prefix}_s1_rank") > 1).max().over("s1_uid")
    rec_top2 = s.filter(pl.col(f"{prefix}_rec_rank") > 1).max().over("rec_uid")
    df = df.with_columns([
        (s.max().over("s1_uid") - s).alias(f"{prefix}_s1_gap"),
        (s.max().over("rec_uid") - s).alias(f"{prefix}_rec_gap"),
        pl.when(pl.col(f"{prefix}_s1_rank") == 1).then(s1_top2.fill_null(0.0)).otherwise(s.max().over("s1_uid"))
        .alias(f"{prefix}_s1_other"),
        pl.when(pl.col(f"{prefix}_rec_rank") == 1).then(rec_top2.fill_null(0.0)).otherwise(s.max().over("rec_uid"))
        .alias(f"{prefix}_rec_comp"),
        (s / (s.sum().over("rec_uid") + 1e-6)).alias(f"{prefix}_rec_share"),
        (s / (s.sum().over("s1_uid") + 1e-6)).alias(f"{prefix}_s1_share"),
        pl.len().over("s1_uid").cast(pl.Int32).alias(f"{prefix}_s1_n"),
        pl.len().over("rec_uid").cast(pl.Int32).alias(f"{prefix}_rec_n"),
        (s >= 0.5).sum().over("rec_uid").cast(pl.Int16).alias(f"{prefix}_rec_n50"),
        (s >= 0.5).sum().over("s1_uid").cast(pl.Int16).alias(f"{prefix}_s1_n50"),
    ])
    return df.with_columns((s - pl.col(f"{prefix}_rec_comp")).alias(f"{prefix}_rec_margin"))


def density_normalize(df: pl.DataFrame, cols: list[str], by: str | None = None) -> pl.DataFrame:
    """Divide list-size counts by their median (over ``df``, or per group ``by``, e.g. the country profile).

    Candidate lists are about twice as long for France test (20.4 per S1) as in train (10.8); raw counts would put
    every France pair outside the range the models were trained on. After this, a value of 1.0 means "a typical
    list for this country and split". A median below 1 is treated as 1 (empty lists).
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    if by is None:
        med = {c: max(1.0, float(df[c].median() or 1.0)) for c in present}
        return df.with_columns([(pl.col(c) / med[c]).cast(pl.Float32).alias(c) for c in present])
    return df.with_columns([(pl.col(c) / pl.col(c).median().over(by).clip(lower_bound=1.0)).cast(pl.Float32).alias(c)
                            for c in present])


def context_names(prefix: str) -> list[str]:
    return [f"{prefix}_{x}" for x in ("s1_rank", "rec_rank", "s1_gap", "rec_gap", "s1_other", "rec_comp",
                                      "rec_share", "s1_share", "s1_n", "rec_n", "rec_n50", "s1_n50", "rec_margin")]
