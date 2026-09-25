"""Build a small, realistic sample dataset for smoke-testing the pipeline.

train: N S1 entities + all their matched records + distractor records (records
       of other S1s / unmatched records, i.e. plausible non-matches).
test:  a disjoint set of S1 entities built the same way; a fraction of US
       clusters is relabelled "France" to exercise the unseen-country path.
       Its ground truth is written to test_truth.tsv (never read by the pipeline).

    python scripts/make_sample.py --data ../../dataset --out sample_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../dataset")
    ap.add_argument("--out", default="sample_data")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--distractor-frac", type=float, default=0.35)
    ap.add_argument("--france-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = Path(a.data) / "train"
    s1 = read(d / "train_source1.tsv")
    recs = pl.concat([read(d / "train_source2.tsv"), read(d / "train_source3.tsv")])
    gt = read(d / "train_ground_truth.tsv")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").rename({"matched_entity_ids": "entity_id"}))
    perm = rng.permutation(s1.height)
    groups = {"train": s1[perm[:a.n_train]], "test": s1[perm[a.n_train:a.n_train + a.n_test]]}
    out = Path(a.out)
    for split, g in groups.items():
        ids = g["entity_id"]
        p = pairs.filter(pl.col("source1_entity_id").is_in(ids.implode()))
        matched = recs.filter(pl.col("entity_id").is_in(p["entity_id"].implode()))
        pool = recs.filter(~pl.col("entity_id").is_in(p["entity_id"].implode()))
        n_dis = int(matched.height * a.distractor_frac)
        # distractors: records of other (absent) S1s or unmatched records -> plausible non-matches
        dis = pool.sample(n=min(pool.height, n_dis), seed=a.seed + (split == "test"))
        g_out, r_out = g, pl.concat([matched, dis])
        if split == "test" and a.france_frac > 0:
            us = g.filter(pl.col("country") == "US")["entity_id"]
            fr = us.sample(fraction=a.france_frac, seed=a.seed)
            fr_recs = p.filter(pl.col("source1_entity_id").is_in(fr.implode()))["entity_id"]
            g_out = g.with_columns(pl.when(pl.col("entity_id").is_in(fr.implode())).then(pl.lit("France"))
                                   .otherwise(pl.col("country")).alias("country"))
            r_out = r_out.with_columns(pl.when(pl.col("entity_id").is_in(fr_recs.implode())).then(pl.lit("France"))
                                       .otherwise(pl.col("country")).alias("country"))
        sd = out / split
        sd.mkdir(parents=True, exist_ok=True)
        g_out.write_csv(sd / f"{split}_source1.tsv", separator="\t", quote_style="never")
        for src in ("S2", "S3"):
            r_out.filter(pl.col("entity_id").str.starts_with(src)).sample(fraction=1.0, shuffle=True, seed=a.seed) \
                .write_csv(sd / f"{split}_source{src[1]}.tsv", separator="\t", quote_style="never")
        truth = gt.filter(pl.col("source1_entity_id").is_in(ids.implode()))
        name = "train_ground_truth.tsv" if split == "train" else "../test_truth.tsv"
        truth.write_csv(sd / name, separator="\t", quote_style="never")
        print(f"{split}: {g.height} S1, {matched.height} matched records, {dis.height} distractors")


if __name__ == "__main__":
    main()
