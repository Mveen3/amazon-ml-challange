"""S1<->S1 diagnostic (docs §10.6): label-free false-positive tendency per country.

S1 is deduplicated, so every S1<->S1 pair is a true non-match. We retrieve each
S1's nearest S1 neighbours, compute the pair-intrinsic round-1 features and
score them with an auxiliary model trained on those same features (no context
features, which do not exist for S1<->S1 pairs). Higher scores for France than
for US/India suggest raising France thresholds. Diagnostic only; nothing is
trained on test.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..blocking.candidates import safe
from ..blocking.knn import topk
from ..blocking.prerank import attach_norm
from ..features import pair as P
from ..models.gbdt import GBDT, to_matrix
from ..utils import log, n_workers, save_json, work_dir

INTRINSIC = P.VEC_FEATURES + P.LOOP_FEATURES


def run_s1s1(cfg) -> dict:
    tr = P.load_r1(cfg, "train", columns=["s1_uid", "label"] + INTRINSIC)
    if tr.height > int(cfg.s1s1.train_rows):
        tr = tr.sample(int(cfg.s1s1.train_rows), seed=int(cfg.run.seed))
    aux = GBDT(cfg.s1s1.gbdt, INTRINSIC, seed=int(cfg.run.seed), threads=n_workers(cfg))
    aux.fit(to_matrix(tr, INTRINSIC), tr["label"].to_numpy())
    res = {}
    for split in ("train", "test"):
        rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src", "country"])
        s1p = pl.read_parquet(work_dir(cfg, split, "s1.parquet"), columns=["s1_uid", "prof"])
        norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=["uid"] + P.NORM_COLS)
        P._init(str(work_dir(cfg, split)), str(work_dir(cfg, None, "tables.pkl")),
                {"ngram": list(cfg.blocking.ngram), "hash_features": int(cfg.blocking.hash_features)})
        for country in rec["country"].unique().to_list():
            vdir = work_dir(cfg, split, "vec", safe(country))
            uids = np.load(vdir / "uids.npy")
            C = np.load(vdir / "comb_rp.npy")
            src = rec["src"].to_numpy()[uids]
            s1_pos = np.where(src == 1)[0]
            rng = np.random.default_rng(int(cfg.run.seed))
            q = rng.choice(s1_pos, min(len(s1_pos), int(cfg.s1s1.queries)), replace=False)
            idx, sim = topk(C[s1_pos], C[q], 2, cfg.blocking.knn.device, float(cfg.blocking.knn.mem_gb))
            nb = s1_pos[np.clip(idx[:, 1], 0, None)]  # column 0 is (almost always) the query itself
            pairs = pl.DataFrame({"s1_uid": uids[q], "rec_uid": uids[nb]}).filter(pl.col("s1_uid") != pl.col("rec_uid"))
            pairs = pairs.join(s1p, on="s1_uid").with_columns([pl.lit(country).alias("country"),
                                                               pl.lit(-1, dtype=pl.Int8).alias("fold")])
            feats = P.compute(attach_norm(pairs, norm, P.NORM_COLS))
            p = aux.predict(to_matrix(feats, INTRINSIC))
            res[f"{split}/{country}"] = {"n": int(len(p)), "mean": float(p.mean()), "p90": float(np.quantile(p, 0.9)),
                                         "p99": float(np.quantile(p, 0.99)), "frac_ge_0.5": float((p >= 0.5).mean())}
            log().info("  S1<->S1 %s/%s: %s", split, country, res[f"{split}/{country}"])
    save_json(res, work_dir(cfg, None, "models", "s1s1.json"))
    return res
