"""GBDT wrapper (LightGBM preferred, XGBoost fallback) + entity-grouped OOF training.

OOF protocol: rows carry the fold of their S1 entity; fold ``k`` is predicted by
a model trained on the other folds. Early stopping uses a held-out 5% of the
*training* entities, never the predicted fold. Training entities can be
subsampled per fold (whole entities, so candidate lists stay intact).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils import ensure_dir, log


def _backend(name: str) -> str:
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
            return "lightgbm"
        except ImportError:
            log().warning("lightgbm not installed -> falling back to xgboost")
            return "xgboost"
    return name


class GBDT:
    def __init__(self, gcfg, features: list[str], monotone: list[str] | None = None, seed: int = 42,
                 threads: int = -1):
        self.backend = _backend(gcfg.backend)
        self.params = dict(gcfg.params)
        self.rounds = int(gcfg.rounds)
        self.early_stop = int(gcfg.early_stop)
        self.features = list(features)
        self.monotone = [1 if f in set(monotone or []) else 0 for f in self.features]
        self.seed = seed
        self.threads = threads
        self.model = None
        self.best_iter = None

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, y, Xv=None, yv=None):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            p = {"objective": "binary", "verbose": -1, "seed": self.seed, "num_threads": self.threads,
                 "deterministic": True, "force_col_wise": True, **self.params}
            if any(self.monotone):
                p["monotone_constraints"] = self.monotone
                p["monotone_constraints_method"] = "advanced"
            dtr = lgb.Dataset(X, y, feature_name=self.features, free_raw_data=True)
            valid, cbs = [], [lgb.log_evaluation(200)]
            if Xv is not None and len(Xv):
                valid = [lgb.Dataset(Xv, yv, reference=dtr)]
                cbs.append(lgb.early_stopping(self.early_stop, verbose=False))
            self.model = lgb.train(p, dtr, num_boost_round=self.rounds, valid_sets=valid, callbacks=cbs)
            self.best_iter = self.model.best_iteration or self.model.current_iteration()
        else:
            import xgboost as xgb

            from ..utils import torch_device

            pp = self.params
            p = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                 "eta": pp.get("learning_rate", 0.05), "max_leaves": pp.get("num_leaves", 255),
                 "grow_policy": "lossguide", "max_depth": 0, "subsample": pp.get("bagging_fraction", 0.8),
                 "colsample_bytree": pp.get("feature_fraction", 0.8), "lambda": pp.get("lambda_l2", 1.0),
                 "max_bin": pp.get("max_bin", 255), "seed": self.seed, "nthread": self.threads,
                 "device": "cuda" if torch_device("auto").type == "cuda" else "cpu"}
            if any(self.monotone):
                p["monotone_constraints"] = "(" + ",".join(map(str, self.monotone)) + ")"
            dtr = xgb.DMatrix(X, y, feature_names=self.features, missing=np.nan)
            evals = []
            if Xv is not None and len(Xv):
                evals = [(xgb.DMatrix(Xv, yv, feature_names=self.features, missing=np.nan), "val")]
            self.model = xgb.train(p, dtr, self.rounds, evals=evals, verbose_eval=200,
                                   early_stopping_rounds=self.early_stop if evals else None)
            self.best_iter = getattr(self.model, "best_iteration", None)
        return self

    def predict(self, X) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, np.float32)
        if self.backend == "lightgbm":
            return self.model.predict(X, num_iteration=self.best_iter).astype(np.float32)
        import xgboost as xgb

        d = xgb.DMatrix(X, feature_names=self.features, missing=np.nan)
        rng = (0, self.best_iter + 1) if self.best_iter is not None else (0, 0)
        return self.model.predict(d, iteration_range=rng).astype(np.float32)

    def importance(self) -> dict:
        if self.backend == "lightgbm":
            return dict(zip(self.features, self.model.feature_importance("gain").tolist()))
        return self.model.get_score(importance_type="gain")

    # ------------------------------------------------------------------ io
    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        ensure_dir(prefix.parent)
        if self.backend == "lightgbm":
            self.model.save_model(str(prefix) + ".txt", num_iteration=self.best_iter)
        else:
            self.model.save_model(str(prefix) + ".json")
        meta = {"backend": self.backend, "features": self.features, "best_iter": self.best_iter}
        Path(str(prefix) + ".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, prefix: Path) -> "GBDT":
        meta = json.loads(Path(str(prefix) + ".meta.json").read_text())
        obj = cls.__new__(cls)
        obj.backend, obj.features, obj.best_iter = meta["backend"], meta["features"], meta["best_iter"]
        if obj.backend == "lightgbm":
            import lightgbm as lgb

            obj.model = lgb.Booster(model_file=str(prefix) + ".txt")
            obj.best_iter = None  # file already truncated to best iteration
        else:
            import xgboost as xgb

            obj.model = xgb.Booster()
            obj.model.load_model(str(prefix) + ".json")
        return obj


def to_matrix(df, features: list[str]) -> np.ndarray:
    import polars as pl

    return df.select([pl.col(f).cast(pl.Float32) for f in features]).to_numpy()


def train_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, ents: np.ndarray, gcfg, features: list[str],
              out_dir: Path, seed: int = 42, threads: int = -1, monotone=None) -> tuple[np.ndarray, list[GBDT]]:
    """Entity-grouped OOF training. Returns OOF predictions (NaN where fold == -1) and fold models."""
    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    models = []
    for k in sorted(int(f) for f in np.unique(folds) if f >= 0):
        tr = folds != k
        pool = np.unique(ents[tr])
        if len(pool) > int(gcfg.max_entities):
            pool = rng.choice(pool, int(gcfg.max_entities), replace=False)
        pool = rng.permutation(pool)
        n_val = max(1, int(len(pool) * float(gcfg.es_frac)))
        fit_m = tr & np.isin(ents, pool[n_val:])
        val_m = tr & np.isin(ents, pool[:n_val])
        log().info("   fold %d: fit %d rows (%d pos), val %d rows", k, fit_m.sum(), int(y[fit_m].sum()), val_m.sum())
        m = GBDT(gcfg, features, monotone=monotone, seed=seed + k, threads=threads)
        m.fit(X[fit_m], y[fit_m], X[val_m], y[val_m])
        pred_m = folds == k
        oof[pred_m] = m.predict(X[pred_m])
        m.save(Path(out_dir) / f"fold{k}")
        models.append(m)
    imp = {}
    for m in models:
        for f, v in m.importance().items():
            imp[f] = imp.get(f, 0.0) + float(v)
    ensure_dir(Path(out_dir))
    (Path(out_dir) / "importance.json").write_text(json.dumps(dict(sorted(imp.items(), key=lambda x: -x[1])), indent=1))
    return oof, models


def load_folds(out_dir: Path) -> list[GBDT]:
    return [GBDT.load(p.with_suffix("").with_suffix("")) for p in sorted(Path(out_dir).glob("fold*.meta.json"))]


def predict_mean(models: list[GBDT], X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0).astype(np.float32) if len(X) else np.zeros(0, np.float32)


def predict_by_fold(models: list[GBDT], X: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Train rows: model of their own fold (OOF). Rows with fold -1: mean of all models."""
    out = np.empty(len(X), dtype=np.float32)
    for k, m in enumerate(models):
        sel = folds == k
        if sel.any():
            out[sel] = m.predict(X[sel])
    rest = folds < 0
    if rest.any():
        out[rest] = predict_mean(models, X[rest])
    return out
