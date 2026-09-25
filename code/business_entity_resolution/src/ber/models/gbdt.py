"""GBDT wrapper (LightGBM preferred, XGBoost fallback) + entity-grouped OOF training.

OOF protocol: rows carry the fold of their S1 entity; fold ``k`` is predicted by
a model trained on the other folds. Early stopping uses a held-out 5% of the
*training* entities, never the predicted fold. Training entities can be
subsampled per fold (whole entities, so candidate lists stay intact).
"""
from __future__ import annotations

import json
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..utils import ensure_dir, gpu_cap, log

_GPU_OK: bool | None = None


def xgb_gpu_available() -> bool:
    """True if this XGBoost build has CUDA and torch sees a GPU (cached)."""
    global _GPU_OK
    if _GPU_OK is None:
        try:
            import torch
            import xgboost as xgb

            _GPU_OK = bool(torch.cuda.is_available() and xgb.build_info().get("USE_CUDA"))
        except Exception:  # noqa: BLE001
            _GPU_OK = False
    return _GPU_OK


def gpu_ids(gcfg) -> list[int]:
    """GPUs that can train XGBoost fold models concurrently ([] for LightGBM / CPU / a pinned single device)."""
    if _backend(gcfg.backend) != "xgboost" or str(gcfg.get("device", "auto")) not in ("auto", "cuda"):
        return []
    if not xgb_gpu_available():
        return []
    import torch

    n = torch.cuda.device_count()
    for cap in (int(gcfg.get("max_gpus", 0)), gpu_cap()):
        if cap > 0:
            n = min(n, cap)
    return list(range(n))


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
        self.device = str(gcfg.get("device", "auto"))
        self.model = None
        self.best_iter = None
        self._pred_ready = False

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
            max_bin = int(pp.get("max_bin", 255))
            p = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                 "eta": pp.get("learning_rate", 0.05), "max_leaves": pp.get("num_leaves", 255),
                 "grow_policy": "lossguide", "max_depth": 0, "subsample": pp.get("bagging_fraction", 0.8),
                 "colsample_bytree": pp.get("feature_fraction", 0.8), "lambda": pp.get("lambda_l2", 1.0),
                 "max_bin": max_bin, "seed": self.seed, "nthread": max(1, self.threads),
                 "device": self.device if self.device != "auto" else
                 ("cuda" if torch_device("auto").type == "cuda" and xgb.build_info().get("USE_CUDA") else "cpu")}
            if any(self.monotone):
                p["monotone_constraints"] = "(" + ",".join(map(str, self.monotone)) + ")"
            # QuantileDMatrix: pre-binned, far less host/GPU memory than DMatrix
            dtr = xgb.QuantileDMatrix(X, y, feature_names=self.features, missing=np.nan, max_bin=max_bin)
            evals = []
            if Xv is not None and len(Xv):
                evals = [(xgb.QuantileDMatrix(Xv, yv, ref=dtr, feature_names=self.features, missing=np.nan,
                                              max_bin=max_bin), "val")]
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

        if not self._pred_ready:  # a model loaded from disk predicts on the CPU unless told otherwise
            self._pred_ready = True
            if str(self.device) in ("auto", "cuda") and xgb_gpu_available():
                self.model.set_param({"device": "cuda:0"})
        d = xgb.DMatrix(X, feature_names=self.features, missing=np.nan)
        rng = (0, self.best_iter + 1) if self.best_iter is not None else (0, 0)
        try:
            return self.model.predict(d, iteration_range=rng).astype(np.float32)
        except xgb.core.XGBoostError as e:  # e.g. GPU out of memory: fall back to the CPU for this model
            log().warning("   xgboost GPU predict failed (%s) -> CPU", str(e).splitlines()[0][:120])
            self.model.set_param({"device": "cpu"})
            self.device = "cpu"
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
        obj.device, obj._pred_ready = "auto", False
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


def _fold_splits(folds, ents, gcfg, seed: int):
    """(fold, fit mask, validation mask) per fold. Uses the RNG in fold order, so it is identical whether the
    models are then trained one after another or concurrently."""
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(int(f) for f in np.unique(folds) if f >= 0):
        tr = folds != k
        pool = np.unique(ents[tr])
        if len(pool) > int(gcfg.max_entities):
            pool = rng.choice(pool, int(gcfg.max_entities), replace=False)
        pool = rng.permutation(pool)
        n_val = max(1, int(len(pool) * float(gcfg.es_frac)))
        out.append((k, tr & np.isin(ents, pool[n_val:]), tr & np.isin(ents, pool[:n_val])))
    return out


def fit_folds(X: np.ndarray, y: np.ndarray, folds: np.ndarray, ents: np.ndarray, gcfg, features: list[str],
              out_dir: Path, seed: int = 42, threads: int = -1, monotone=None) -> list[GBDT]:
    """One model per fold k, trained on rows whose entity fold != k (entity-grouped, early stopping on a
    held-out slice of the *training* entities). ``models[k]`` never saw fold k, so it gives OOF scores.

    With XGBoost on several GPUs the fold models are trained concurrently, one per GPU (they are independent);
    if that fails (e.g. out of memory) the folds that did not finish are trained one after another on GPU 0.
    """
    splits = _fold_splits(folds, ents, gcfg, seed)
    gpus = gpu_ids(gcfg)
    workers = len(gpus) if len(gpus) >= 2 and len(splits) >= 2 else 1

    def train(sp, gpu: int | None):
        k, fit_m, val_m = sp
        log().info("   fold %d: fit %d rows (%d pos), val %d rows%s", k, fit_m.sum(), int(y[fit_m].sum()), val_m.sum(),
                   f" on cuda:{gpu}" if gpu is not None and workers > 1 else "")
        m = GBDT(gcfg, features, monotone=monotone, seed=seed + k, threads=max(1, threads // workers) if threads > 0 else threads)
        if gpu is not None and workers > 1:
            m.device = f"cuda:{gpu}"
        m.fit(X[fit_m], y[fit_m], X[val_m], y[val_m])
        m.save(Path(out_dir) / f"fold{k}")
        return m

    done: dict[int, GBDT] = {}
    if workers > 1:
        free: queue.SimpleQueue = queue.SimpleQueue()
        for g in gpus[:workers]:
            free.put(g)

        def run(sp):
            g = free.get()
            try:
                return train(sp, g)
            finally:
                free.put(g)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {sp[0]: ex.submit(run, sp) for sp in splits}
        for k, f in futs.items():
            try:
                done[k] = f.result()
            except Exception as e:  # noqa: BLE001
                log().warning("   fold %d failed on a GPU (%s: %s) -> retrying it on one GPU", k, type(e).__name__,
                              str(e).splitlines()[0][:120])
    models = []
    for sp in splits:
        models.append(done[sp[0]] if sp[0] in done else train(sp, gpus[0] if gpus else None))
    imp = {}
    for m in models:
        for f, v in m.importance().items():
            imp[f] = imp.get(f, 0.0) + float(v)
    ensure_dir(Path(out_dir))
    (Path(out_dir) / "importance.json").write_text(json.dumps(dict(sorted(imp.items(), key=lambda x: -x[1])), indent=1))
    return models


def train_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, ents: np.ndarray, gcfg, features: list[str],
              out_dir: Path, seed: int = 42, threads: int = -1, monotone=None) -> tuple[np.ndarray, list[GBDT]]:
    """In-memory variant (small tables, e.g. the entity gate): fit fold models and return OOF predictions."""
    models = fit_folds(X, y, folds, ents, gcfg, features, out_dir, seed, threads, monotone)
    return predict_by_fold(models, X, folds), models


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
