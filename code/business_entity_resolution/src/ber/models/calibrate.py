"""Per-profile isotonic calibration fitted on OOF predictions.

Profiles without training data (e.g. France) use ``fallback`` (default: us).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self, fallback: str = "us", min_rows: int = 1000):
        self.models: dict[str, IsotonicRegression] = {}
        self.fallback = fallback
        self.min_rows = min_rows

    def fit(self, p: np.ndarray, y: np.ndarray, prof: np.ndarray) -> "Calibrator":
        ok = ~np.isnan(p)
        for name in np.unique(prof[ok]):
            m = ok & (prof == name)
            if m.sum() >= self.min_rows and 0 < y[m].sum() < m.sum():
                self.models[str(name)] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[m], y[m])
        if self.fallback not in self.models and self.models:
            self.fallback = max(self.models, key=lambda k: len(self.models[k].X_thresholds_))
        return self

    def transform(self, p: np.ndarray, prof: np.ndarray) -> np.ndarray:
        out = p.astype(np.float64).copy()
        if not self.models:
            return out.astype(np.float32)
        for name in np.unique(prof):
            m = prof == name
            model = self.models.get(str(name), self.models.get(self.fallback))
            out[m] = model.predict(p[m])
        return out.astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Calibrator":
        with open(path, "rb") as f:
            return pickle.load(f)
