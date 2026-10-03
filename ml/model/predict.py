"""Load a saved diagnoser and score feature rows (package M1).

    model = Diagnoser.load("ml/artifacts/diagnoser_3f9a12cd")      # or Diagnoser.latest()
    p_code = model.proba(row)            # calibrated (temperature) and structurally masked
    novelty = model.novelty(row, p_code[0])

The Bayes layer (package D1) takes `p_code` from here; `decide.py` takes its posterior.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ml.model import novelty as N
from ml.model.calibrate import softmax
from ml.model.mask import allowed_classes, apply_mask

DEFAULT_ARTIFACTS = Path("ml/artifacts")


class Diagnoser:
    def __init__(self, booster, meta):
        self.booster = booster
        self.meta = meta
        self.model_version = meta["model_version"]
        self.labels = list(meta["classes"])
        self.features = list(meta["features"])
        self.temperature = float(meta["temperature"])
        self.tau_p = float(meta["tau_p"])
        self.tau_d = float(meta["tau_d"])
        self.space = N.Novelty.from_meta(meta["novelty"])

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_str=(folder / "model.txt").read_text(encoding="utf-8"))
        return cls(booster, meta)

    @classmethod
    def latest(cls, root=DEFAULT_ARTIFACTS):
        """The most recently written `diagnoser_*` folder under `root`."""
        folders = [p for p in Path(root).glob("diagnoser_*") if (p / "meta.json").exists()]
        if not folders:
            raise FileNotFoundError(f"no diagnoser_* artifact under {root}; run `python -m ml.model.train`")
        return cls.load(max(folders, key=lambda p: (p / "meta.json").stat().st_mtime))

    def _rows(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features, got {X.shape[1]}")
        return X

    def logits(self, X):
        return np.asarray(self.booster.predict(self._rows(X), raw_score=True), dtype=np.float64)

    def proba(self, X, masked=True):
        """softmax(logits / T), then structural masking (03 §4.1b). Shape (n, n_labels)."""
        X = self._rows(X)
        probs = softmax(self.logits(X), self.temperature)
        if masked:
            probs = apply_mask(probs, allowed_classes(X, self.features, self.labels))
        return probs

    def posterior_dict(self, probs):
        return {name: float(p) for name, p in zip(self.labels, probs)}

    def knn_distance(self, X):
        return self.space.distance(self._rows(X))

    def novelty(self, x, probs=None):
        """The `novelty` object (03 §11.1) for one feature row."""
        x = self._rows(x)[:1]
        probs = self.proba(x)[0] if probs is None else np.asarray(probs)
        return N.assess(self.knn_distance(x)[0], probs, self.tau_d, self.tau_p, self.labels)

    def contributions(self, x):
        """SHAP-style contributions, shape (n_labels, n_features + 1); the last column is the bias."""
        raw = np.asarray(self.booster.predict(self._rows(x)[:1], pred_contrib=True), dtype=np.float64)
        return raw.reshape(len(self.labels), len(self.features) + 1)
