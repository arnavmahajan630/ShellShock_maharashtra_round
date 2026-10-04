"""Retrains for E-b. Every artifact goes under ml/eval/out/e-b/, never ml/artifacts/.

Each fit is the M1 procedure with the default LightGBM config (no 12-config grid).
06 §6 budgets about 7 minutes for all evaluation fits; a grid inside every retrain is 12x that.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path

import numpy as np

from ml.model.calibrate import softmax
from ml.model.mask import allowed_classes, apply_mask
from ml.model.predict import Diagnoser
from ml.model.train import train
from ml.eval._eb_data import OUT, STAMP, bank_train_mask, make_train_data
from ml.eval._eb_metrics import read_json, write_json


def _ready(dest, need_oof):
    summary = read_json(Path(dest) / "summary.json")
    if not summary or summary.get("stamp") != STAMP:
        return None
    folder = Path(summary["path"])
    if not (folder / "model.txt").exists() or not (folder / "meta.json").exists():
        return None
    if need_oof and not (Path(dest) / "oof.npz").exists():
        return None
    return summary


def fit_diagnoser(data, out_dir, name, save_oof=False, oof_index=None):
    """Train one diagnoser. Returns a small summary; the model lives in `path`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _ready(out_dir, need_oof=save_oof)
    if cached:
        print(f"[{name}] reuse {cached['model_version']}", flush=True)
        return cached
    print(f"[{name}] fitting {len(data)} rows, {data.X.shape[1]} features", flush=True)
    summary = train(data, out_dir, grid=[{}], log=lambda *_: None)
    brief = {
        "stamp": STAMP,
        "name": name,
        "model_version": summary["model_version"],
        "path": summary["path"],
        "temperature": summary["temperature"],
        "tau_p": summary["tau_p"],
        "tau_d": summary["tau_d"],
        "n_rows": summary["n_rows"],
        "num_boost_round": summary["num_boost_round"],
        "seconds": summary["timings"]["total_s"],
        "oof_macro_f1": summary["cv"]["oof_masked"]["macro_f1"],
        "oof_accuracy": summary["cv"]["oof_masked"]["accuracy"],
    }
    if save_oof:
        probs = apply_mask(
            softmax(summary["oof_logits"], summary["temperature"]),
            allowed_classes(data.X, data.features, data.labels),
        )
        np.savez_compressed(
            out_dir / "oof.npz",
            index=np.asarray(oof_index if oof_index is not None else np.arange(len(data)), dtype=np.int64),
            probs=probs.astype(np.float32),
            y=np.asarray(data.y, dtype=np.int64),
            domain=np.asarray(data.domain),
            problem=np.asarray(data.groups),
        )
    write_json(out_dir / "summary.json", brief)
    print(f"[{name}] {brief['model_version']} in {brief['seconds']}s  "
          f"OOF macro-F1 {brief['oof_macro_f1']:.3f}", flush=True)
    return brief


def ensure_split(bundle, name, index, dest, save_oof=False):
    dest = OUT / dest
    cached = _ready(dest, need_oof=save_oof)
    if cached:
        print(f"[{name}] reuse {cached['model_version']}", flush=True)
        return cached
    data = make_train_data(bundle, index)
    return fit_diagnoser(data, dest, name, save_oof=save_oof, oof_index=index)


def ensure_full(bundle):
    """Diagnoser fit on all of TRAIN. Shared by E5 and E15(b)."""
    return ensure_split(bundle, "E15/E5 full TRAIN", np.flatnonzero(bank_train_mask(bundle)), "full", save_oof=True)


def load_oof(dest):
    data = np.load(OUT / dest / "oof.npz", allow_pickle=False)
    return {key: data[key] for key in ("index", "probs", "y", "domain", "problem")}


def load_model(summary):
    return Diagnoser.load(summary["path"])
