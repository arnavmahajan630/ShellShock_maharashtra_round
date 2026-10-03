"""Train the diagnoser (plans/03 §5.2–5.3), package M1.

    python -m ml.model.train                         # ml/data/dataset.jsonl -> ml/artifacts/
    python -m ml.model.train --data tests/fixtures/features_synth.npz --out <dir> [--no-grid]

Steps (03 §5.3):
  1. data comes in as `TrainData` (ml/model/data.py loads it; nothing here reads files);
  2. 5-fold StratifiedGroupKFold by problem, stratified by domain when the data has one;
     every model stops early on an inner grouped validation fold;
  3. the 12-config grid is scored by mean grouped-CV macro-F1 (tie: lower log-loss), with
     structural masking applied to the out-of-fold predictions exactly as at serving time;
  4. the chosen config gives the final out-of-fold logits (each fold refitted on its whole
     training part for the early-stopped number of rounds), then one model is fitted on everything;
  5. temperature on the OOF logits, novelty thresholds on OOF distances and OOF probabilities;
  6. `artifacts/diagnoser_<sha8>/{model.txt, meta.json}` is written.

Soft-label rows are trained as weighted duplicates (weight = soft_p × balanced class weight).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.model import calibrate as C
from ml.model import novelty as N
from ml.model.data import DEFAULT_DATASET, TrainData, class_weights, expand_soft, load
from ml.model.mask import allowed_classes, apply_mask

DEFAULT_ARTIFACTS = Path("ml/artifacts")
N_SPLITS = 5
INNER_SPLITS = 4                # the inner early-stopping fold is 1/4 of each training part
MAX_ROUNDS = 400
EARLY_STOPPING = 30
SEED = 42

BASE_PARAMS = dict(
    objective="multiclass", learning_rate=0.05, num_leaves=15, max_depth=5, min_data_in_leaf=20,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
    deterministic=True, force_row_wise=True,
    num_threads=4,              # measured faster than 12 on the ML laptop (06 §6)
)
GRID = [dict(num_leaves=nl, min_data_in_leaf=md, feature_fraction=ff)
        for nl, md, ff in itertools.product((7, 15, 31), (10, 30), (0.6, 0.9))]


# ---------------------------------------------------------------- folds

def make_folds(groups, domain=None, n_splits=N_SPLITS, seed=SEED):
    """Grouped folds by problem, stratified by domain (main / dsa) when it is known."""
    groups = np.asarray(groups)
    strata = np.zeros(len(groups), dtype=int) if domain is None else np.unique(np.asarray(domain), return_inverse=True)[1]
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(np.zeros(len(groups)), strata, groups)]


def inner_split(index, groups, domain=None, seed=SEED):
    """Split a training part into (fit, early-stopping) rows, again by problem."""
    index = np.asarray(index)
    sub_domain = None if domain is None else np.asarray(domain)[index]
    fit, stop = make_folds(np.asarray(groups)[index], sub_domain, n_splits=INNER_SPLITS, seed=seed)[0]
    return index[fit], index[stop]


# ---------------------------------------------------------------- one model

def _dataset(X, Y, weights, features, reference=None):
    Xe, ye, we, _ = expand_soft(X, Y, weights)
    return lgb.Dataset(Xe, label=ye, weight=we, feature_name=list(features), reference=reference, free_raw_data=False)


def fit_model(params, data, index, num_boost_round, stop_index=None):
    """Train on rows `index`. With `stop_index`, stop early on those rows and return the best round."""
    weights = class_weights(data.Y[index])
    train_set = _dataset(data.X[index], data.Y[index], weights, data.features)
    full = dict(BASE_PARAMS, num_class=len(data.labels), **params)
    if stop_index is None:
        return lgb.train(full, train_set, num_boost_round=num_boost_round), num_boost_round
    stop_set = _dataset(data.X[stop_index], data.Y[stop_index], weights, data.features, reference=train_set)
    booster = lgb.train(full, train_set, num_boost_round=num_boost_round, valid_sets=[stop_set],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return booster, max(int(booster.best_iteration or num_boost_round), 1)


def raw_logits(booster, X, num_iteration=None):
    return np.asarray(booster.predict(X, raw_score=True, num_iteration=num_iteration), dtype=np.float64)


# ---------------------------------------------------------------- scoring

def macro_f1(Y, y, pred):
    """Macro-F1 over the classes that occur in `y`. A prediction that names any label of a
    soft-labelled row counts as that row's label (a T1 row is right as M01 or as M08)."""
    hit = Y[np.arange(len(pred)), pred] > 0
    truth = np.where(hit, pred, y)
    return float(f1_score(truth, pred, labels=np.unique(y), average="macro", zero_division=0))


def score(logits, data, index, T=1.0, masked=True):
    probs = C.softmax(logits, T)
    if masked:
        probs = apply_mask(probs, allowed_classes(data.X[index], data.features, data.labels))
    pred = probs.argmax(axis=1)
    Y, y = data.Y[index], data.y[index]
    return {"macro_f1": macro_f1(Y, y, pred), "accuracy": float((Y[np.arange(len(pred)), pred] > 0).mean()),
            "log_loss": C.nll(C.softmax(logits, T), Y), "n": int(len(index))}


def cross_validate(params, data, folds, max_rounds=MAX_ROUNDS, refit=False, seed=SEED):
    """Out-of-fold logits for one config.

    refit=False  the early-stopped model (fitted on 3/4 of the training part) predicts the fold.
    refit=True   the fold's training part is refitted whole for the early-stopped number of
                 rounds before predicting; used for the final OOF of the chosen config.
    """
    oof = np.zeros((len(data), len(data.labels)))
    rounds, per_fold = [], []
    for train_idx, valid_idx in folds:
        fit_idx, stop_idx = inner_split(train_idx, data.groups, data.domain, seed)
        booster, best = fit_model(params, data, fit_idx, max_rounds, stop_idx)
        if refit:
            booster, _ = fit_model(params, data, train_idx, best)
            oof[valid_idx] = raw_logits(booster, data.X[valid_idx])
        else:
            oof[valid_idx] = raw_logits(booster, data.X[valid_idx], num_iteration=best)
        rounds.append(best)
        per_fold.append(score(oof[valid_idx], data, valid_idx))
    f1 = [fold["macro_f1"] for fold in per_fold]
    return {"oof_logits": oof, "rounds": rounds, "folds": per_fold,
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
            "log_loss_mean": float(np.mean([fold["log_loss"] for fold in per_fold]))}


def select_config(data, folds, grid=GRID, max_rounds=MAX_ROUNDS, log=None):
    """Grid search by mean grouped-CV macro-F1, ties broken by log-loss (03 §5.2)."""
    results = []
    for params in grid:
        cv = cross_validate(params, data, folds, max_rounds)
        results.append({"params": params, "macro_f1_mean": cv["macro_f1_mean"], "macro_f1_std": cv["macro_f1_std"],
                        "log_loss_mean": cv["log_loss_mean"], "rounds": cv["rounds"]})
        if log:
            log(f"  {params}  macro-F1 {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f}  "
                f"log-loss {cv['log_loss_mean']:.4f}  rounds {cv['rounds']}")
    best = min(results, key=lambda r: (-round(r["macro_f1_mean"], 6), r["log_loss_mean"]))
    return best["params"], results


# ---------------------------------------------------------------- the whole procedure

def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def train(data: TrainData, out_dir=DEFAULT_ARTIFACTS, *, grid=GRID, max_rounds=MAX_ROUNDS, n_splits=N_SPLITS,
          seed=SEED, log=print):
    """Run 03 §5.3 on `data` and write the artifact. Returns a summary dict (also in meta.json)."""
    log = log or (lambda *_: None)
    started = time.perf_counter()
    timings = {}
    folds = make_folds(data.groups, data.domain, n_splits, seed)
    log(f"rows {len(data)}  features {data.X.shape[1]}  problems {len(np.unique(data.groups))}  folds {len(folds)}")

    t = time.perf_counter()
    if grid and len(grid) > 1:
        log(f"grid: {len(grid)} configs")
        params, grid_results = select_config(data, folds, grid, max_rounds, log)
    else:
        params, grid_results = (dict(grid[0]) if grid else {}), []
    timings["grid_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    cv = cross_validate(params, data, folds, max_rounds, refit=True, seed=seed)
    timings["oof_s"] = round(time.perf_counter() - t, 2)
    oof = cv["oof_logits"]
    index = np.arange(len(data))

    t = time.perf_counter()
    final_rounds = max(int(math.ceil(float(np.mean(cv["rounds"])))), 1)
    booster, _ = fit_model(params, data, index, final_rounds)
    timings["final_fit_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    calibration = C.calibrate(oof, data.Y)
    T = calibration["temperature"]
    allowed = allowed_classes(data.X, data.features, data.labels)
    oof_probs = apply_mask(C.softmax(oof, T), allowed)
    correct = C.is_correct(oof_probs, data.Y)
    distances = N.oof_distances(data.X, folds, data.features, seed=seed)
    tau_d, tau_p = N.choose_tau_d(distances), N.choose_tau_p(oof_probs, correct)
    space = N.Novelty.fit(data.X, data.features, seed=seed)
    timings["calibrate_novelty_s"] = round(time.perf_counter() - t, 2)

    model_text = booster.model_to_string()
    sha8 = hashlib.sha256(model_text.encode("utf-8")).hexdigest()[:8]
    version = f"diagnoser_{sha8}"
    accepted = oof_probs.max(axis=1) >= tau_p
    summary = {
        "model_version": version,
        "classes": list(data.labels), "features": list(data.features),
        "temperature": T, "tau_p": tau_p, "tau_d": tau_d,
        "params": dict(BASE_PARAMS, num_class=len(data.labels), **params), "chosen": params,
        "num_boost_round": final_rounds,
        "cv": {
            "n_splits": len(folds), "stratified_by_domain": data.domain is not None,
            "macro_f1_mean": cv["macro_f1_mean"], "macro_f1_std": cv["macro_f1_std"],
            "folds": cv["folds"], "rounds": cv["rounds"],
            "oof_masked": score(oof, data, index, T, masked=True),
            "oof_unmasked": score(oof, data, index, T, masked=False),
            "grid": grid_results,
        },
        "calibration": calibration,
        "novelty_thresholds": {
            "tau_d_percentile": N.TAU_D_PERCENTILE, "target_precision": N.TARGET_PRECISION,
            "accepted_share": float(accepted.mean()),
            "accepted_precision": float(correct[accepted].mean()) if accepted.any() else None,
        },
        "n_rows": int(len(data)), "n_problems": int(len(np.unique(data.groups))),
        "data_hash": data.data_hash(), "git_commit": git_commit(),
        "lightgbm": lgb.__version__, "seed": seed, "timings": timings,
    }
    timings["total_s"] = round(time.perf_counter() - started, 2)

    folder = Path(out_dir) / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.txt").write_text(model_text, encoding="utf-8", newline="\n")
    meta = dict(summary, novelty=space.to_meta())
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")

    log(f"chosen {params}  rounds {final_rounds}")
    log(f"grouped-CV macro-F1 {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f} (masked, per fold)")
    log(f"OOF masked: macro-F1 {summary['cv']['oof_masked']['macro_f1']:.4f}  acc {summary['cv']['oof_masked']['accuracy']:.4f}"
        f"   unmasked: macro-F1 {summary['cv']['oof_unmasked']['macro_f1']:.4f}")
    log(f"temperature {T:.3f}  ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}"
        f"  tau_p {tau_p:.2f}  tau_d {tau_d:.3f}")
    log(f"saved {folder}  ({timings['total_s']} s)")
    return dict(summary, path=str(folder), oof_logits=oof, oof_distances=distances,
                final_logits=raw_logits(booster, data.X))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the diagnoser (03 §5).")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="dataset.jsonl or a feature matrix .npz")
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACTS), help="folder for diagnoser_<sha8>/")
    parser.add_argument("--no-grid", action="store_true", help="train the default config only (no 12-config grid)")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = parser.parse_args(argv)
    if not Path(args.data).exists():
        print(f"{args.data} does not exist. Build it with `make data` (package C3), or pass --data <matrix.npz>.",
              file=sys.stderr)
        return 2
    data = load(args.data)
    if str(args.data).endswith("features_synth.npz"):
        print("NOTE: this is the made-up matrix (random columns, one telltale column per class). "
              "The scores below only show that the code runs.")
    train(data, args.out, grid=[{}] if args.no_grid else GRID, max_rounds=args.max_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
