"""E3 — unseen surface forms (03 §9.2).

For each class with at least two base operators, hold one operator out of TRAIN, retrain,
and score accuracy on that operator's rows.
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import LABELS
from ml.eval._eb_data import STAMP, load_bundle, make_train_data
from ml.eval._eb_metrics import (
    bar_with_intervals, cluster_mean_ci, new_card, read_json, write_card, write_json,
)
from ml.eval._eb_splits import plan_jobs
from ml.eval._eb_train import fit_diagnoser, load_model
from ml.eval._eb_data import OUT


def _score(bundle, model, test_index):
    probs = model.proba(bundle.X_full[test_index])
    pred = probs.argmax(axis=1)
    y = bundle.y[test_index]
    soft = bundle.Y[test_index, pred] > 0
    primary = pred == y
    return pred, soft.astype(float), primary.astype(float)


def _run_job(bundle, job):
    dest = OUT / "e03" / job["class"]
    cached = read_json(dest / "result.json")
    if cached and cached.get("stamp") == STAMP:
        print(f"[E3 {job['class']}] reuse", flush=True)
        return cached
    data = make_train_data(bundle, job["train_index"])
    summary = fit_diagnoser(data, dest, f"E3 {job['class']} without {job['op_id']}")
    model = load_model(summary)
    _, soft, primary = _score(bundle, model, job["test_index"])
    problems = [str(p) for p in bundle.problem_id[job["test_index"]]]
    result = {
        "stamp": STAMP,
        "class": job["class"],
        "op_id": job["op_id"],
        "n_base_ops": job["n_base_ops"],
        "n_test": job["n_test"],
        "n_train": job["n_train"],
        "dropped_for_hash": job["dropped_for_hash"],
        "model_version": summary["model_version"],
        "seconds": summary["seconds"],
        "accuracy": cluster_mean_ci(soft, problems, seed=42),
        "primary_accuracy": cluster_mean_ci(primary, problems, seed=43),
        "n_correct_soft": int(soft.sum()),
        "n_correct_primary": int(primary.sum()),
        "hits_soft": soft.astype(int).tolist(),
        "hits_primary": primary.astype(int).tolist(),
        "problems": problems,
    }
    write_json(dest / "result.json", result)
    return result


def _mean_of(results, key):
    estimates = [row[key]["estimate"] for row in results]
    return float(np.mean(estimates)) if estimates else None


def run():
    bundle = load_bundle()
    jobs, skipped = plan_jobs(bundle)
    results = [_run_job(bundle, job) for job in jobs]
    hits, problems = [], []
    hits_p, problems_p = [], []
    for row in results:
        hits.extend(row["hits_soft"])
        problems.extend(row["problems"])
        hits_p.extend(row["hits_primary"])
        problems_p.extend(row["problems"])
    names = [row["class"] for row in results]
    estimates = [row["accuracy"]["estimate"] for row in results]
    plot = bar_with_intervals(
        names, estimates,
        [row["accuracy"]["lo"] for row in results],
        [row["accuracy"]["hi"] for row in results],
        "accuracy on the held-out operator",
        "E3  held-out operator accuracy (soft-label hit)",
        "e03_operator_holdout.png",
    )
    per_class = {}
    for row in results:
        per_class[row["class"]] = {
            "op_id": row["op_id"], "n": row["n_test"], "n_train": row["n_train"],
            "n_base_ops": row["n_base_ops"], "dropped_for_hash": row["dropped_for_hash"],
            "accuracy": row["accuracy"], "primary_accuracy": row["primary_accuracy"],
            "model_version": row["model_version"],
        }
    card = new_card(
        "E3",
        "Unseen surface forms",
        "OPHOLD: one base operator removed from TRAIN per class with at least two; "
        "test rows are that operator on the training problems",
        n=int(sum(row["n_test"] for row in results)),
        metrics={
            "mean_class_accuracy": {
                "estimate": _mean_of(results, "accuracy"),
                "per_class_min": None if not estimates else float(np.min(estimates)),
                "per_class_max": None if not estimates else float(np.max(estimates)),
                "n_classes": len(results),
            },
            "mean_class_primary_accuracy": {"estimate": _mean_of(results, "primary_accuracy")},
            "micro_accuracy": cluster_mean_ci(hits, problems, seed=44),
            "micro_primary_accuracy": cluster_mean_ci(hits_p, problems_p, seed=45),
        },
        caveat=(
            "Synthetic rows from the same generator as training. Accuracy counts a soft-label "
            "row as correct when the prediction is either of its labels (the training definition); "
            "primary_accuracy requires the row's primary label. The held-out operator is the "
            "lexicographically last base operator with at least 8 TRAIN rows. Two-bug rows that "
            "contain it are removed from training and are not in the test. Default LightGBM "
            "config, no grid. Classes with one base operator are skipped, not scored as zero."
        ),
        per_class=per_class,
        plots=[plot],
        extra={"skipped": skipped, "n_classes_scored": len(results), "n_classes_defined": len(LABELS)},
    )
    write_card(card, "e03_card.json")
    return card


if __name__ == "__main__":
    run()
