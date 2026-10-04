"""Shared helpers for E17 and E18 (package E-c). Nothing here is an experiment.

E16 (e16_text.py) was written first and does not use this file.
"""
from __future__ import annotations

import os

# Set before numpy / onnxruntime load: this package keeps CPU jobs to 4 threads.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ml" / "eval"
N_BOOT = 1000
SEED = 20261004

LLM_CAVEAT = (
    "Every reason sentence used here is LLM-written. The 270 'persona' sentences are role-played students "
    "(a different model and prompt from the training sentences); no real student sentence was collected. "
    "Results on real phrasing are not measured."
)


def limit_threads(n=4):
    """Cap onnxruntime sessions created later in this process. The reader code builds its sessions with
    default options, so the constructor is wrapped here (this process only; no repo file is touched)."""
    import onnxruntime

    if getattr(onnxruntime, "_ec_limited", False):
        return
    original = onnxruntime.InferenceSession

    def limited(path, sess_options=None, *args, **kwargs):
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
        return original(path, sess_options, *args, **kwargs)

    onnxruntime.InferenceSession = limited
    onnxruntime._ec_limited = True


def read_jsonl(relative):
    path = ROOT / relative
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cluster_bootstrap(per_trial, clusters, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% interval of the mean of `per_trial`, resampling whole clusters.

    `per_trial` is a 1-D array (or a function of an index array); `clusters` gives each trial's cluster.
    Returns None if there are fewer than two clusters.
    """
    clusters = list(clusters)
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    members = [np.array(v) for v in groups.values()]
    if len(members) < 2:
        return None
    rng = np.random.default_rng(seed)
    stat = per_trial if callable(per_trial) else (lambda ix, a=np.asarray(per_trial, dtype=float): a[ix].mean())
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        values.append(stat(np.concatenate([members[j] for j in pick])))
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def row_bootstrap(per_trial, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% interval of the mean, resampling single trials."""
    a = np.asarray(per_trial, dtype=float)
    rng = np.random.default_rng(seed)
    values = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def today():
    return datetime.date.today().isoformat()


def write_card(card, stem, markdown):
    """ml/eval/<stem>_card.json and .md, next to the script (the layout E16 uses)."""
    card = dict(card)
    card.setdefault("plots", [])
    card.setdefault("generated_at", today())
    (EVAL / f"{stem}_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EVAL / f"{stem}_card.md").write_text(markdown, encoding="utf-8")
    return EVAL / f"{stem}_card.json"


def md_table(header, lines):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in line) + " |" for line in lines] + [""]


def ci_text(ci):
    return "-" if not ci else f"{ci[0]:.2f} to {ci[1]:.2f}"
