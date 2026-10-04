"""E5 — structural pair accuracy and hard-twin probing (03 §9.2).

STRUCTURAL twins are scored on HOLDOUT-P and R-llm. HARD twins (soft-labelled T1) are
scored for an ambiguity flag, then probed. The engine keeps p_b = 0.6; the simulated
learner's true p_b is swept from 0.4 to 0.9. R-llm is an LLM-written stand-in, not R-blind.
"""
from __future__ import annotations

import numpy as np

from ml.bayes.posterior import prior
from ml.contracts.classes import LABELS, TWIN_SETS
from ml.eval._eb_data import (
    N_BOOT, OUT, SEED, bank_train_mask, holdout_mask, load_bundle, rllm_mask,
)
from ml.eval._eb_metrics import cluster_mean_ci, new_card, write_card
from ml.eval._eb_probe import N_REPS, P_B_GRID, argmax_label, load_probe_bank, probe_once, rng_for
from ml.eval._eb_train import ensure_full, load_model
from ml.model.decide import decide
from ml.model.novelty import assess
from ml.bayes.eig import make_best_probe


def _structural_sets():
    sets = []
    for set_id, info in TWIN_SETS.items():
        members = info["members"]
        if info["type"] == "STRUCTURAL" and all(member in LABELS for member in members):
            sets.append((set_id, tuple(members)))
    return sets


def _pair_hit(posterior, true, twin):
    left, right = float(posterior[true]), float(posterior[twin])
    if left > right:
        return 1.0
    if left < right:
        return 0.0
    return 1.0 if LABELS.index(true) < LABELS.index(twin) else 0.0


def _mean_of_sets(records):
    by_set = {}
    for record in records:
        by_set.setdefault(record["set"], []).append(record["hit"])
    if not by_set:
        return None
    return float(np.mean([np.mean(values) for values in by_set.values()]))


def _ci_mean_of_sets(records, seed):
    point = _mean_of_sets(records)
    n_sets = len({record["set"] for record in records})
    if point is None:
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_sets": 0, "n_clusters": 0}
    by_problem = {}
    for record in records:
        by_problem.setdefault(record["problem"], []).append(record)
    keys = list(by_problem)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(N_BOOT):
        chosen = rng.integers(0, len(keys), size=len(keys))
        sample = []
        for pick in chosen:
            sample.extend(by_problem[keys[pick]])
        value = _mean_of_sets(sample)
        if value is not None:
            stats.append(value)
    lo = hi = None
    if len(stats) >= 100:
        lo, hi = (float(v) for v in np.quantile(stats, [0.025, 0.975]))
    return {"estimate": point, "lo": lo, "hi": hi, "n": len(records), "n_sets": n_sets,
            "n_clusters": len(keys)}


def _filter(records, domain=None, prefix=None, slice_name=None):
    out = records
    if domain is not None:
        out = [record for record in out if record["domain"] == domain]
    if prefix is not None:
        out = [record for record in out if record["label"].startswith(prefix)]
    if slice_name is not None:
        out = [record for record in out if record["slice"] == slice_name]
    return out


def _pack(records, seed):
    per_set = {}
    for set_id in sorted({record["set"] for record in records}):
        sub = [record for record in records if record["set"] == set_id]
        per_set[set_id] = cluster_mean_ci(
            [record["hit"] for record in sub], [record["problem"] for record in sub], seed=seed)
    return {
        "mean_of_sets": _ci_mean_of_sets(records, seed),
        "micro": cluster_mean_ci(
            [record["hit"] for record in records], [record["problem"] for record in records], seed=seed + 1)
        if records else {"estimate": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0},
        "per_set": per_set,
    }


def _eval_index(bundle):
    """HOLDOUT-P plus R-llm, dropping any R-llm ast_hash that already appears in TRAIN."""
    index = np.flatnonzero(holdout_mask(bundle) | rllm_mask(bundle))
    train_hashes = set(str(h) for h in bundle.ast_hash[bank_train_mask(bundle)] if str(h))
    keep, dropped = [], 0
    for i in index:
        if str(bundle.kind[i]) == "rllm" and str(bundle.ast_hash[i]) in train_hashes:
            dropped += 1
            continue
        keep.append(int(i))
    n_gate = int(((bundle.kind == "rllm") & (bundle.y < 0)).sum())
    return np.asarray(keep, dtype=np.int64), dropped, n_gate


def _predict(model, bundle, index):
    X = bundle.X_full[index]
    probs = model.proba(X)
    dists = model.knn_distance(X)
    posts, novels = [], []
    for i in range(len(index)):
        posts.append(prior(model.posterior_dict(probs[i])))
        novels.append(assess(float(dists[i]), probs[i], model.tau_d, model.tau_p, model.labels))
    return posts, novels


def _structural_records(bundle, index, posts):
    records = []
    for local, row in enumerate(index):
        label = str(bundle.label[row])
        for set_id, members in _structural_sets():
            if label not in members:
                continue
            twin = members[0] if members[1] == label else members[1]
            top = argmax_label(posts[local])
            records.append({
                "set": set_id, "hit": _pair_hit(posts[local], label, twin),
                "top_hit": 1.0 if top == label else 0.0,
                "problem": str(bundle.problem_id[row]), "domain": str(bundle.domain[row]),
                "label": label, "slice": "rllm" if str(bundle.kind[row]) == "rllm" else "holdout",
                "row": int(row),
            })
    return records


def _hard_rows(bundle, index):
    m01, m08 = LABELS.index("M01"), LABELS.index("M08")
    soft = (bundle.Y[index, m01] > 0) & (bundle.Y[index, m08] > 0)
    named = np.isin(bundle.label[index], np.array(["M01", "M08"]))
    return soft, named & ~soft


def _simulate(bundle, index, local_rows, posts, novels, probes, by_id):
    """Per true p_b, one accuracy per row (mean over N_REPS) and the code-only flag."""
    flags, pres, problems, domains, slices = [], [], [], [], []
    asked = {p_b: [] for p_b in P_B_GRID}
    post_hits = {p_b: [] for p_b in P_B_GRID}
    for local in local_rows:
        row = int(index[local])
        label = str(bundle.label[row])
        diagnosis = decide(
            posts[local], tests_passed=bool(bundle.tests_passed[row]), novelty=novels[local],
            best_probe=make_best_probe([], probes), probes_asked=[],
        )
        flags.append(1.0 if diagnosis["status"] == "ambiguous" else 0.0)
        pres.append(1.0 if argmax_label(posts[local]) == label else 0.0)
        problems.append(str(bundle.problem_id[row]))
        domains.append(str(bundle.domain[row]))
        slices.append("rllm" if str(bundle.kind[row]) == "rllm" else "holdout")
        for p_b in P_B_GRID:
            hits, n_asked = [], []
            for rep in range(N_REPS):
                pred, n_probes, _ = probe_once(
                    posts[local], label, p_b, rng_for(str(bundle.id[row]), p_b, rep),
                    probes, by_id, novels[local], bool(bundle.tests_passed[row]),
                )
                hits.append(1.0 if pred == label else 0.0)
                n_asked.append(n_probes)
            post_hits[p_b].append(float(np.mean(hits)))
            asked[p_b].append(float(np.mean(n_asked)))
    return {
        "flag": np.asarray(flags), "pre": np.asarray(pres), "problem": np.asarray(problems),
        "domain": np.asarray(domains), "slice": np.asarray(slices),
        "post": {p_b: np.asarray(values) for p_b, values in post_hits.items()},
        "asked": {p_b: np.asarray(values) for p_b, values in asked.items()},
    }


def _subset_ci(values, problems, mask, seed):
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    if not np.any(mask):
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    return cluster_mean_ci(values[mask], problems[mask], seed=seed)


def _plot(pre, posts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(P_B_GRID)

    def _num(value):
        return np.nan if value is None else value

    est = [_num(posts[p]["estimate"]) for p in xs]
    lo = [_num(posts[p]["lo"] if posts[p]["lo"] is not None else posts[p]["estimate"]) for p in xs]
    hi = [_num(posts[p]["hi"] if posts[p]["hi"] is not None else posts[p]["estimate"]) for p in xs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].plot(xs, est, marker="o", color="#3c6e71")
    axes[0].fill_between(xs, lo, hi, color="#3c6e71", alpha=0.2)
    if pre["estimate"] is not None:
        axes[0].axhline(pre["estimate"], color="#c44536", linestyle="--", label="pre-probe")
        axes[0].legend(frameon=False)
    axes[0].set_xlabel("simulated learner true p_b")
    axes[0].set_ylabel("accuracy after the engine's probes")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("E5  post-probe accuracy")
    labels = ["pre"] + [f"post {p:.1f}" for p in xs]
    heights = [pre["estimate"] or 0] + est
    axes[1].bar(np.arange(len(labels)), heights, color=["#c44536"] + ["#3c6e71"] * len(xs))
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("E5  pre-probe vs post-probe")
    fig.tight_layout()
    path = OUT / "e05_twins.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return "e05_twins.png"


def run():
    bundle = load_bundle()
    summary = ensure_full(bundle)
    model = load_model(summary)
    index, n_hash_drop, n_gate = _eval_index(bundle)
    posts, novels = _predict(model, bundle, index)
    records = _structural_records(bundle, index, posts)
    pair = {
        "overall": _pack(records, seed=SEED),
        "holdout": _pack(_filter(records, slice_name="holdout"), seed=SEED + 2),
        "rllm": _pack(_filter(records, slice_name="rllm"), seed=SEED + 3),
        "main": _pack(_filter(records, domain="main"), seed=SEED + 4),
        "dsa": _pack(_filter(records, domain="dsa"), seed=SEED + 5),
        "M": _pack(_filter(records, prefix="M"), seed=SEED + 6),
        "D": _pack(_filter(records, prefix="D"), seed=SEED + 7),
    }
    probes, by_id = load_probe_bank()
    soft, hard_named = _hard_rows(bundle, index)
    sim = _simulate(bundle, index, np.flatnonzero(soft), posts, novels, probes, by_id)
    post = {}
    asked = {}
    for offset, p_b in enumerate(P_B_GRID):
        post[f"{p_b:.1f}"] = _subset_ci(sim["post"][p_b], sim["problem"], None, seed=SEED + 10 + offset)
        asked[f"{p_b:.1f}"] = _subset_ci(sim["asked"][p_b], sim["problem"], None, seed=SEED + 20 + offset)
    domains = {
        name: _subset_ci(sim["flag"], sim["problem"], sim["domain"] == name, seed=SEED + 30)
        for name in ("main", "dsa")
    }
    pre = _subset_ci(sim["pre"], sim["problem"], None, seed=SEED + 8)
    flag = _subset_ci(sim["flag"], sim["problem"], None, seed=SEED + 9)
    hard_local = np.flatnonzero(hard_named)
    hard_flags = []
    hard_problems = []
    for local in hard_local:
        row = int(index[local])
        diagnosis = decide(
            posts[local], tests_passed=bool(bundle.tests_passed[row]), novelty=novels[local],
            best_probe=make_best_probe([], probes), probes_asked=[],
        )
        hard_flags.append(1.0 if diagnosis["status"] == "ambiguous" else 0.0)
        hard_problems.append(str(bundle.problem_id[row]))
    plot = _plot(pre, {p_b: post[f"{p_b:.1f}"] for p_b in P_B_GRID})
    card = new_card(
        "E5",
        "Twins",
        "HOLDOUT-P + R-llm. STRUCTURAL: pair accuracy (true class beats its twin). "
        "HARD: soft-labelled T1 rows, ambiguity flag, then simulated probes.",
        n=int(len(index)),
        metrics={
            "pair_accuracy": pair["overall"]["mean_of_sets"],
            "pair_accuracy_micro": pair["overall"]["micro"],
            "pair_accuracy_by_slice": {
                "holdout": pair["holdout"]["mean_of_sets"],
                "rllm": pair["rllm"]["mean_of_sets"],
                "main": pair["main"]["mean_of_sets"],
                "dsa": pair["dsa"]["mean_of_sets"],
                "M": pair["M"]["mean_of_sets"],
                "D": pair["D"]["mean_of_sets"],
            },
            "ambiguity_flag_rate": flag,
            "ambiguity_flag_rate_by_domain": domains,
            "pre_probe_accuracy": pre,
            "post_probe_accuracy": post,
            "mean_probes_asked": asked,
            "hard_labelled_T1_flag_rate": cluster_mean_ci(hard_flags, hard_problems, seed=SEED + 40)
            if hard_flags else {"estimate": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0},
        },
        caveat=(
            "Pair accuracy is the unweighted mean of per-twin-set accuracies: within the pair, "
            "the true class has the higher probability after the population prior. "
            "R-llm is 58 LLM-written programs (notes/T1), not the hand-written R-team and not "
            "R-blind, which was not opened. The engine updates with p_b = 0.6; each simulated "
            f"learner draws {N_REPS} times at the true p_b, with q and s left at the engine values. "
            "A row is flagged ambiguous only when decide() says so, including the novelty check. "
            "The fixer is not called. Soft T1 rows are scored against their primary label, the "
            "class the simulated learner believes."
        ),
        per_class=pair["overall"]["per_set"],
        plots=[plot],
        extra={
            "model_version": summary["model_version"],
            "n_holdout": int(holdout_mask(bundle).sum()),
            "n_rllm": int(rllm_mask(bundle).sum()),
            "n_rllm_hash_overlap_dropped": n_hash_drop,
            "n_gate_excluded": n_gate,
            "n_soft_t1": int(soft.sum()),
            "n_hard_labelled_t1": int(hard_named.sum()),
            "n_structural_evaluations": len(records),
            "reps": N_REPS,
        },
    )
    write_card(card, "e05_card.json")
    return card


if __name__ == "__main__":
    run()
