"""Run 03 §10 and write docs/eda.md.

Groups B and R are not in the matrix: dataset rows store ``trace_summary``, not a full trace.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ml.contracts.feature_names import GROUP_A, GROUP_C
from ml.eda.checks import (
    check_counts,
    check_drops,
    check_duplicates,
    check_events,
    check_exposure,
    check_health,
    check_length,
    check_masking,
    check_outliers,
    check_predicates,
    check_projection,
    adversarial_auc,
    check_realistic,
    check_shift,
    check_stumps,
    load_jsonl,
)
from ml.eda.report import (
    body_counts,
    body_domain,
    body_drops,
    body_duplicates,
    body_events,
    body_exposure,
    body_health,
    body_length,
    body_outliers,
    body_predicates,
    body_projection,
    body_shift,
    body_stumps,
    render,
)
from ml.features.ast_feats import ast_features
from ml.features.extract import task_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "data" / "dataset.jsonl"
LUCK = ROOT / "ml" / "data" / "passes_by_luck.jsonl"
RLLM = ROOT / "ml" / "data" / "realistic_llm.jsonl"
ITSP_CANDIDATES = (
    ROOT / "ml" / "data" / "itsp_slice.jsonl",
    ROOT / "ml" / "data" / "external" / "itsp_slice.jsonl",
)
REPORT = ROOT / "docs" / "eda.md"
_LABEL_KEYS = ("label", "soft_label", "labels_all", "rater2_label", "belief")
_SIGNATURE = re.compile(
    r"(?m)^[ \t]*((?:unsigned\s+|signed\s+)?(?:void|int|float|double|char|long|short)(?:\s+\w+)*)\s+"
    r"(\w+)\s*\(([^;{}]*)\)\s*\{"
)
FIGURES = Path(__file__).resolve().parent / "figures"


def load_problems():
    found = {}
    for folder in ("main", "dsa"):
        for path in sorted((ROOT / "ml" / "problems" / folder).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            found[data["problem_id"]] = data
    return found


def count_attempts(problems):
    """Site counts. Same loop as ``ml.generate.build_dataset`` without applying mutants."""
    from ml.generate.registry import allowed, find_sites, operators

    ops = operators()
    attempts = {op.op_id: 0 for op in ops}
    for problem in problems.values():
        use = [op for op in ops if allowed(op, problem.get("allowed_ops"))]
        for code in problem.get("correct_variants") or []:
            for op in use:
                attempts[op.op_id] += len(find_sites(op, code))
    return attempts


def load_inputs(path):
    """External rows with class labels removed. Adversarial validation must not see them."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in _LABEL_KEYS:
                row.pop(key, None)
            rows.append(row)
    return rows


def find_itsp():
    for path in ITSP_CANDIDATES:
        if path.is_file():
            return path
    return None


def _signature_from_code(code):
    found = _SIGNATURE.findall(code or "")
    if not found:
        return None
    picked = next((item for item in found if item[1] != "main"), found[0])
    returns, name, args = picked
    return f"{returns} {name}({args.strip()})"


def feature_matrix(rows, problems, *, signature_fallback=False):
    names = list(GROUP_A) + list(GROUP_C)
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    parse_fail = 0
    width = len(GROUP_A)
    for index, row in enumerate(rows):
        try:
            feats, meta = ast_features(row.get("code") or "")
        except Exception:
            feats, meta = {}, {"parse_ok": False}
        if not meta.get("parse_ok"):
            parse_fail += 1
        for column, name in enumerate(GROUP_A):
            value = feats.get(name, np.nan)
            matrix[index, column] = np.nan if value is None else value
        problem = problems.get(row.get("problem_id"))
        if problem is None and signature_fallback:
            signature = _signature_from_code(row.get("code") or "")
            if signature:
                problem = {"signature": signature, "tests": []}
        try:
            coarse = task_features(problem) if problem else {}
        except Exception:
            coarse = {}
        for column, name in enumerate(GROUP_C):
            value = coarse.get(name, np.nan)
            matrix[index, width + column] = np.nan if value is None else value
        if index and index % 2000 == 0:
            print(f"features {index}/{len(rows)}", flush=True)
    return names, matrix, parse_fail


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=100)
    return name


def plot_heatmap(matrix, row_labels, col_labels, title, name, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.array(matrix, dtype=float)
    fig_w = max(8.0, 0.38 * len(col_labels) + 2)
    fig_h = max(5.0, 0.32 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)
    return name


def plot_length(by_label, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml.contracts.classes import LABELS

    labels = [label for label in LABELS if by_label.get(label)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.violinplot([by_label[label] for label in labels], showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("len(code)")
    ax.set_title("Code length by class")
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)
    return name


def plot_scatter(xy, labels, title, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.45, label=label)
    ax.legend(fontsize=7, ncol=2, markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)
    return name


def _scope(n_rows, n_train, n_features, parse_fail, matplotlib_ok):
    plots = "Plots are under `ml/eda/figures/`." if matplotlib_ok else "matplotlib could not draw plots; tables only."
    return "\n".join([
        "## Scope",
        "",
        f"Rows: **{n_rows}**. Train rows (`split=train`): **{n_train}**. "
        f"Feature matrix: **{n_features}** columns, group A then group C. "
        f"AST parse failures: **{parse_fail}**.",
        "",
        "Group B and group R are not computed. A dataset row stores `trace_summary` "
        "(status, event names, `loop_iters_delta`), not the trace those features read. "
        "Re-running the interpreter on every row is not part of this package.",
        "",
        "Stumps, feature health, PCA, and IsolationForest use train rows. "
        "Counts, hashes, length, events, exposure, and masking use every row.",
        "",
        plots + " `docs/eda/*.png` is not written: the only `docs/` file this package writes is `docs/eda.md`.",
        "",
        "UMAP is not installed. PCA is the projection.",
    ])


def _try_plot(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        print(f"plot skipped: {exc}", flush=True)
        return None


def _external_counts(rows, path, problems, parse_fail):
    families = Counter(row.get("family") or "(none)" for row in rows)
    n_bank = sum(1 for row in rows if row.get("problem_id") in problems)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "n": len(rows),
        "n_problems": len({row.get("problem_id") for row in rows}),
        "n_bank": n_bank,
        "n_signature": len(rows) - n_bank,
        "parse_fail": parse_fail,
        "families": [[name, count] for name, count in families.most_common()],
    }


def _score_external(path, problems, names, X_train, groups_train):
    rows = load_inputs(path)
    print(f"N1: adversarial {path.name} ({len(rows)} rows)", flush=True)
    _, matrix, parse_fail = feature_matrix(rows, problems, signature_fallback=True)
    groups = [str(row.get("problem_id") or f"{path.stem}-{index}") for index, row in enumerate(rows)]
    result = adversarial_auc(X_train, groups_train, matrix, groups, names)
    result["counts"] = _external_counts(rows, path, problems, parse_fail)
    return result


def run(report_path=None):
    print("N1: loading rows", flush=True)
    rows = load_jsonl(DATA)
    luck_rows = load_jsonl(LUCK) if LUCK.is_file() else []
    problems = load_problems()
    exposures = {
        problem_id: problem.get("exposure") or {}
        for problem_id, problem in problems.items()
        if problem_id.startswith("Q")
    }
    print("N1: counting operator sites", flush=True)
    attempts = count_attempts(problems)
    print("N1: extracting group A and C", flush=True)
    names, matrix, parse_fail = feature_matrix(rows, problems)

    counts = check_counts(rows)
    drops = check_drops(rows, luck_rows, attempts)
    duplicates = check_duplicates(rows)
    length = check_length(rows)
    events = check_events(rows)
    exposure = check_exposure(rows, luck_rows, exposures)
    realistic = check_realistic(rows)

    train = [index for index, row in enumerate(rows) if row.get("split") == "train"]
    train_rows = [rows[index] for index in train]
    X_train = matrix[train]
    groups_train = [str(rows[index].get("problem_id") or f"train-{index}") for index in train]
    if RLLM.is_file():
        rllm = _score_external(RLLM, problems, names, X_train, groups_train)
    else:
        rllm = {"auc": None, "headline": "realistic_llm.jsonl was not found", "top": []}
    itsp_path = find_itsp()
    if itsp_path is not None:
        itsp = _score_external(itsp_path, problems, names, X_train, groups_train)
    else:
        itsp = {"auc": None, "headline": "itsp_slice.jsonl was not found", "top": []}
    shift = check_shift(rllm, itsp)
    y_all = np.array([row.get("label") for row in rows])
    y_train = y_all[train]
    print("N1: stumps, health, predicates, masking", flush=True)
    stumps = check_stumps(names, X_train, y_train)
    health = check_health(names, X_train)
    predicates = check_predicates(names, matrix, y_all)
    masking = check_masking(
        names,
        matrix,
        y_all.tolist(),
        [{"id": row.get("id"), "label": row.get("label"), "problem_id": row.get("problem_id")} for row in rows],
    )
    print("N1: PCA and outliers", flush=True)
    projection = check_projection(
        X_train,
        y_train,
        np.array([row.get("source") or "" for row in train_rows]),
        np.array([row.get("ast_hash") or "" for row in train_rows]),
    )
    outliers = check_outliers(X_train, train_rows)
    projection.setdefault("rule", "classes overlapping heavily is expected; otherwise look at features")

    matplotlib_ok = True
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        matplotlib_ok = False

    plots = {}
    if matplotlib_ok:
        plots["counts"] = _try_plot(
            plot_heatmap,
            counts["class_problem"], list(counts["per_class"]), counts["problems"],
            "Class x problem counts", "class_problem.png",
        )
        plots["source"] = _try_plot(
            plot_heatmap,
            counts["class_source"], list(counts["per_class"]), counts["sources"],
            "Class x source counts", "class_source.png",
        )
        plots["domain"] = _try_plot(
            plot_heatmap,
            [row[:2] for row in counts["class_domain"]], list(counts["per_class"]), ["main", "dsa"],
            "Class x domain counts", "domain.png",
        )
        if length.get("by_label"):
            plots["length"] = _try_plot(plot_length, length["by_label"], "length_violin.png")
        if predicates.get("matrix"):
            numeric = [[np.nan if value is None else value for value in row] for row in predicates["matrix"]]
            plots["predicates"] = _try_plot(
                plot_heatmap,
                numeric, predicates["row_labels"], predicates["col_labels"],
                "Mean of group-A defining features", "predicate.png",
            )
        if projection.get("xy") is not None:
            plots["pca_class"] = _try_plot(
                plot_scatter, projection["xy"], projection["y"], "PCA by class (train)", "pca_class.png",
            )
            plots["pca_source"] = _try_plot(
                plot_scatter, projection["xy"], projection["sources"], "PCA by source (train)", "pca_source.png",
            )
        if events.get("events"):
            rate = []
            for label in counts["per_class"]:
                denom = counts["per_class"][label] or 1
                rate.append([
                    events["per_event_class"].get(event, {}).get(label, 0) / denom
                    for event in events["events"]
                ])
            plots["events"] = _try_plot(
                plot_heatmap,
                rate, list(counts["per_class"]), events["events"],
                "Event rate within class", "events.png", 0.0, 1.0,
            )

    def _plots(*keys):
        found = []
        captions = {
            "counts": "Class by problem",
            "source": "Class by source",
            "domain": "Class by domain",
            "length": "Code length by class",
            "predicates": "Predicate means",
            "pca_class": "PCA coloured by class",
            "pca_source": "PCA coloured by source",
            "events": "Trace-event rate by class",
        }
        for key in keys:
            name = plots.get(key)
            if name:
                found.append((name, captions[key]))
        return found

    stump_rule = (
        f"non-defining stump macro-acc or accuracy > {0.60:.2f}, "
        "or code length separates a class from CORRECT"
    )
    if length["status"] == "FAIL" or stumps["status"] == "FAIL":
        stump_status = "FAIL"
    elif length["status"] == "SKIP" and stumps["status"] == "SKIP":
        stump_status = "SKIP"
    else:
        stump_status = "PASS"
    stump_headline = f"Stumps: {stumps['headline']}. Length: {length['headline']}"

    masking_status = masking["status"]
    masking_headline = (
        f"{masking['headline']}. Domain main={counts['n_main']} dsa={counts['n_dsa']}"
    )

    sections = [
        {
            "id": "10.1", "title": "Counts", "rule": counts["rule"],
            "status": counts["status"], "headline": counts["headline"],
            "body": body_counts(counts), "plots": _plots("counts", "source"),
        },
        {
            "id": "10.2", "title": "Verification", "rule": drops["rule"],
            "status": drops["status"], "headline": drops["headline"],
            "body": body_drops(drops), "plots": [],
        },
        {
            "id": "10.3", "title": "Duplicates", "rule": duplicates["rule"],
            "status": duplicates["status"], "headline": duplicates["headline"],
            "body": body_duplicates(duplicates), "plots": [],
        },
        {
            "id": "10.4", "title": "Shortcut scan", "rule": stump_rule,
            "status": stump_status, "headline": stump_headline,
            "body": body_stumps(stumps) + "\n\n" + body_length(length),
            "plots": _plots("length"),
        },
        {
            "id": "10.5", "title": "Predicate matrix", "rule": predicates["rule"],
            "status": predicates["status"], "headline": predicates["headline"],
            "body": body_predicates(predicates), "plots": _plots("predicates"),
        },
        {
            "id": "10.6", "title": "Feature health", "rule": health["rule"],
            "status": health["status"], "headline": health["headline"],
            "body": body_health(health, parse_fail, len(rows)), "plots": [],
        },
        {
            "id": "10.7", "title": "Distribution shift", "rule": shift["rule"],
            "status": shift["status"], "headline": shift["headline"],
            "body": body_shift(shift),
            "plots": [],
        },
        {
            "id": "10.8", "title": "Projection", "rule": projection.get("rule") or shift["rule"],
            "status": projection["status"], "headline": projection["headline"],
            "body": body_projection(projection), "plots": _plots("pca_class", "pca_source"),
        },
        {
            "id": "10.9", "title": "Outliers", "rule": outliers["rule"],
            "status": outliers["status"], "headline": outliers["headline"],
            "body": body_outliers(outliers), "plots": [],
        },
        {
            "id": "10.10", "title": "Trace events", "rule": events["rule"],
            "status": events["status"], "headline": events["headline"],
            "body": body_events(events), "plots": _plots("events"),
        },
        {
            "id": "10.11", "title": "Realistic set", "rule": realistic["rule"],
            "status": realistic["status"], "headline": realistic["headline"],
            "body": (
                "The realistic set in 03 §10.11 is R-blind and R-team, with a second rater. "
                "There is no second rater, so kappa is not computed. "
                "The LLM-written stand-in for R-team, not hand-written, is scored only in 10.7, "
                "and its labels are not used here."
            ),
            "plots": [],
        },
        {
            "id": "10.12", "title": "Exposure", "rule": exposure["rule"],
            "status": exposure["status"], "headline": exposure["headline"],
            "body": body_exposure(exposure), "plots": [],
        },
        {
            "id": "10.13", "title": "Domain and masking", "rule": "any masked true label",
            "status": masking_status, "headline": masking_headline,
            "body": body_domain(counts, masking), "plots": _plots("domain"),
        },
    ]
    # 10.8 rule text if projection skipped early
    if "rule" not in projection:
        sections[7]["rule"] = "classes overlapping heavily is expected; otherwise look at features"

    scope = _scope(len(rows), len(train), len(names), parse_fail, matplotlib_ok and any(plots.values()))
    text = render(scope, sections)
    path = Path(report_path) if report_path else REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}", flush=True)
    for section in sections:
        print(f"{section['id']} {section['status']}: {section['headline']}", flush=True)
    return path


def main():
    run()
    return 0
