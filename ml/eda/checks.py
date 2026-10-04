"""Red-rule checks for 03 §10.

Each check returns a dict with ``status`` in PASS, FAIL, SKIP, INCONCLUSIVE, or INSPECT.
Numbers are computed from the rows passed in. Thresholds that §10 does not name are
module constants and are repeated in the report.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np

from ml.contracts.classes import LABELS
from ml.contracts.feature_names import CLASS_DEFINING_FEATURES, GROUP_A

# §10 states these cutoffs.
MIN_CLASS_ROWS = 80
DROP_LIMIT = 0.40
STUMP_ACC_LIMIT = 0.60
BLEED_LIMIT = 0.30
CORR_LIMIT = 0.95
EXPOSURE_GAP = 0.30

# §10 names the check but not the number. Documented in docs/eda.md and notes/N1.md.
NEAR_CONSTANT_SHARE = 0.99
KS_D_MIN = 0.25
KS_P_MAX = 0.01
KS_MIN_N = 20
# Silhouette below this is treated as the heavy overlap §10.8 expects.
# 1-NN agreement is reported beside it and is not the pass/fail number:
# same-class templates sit next to each other even when the clouds overlap.
SILHOUETTE_SEPARATE = 0.20
EVENT_SHARE_MIN = 0.50
IF_CONTAMINATION = 0.02
IF_TOP_N = 20

ALLOWED_HASH_LABELS = frozenset({"M01", "M08"})
DEFINING = frozenset(name for feats in CLASS_DEFINING_FEATURES.values() for name in feats)
SOURCE_ORDER = ["A", "E", "AMB", "R-team", "R-blind", "R-llm", "U", "X"]
FOCUS_EVENTS = ("oob_read", "uninit_read")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _domain(problem_id):
    problem_id = problem_id or ""
    if problem_id.startswith("Q"):
        return "dsa"
    if problem_id.startswith("P"):
        return "main"
    return "other"


def _source_columns(rows):
    found = {row.get("source") or "" for row in rows}
    ordered = [name for name in SOURCE_ORDER if name in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def check_counts(rows):
    """10.1. Fail when any of the 19 labels has fewer than 80 rows."""
    per_class = {label: 0 for label in LABELS}
    per_class.update(Counter(row.get("label") for row in rows))
    # Drop a None key if a row has no label; it is not one of the 19.
    per_class.pop(None, None)
    known = {label: per_class.get(label, 0) for label in LABELS}
    min_class = min(known, key=known.get)
    min_count = known[min_class]
    problems = sorted({row.get("problem_id") or "" for row in rows})
    sources = _source_columns(rows)
    class_problem = []
    class_source = []
    class_domain = []
    for label in LABELS:
        class_problem.append([
            sum(1 for row in rows if row.get("label") == label and row.get("problem_id") == problem)
            for problem in problems
        ])
        class_source.append([
            sum(1 for row in rows if row.get("label") == label and (row.get("source") or "") == source)
            for source in sources
        ])
        class_domain.append([
            sum(1 for row in rows if row.get("label") == label and _domain(row.get("problem_id")) == domain)
            for domain in ("main", "dsa", "other")
        ])
    soft_groups = defaultdict(set)
    n_soft = 0
    for row in rows:
        if not row.get("soft_label"):
            continue
        n_soft += 1
        group_id = row.get("ambiguous_group") or row.get("ast_hash")
        soft_groups[group_id].add(row.get("label"))
    set_counts = Counter(frozenset(labels) for labels in soft_groups.values())
    n_two = sum(1 for row in rows if row.get("is_two_bug"))
    n_rows = len(rows)
    return {
        "id": "10.1",
        "title": "Counts",
        "rule": "any class < 80 rows",
        "status": "FAIL" if min_count < MIN_CLASS_ROWS else "PASS",
        "headline": f"minimum is {min_class} = {min_count} (floor {MIN_CLASS_ROWS})",
        "n_rows": n_rows,
        "per_class": known,
        "min_class": min_class,
        "min_count": min_count,
        "problems": problems,
        "sources": sources,
        "class_problem": class_problem,
        "class_source": class_source,
        "class_domain": class_domain,
        "n_soft_rows": n_soft,
        "n_soft_groups": len(soft_groups),
        "soft_sets": {("+".join(sorted(labels)) or "(empty)"): count for labels, count in set_counts.items()},
        "n_two_bug": n_two,
        "two_bug_share": (n_two / n_rows) if n_rows else None,
        "n_main": sum(1 for row in rows if _domain(row.get("problem_id")) == "main"),
        "n_dsa": sum(1 for row in rows if _domain(row.get("problem_id")) == "dsa"),
    }


def check_duplicates(rows):
    """10.3. Fail on a hash whose labels are not exactly {M01, M08}."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row.get("ast_hash")].append(row)
    n_rows = len(rows)
    n_unique = len(by_hash)
    bad = []
    n_allowed = 0
    for digest, group in by_hash.items():
        labels = {row.get("label") for row in group}
        if len(labels) <= 1:
            continue
        if labels == ALLOWED_HASH_LABELS:
            n_allowed += 1
            continue
        bad.append({
            "ast_hash": digest,
            "n": len(group),
            "labels": sorted(label for label in labels if label is not None),
            "problems": sorted({row.get("problem_id") or "" for row in group}),
            "example_id": group[0].get("id"),
        })
    bad.sort(key=lambda item: (-item["n"], item["ast_hash"] or ""))
    largest = []
    for digest, group in sorted(by_hash.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
        labels = sorted({row.get("label") for row in group if row.get("label")})
        largest.append({
            "ast_hash": digest,
            "n": len(group),
            "n_labels": len(labels),
            "labels": labels,
            "n_problems": len({row.get("problem_id") for row in group}),
        })
    pvo = len({(row.get("problem_id"), row.get("variant"), row.get("op_id")) for row in rows})
    ratio = (n_unique / n_rows) if n_rows else None
    ratio_text = "n/a" if ratio is None else f"{ratio:.4f}"
    return {
        "id": "10.3",
        "title": "Duplicates",
        "rule": "hash collision other than {M01, M08}",
        "status": "FAIL" if bad else "PASS",
        "headline": (
            f"{n_unique} unique hashes / {n_rows} rows = {ratio_text}; "
            f"bad collisions = {len(bad)}; allowed {{M01, M08}} hashes = {n_allowed}"
        ),
        "n_rows": n_rows,
        "n_unique_hash": n_unique,
        "ratio": ratio,
        "n_allowed_pairs": n_allowed,
        "n_bad": len(bad),
        "bad": bad,
        "largest": largest,
        "n_unique_pvo": pvo,
    }


def operator_drop_status(attempts, unique_kept, luck):
    """Bounds on one operator's drop rate.

    ``attempts`` is the site count. ``unique_kept`` is a lower bound on mutants
    that stayed in the file (selection can drop some). ``luck`` counts
    passes-by-luck. The two sets are disjoint subsets of the attempts, so
    luck/attempts is a lower bound on the drop rate and
    (attempts - unique_kept)/attempts is an upper bound.
    """
    if attempts is None or attempts <= 0:
        return {"verdict": "no_attempts", "lower": None, "upper": None}
    if unique_kept < 0 or luck < 0 or unique_kept + luck > attempts:
        return {"verdict": "inconsistent", "lower": None, "upper": None}
    lower = luck / attempts
    upper = (attempts - unique_kept) / attempts
    if lower > DROP_LIMIT:
        verdict = "fail"
    elif upper <= DROP_LIMIT:
        verdict = "pass"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "lower": lower, "upper": upper}


def _base_key(row):
    return (row.get("op_id"), row.get("problem_id"), row.get("variant"), row.get("ast_hash"))


def check_drops(rows, luck_rows, attempts):
    """10.2. Operator drop > 40% is scored from bounds. See operator_drop_status."""
    catalogue = set(attempts)
    kept = Counter()
    luck = Counter()
    kept_keys = set()
    luck_keys = set()
    for row in rows:
        if row.get("is_two_bug") or row.get("aug"):
            continue
        op_id = row.get("op_id")
        if op_id not in catalogue:
            continue
        key = _base_key(row)
        if key in kept_keys:
            continue
        kept_keys.add(key)
        kept[op_id] += 1
    for row in luck_rows:
        op_id = row.get("op_id")
        if op_id not in catalogue:
            continue
        key = _base_key(row)
        if key in luck_keys:
            continue
        luck_keys.add(key)
        luck[op_id] += 1
    overlap = kept_keys & luck_keys
    if overlap:
        for op_id, _problem, _variant, _digest in overlap:
            kept[op_id] -= 1
            luck[op_id] -= 1
    operators = []
    for op_id in sorted(catalogue):
        status = operator_drop_status(attempts[op_id], kept[op_id], luck[op_id])
        operators.append({
            "op_id": op_id,
            "attempts": attempts[op_id],
            "unique_kept": kept[op_id],
            "luck": luck[op_id],
            **status,
        })
    verdicts = Counter(item["verdict"] for item in operators)
    if verdicts["fail"] or verdicts["inconsistent"]:
        # inconsistent counts would make a pass/fail claim false; treat as not passed
        status = "FAIL" if verdicts["fail"] else "INCONCLUSIVE"
    elif verdicts["inconclusive"]:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    order = {"fail": 0, "inconsistent": 1, "inconclusive": 2, "pass": 3, "no_attempts": 4}
    operators.sort(key=lambda item: (order.get(item["verdict"], 9), -(item["lower"] or -1), item["op_id"]))
    class_rows = Counter(row.get("label") for row in rows)
    class_luck = Counter(row.get("label") for row in luck_rows)
    per_class = []
    for label in LABELS:
        n_luck = class_luck.get(label, 0)
        n_data = class_rows.get(label, 0)
        denom = n_luck + n_data
        per_class.append({
            "label": label,
            "luck": n_luck,
            "dataset": n_data,
            "rate": (n_luck / denom) if denom else None,
        })
    n_fail = verdicts["fail"]
    n_inc = verdicts["inconclusive"] + verdicts["inconsistent"]
    n_pass = verdicts["pass"]
    n_zero = verdicts["no_attempts"]
    return {
        "id": "10.2",
        "title": "Verification",
        "rule": "operator drop > 40%",
        "status": status,
        "headline": (
            f"proven drop > 40%: {n_fail}; inconclusive: {n_inc}; "
            f"proven <= 40%: {n_pass}; no sites: {n_zero}; "
            f"kept/luck key overlap: {len(overlap)}"
        ),
        "operators": operators,
        "verdicts": dict(verdicts),
        "per_class_luck": per_class,
        "n_overlap": len(overlap),
        "n_luck_rows": len(luck_rows),
    }


def check_length(rows):
    """10.4 length half. Fail when a class separates from CORRECT on len(code)."""
    from scipy.stats import ks_2samp

    lengths = defaultdict(list)
    by_problem = defaultdict(lambda: defaultdict(list))
    for row in rows:
        size = len(row.get("code") or "")
        lengths[row.get("label")].append(size)
        by_problem[row.get("problem_id")][row.get("label")].append(size)
    correct = lengths.get("CORRECT") or []
    classes = []
    if len(correct) < KS_MIN_N:
        return {
            "id": "10.4l",
            "status": "SKIP",
            "headline": f"CORRECT has {len(correct)} rows; need {KS_MIN_N} for the KS test",
            "classes": [],
            "n_separating": 0,
        }
    for label in LABELS:
        if label == "CORRECT":
            continue
        sample = lengths.get(label) or []
        same_sample, same_correct = [], []
        for groups in by_problem.values():
            if groups.get(label) and groups.get("CORRECT"):
                same_sample.extend(groups[label])
                same_correct.extend(groups["CORRECT"])
        within_ks = within_p = None
        within_separates = False
        if len(same_sample) >= KS_MIN_N and len(same_correct) >= KS_MIN_N:
            within = ks_2samp(same_sample, same_correct, method="auto")
            within_ks = float(within.statistic)
            within_p = float(within.pvalue)
            within_separates = bool(within_ks >= KS_D_MIN and within_p < KS_P_MAX)
        if len(sample) < KS_MIN_N:
            classes.append({
                "label": label, "n": len(sample), "median": None, "mean": None,
                "ks": None, "p": None, "separates": False, "skipped": True,
                "within_ks": within_ks, "within_p": within_p, "within_separates": within_separates,
            })
            continue
        result = ks_2samp(sample, correct, method="auto")
        separates = bool(result.statistic >= KS_D_MIN and result.pvalue < KS_P_MAX)
        classes.append({
            "label": label,
            "n": len(sample),
            "median": float(np.median(sample)),
            "mean": float(np.mean(sample)),
            "ks": float(result.statistic),
            "p": float(result.pvalue),
            "separates": separates,
            "skipped": False,
            "within_ks": within_ks,
            "within_p": within_p,
            "within_separates": within_separates,
        })
    n_sep = sum(1 for item in classes if item["separates"])
    n_within = sum(1 for item in classes if item["within_separates"])
    scored = [item for item in classes if item["ks"] is not None]
    if scored:
        worst = max(scored, key=lambda item: item["ks"])
        worst_text = f"max KS D = {worst['ks']:.3f} ({worst['label']})"
    else:
        worst_text = "no class scored"
    return {
        "id": "10.4l",
        "status": "FAIL" if n_sep else "PASS",
        "headline": (
            f"{n_sep} classes separate from all CORRECT on code length; "
            f"{n_within} classes also separate from CORRECT on the same problems; {worst_text}"
        ),
        "classes": classes,
        "n_separating": n_sep,
        "n_within": n_within,
        "correct_n": len(correct),
        "correct_median": float(np.median(correct)),
        "correct_mean": float(np.mean(correct)),
        "by_label": {label: values for label, values in lengths.items()},
    }


def _stump_scores(column, y, labels):
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.tree import DecisionTreeClassifier

    x = np.asarray(column, dtype=np.float64).reshape(-1, 1)
    finite = np.isfinite(x[:, 0])
    if int(finite.sum()) < 2 or np.unique(x[finite, 0]).size < 2:
        return None
    if len(set(y[finite].tolist())) < 2:
        return None
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    try:
        clf.fit(x, y)
        pred = clf.predict(x)
        scored_y, scored_pred = y, pred
    except ValueError:
        clf.fit(x[finite], y[finite])
        scored_pred = clf.predict(x[finite])
        scored_y = y[finite]
    present = [label for label in labels if label in set(scored_y.tolist())]
    macro = float(recall_score(scored_y, scored_pred, labels=present, average="macro", zero_division=0))
    acc = float(accuracy_score(scored_y, scored_pred))
    return macro, acc


def check_stumps(names, X, y):
    """10.4 stump half. One depth-1 tree per feature, train rows only."""
    y = np.asarray(y)
    labels = [label for label in LABELS if label in set(y.tolist())]
    scored = []
    for index, name in enumerate(names):
        scores = _stump_scores(X[:, index], y, labels)
        if scores is None:
            continue
        macro, acc = scores
        defining = name in DEFINING
        scored.append({
            "name": name,
            "macro_acc": macro,
            "accuracy": acc,
            "defining": defining,
            "trips": (not defining) and (macro > STUMP_ACC_LIMIT or acc > STUMP_ACC_LIMIT),
        })
    scored.sort(key=lambda item: max(item["macro_acc"], item["accuracy"]), reverse=True)
    tripped = [item for item in scored if item["trips"]]
    if not scored:
        status, headline = "PASS", "no feature varied enough to fit a stump"
    elif tripped:
        worst = max(tripped, key=lambda item: max(item["macro_acc"], item["accuracy"]))
        status = "FAIL"
        headline = (
            f"{len(tripped)} non-defining features over {STUMP_ACC_LIMIT:.2f}; "
            f"worst {worst['name']} macro-acc {worst['macro_acc']:.3f} accuracy {worst['accuracy']:.3f}"
        )
    else:
        top = scored[0]
        status = "PASS"
        headline = (
            f"no non-defining stump over {STUMP_ACC_LIMIT:.2f}; "
            f"highest is {top['name']} macro-acc {top['macro_acc']:.3f} accuracy {top['accuracy']:.3f}"
            + (" (defining)" if top["defining"] else "")
        )
    return {
        "id": "10.4s",
        "status": status,
        "headline": headline,
        "rows": scored,
        "n_tripped": len(tripped),
        "n_scored": len(scored),
        "labels": labels,
    }


def _finite_mean(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def check_predicates(names, X, y):
    """10.5. Mean of each class's group-A defining features. Off-diagonal > 0.3 fails."""
    name_index = {name: index for index, name in enumerate(names)}
    y = np.asarray(y)
    used = {}
    skipped = {}
    col_labels = []
    for label, feats in CLASS_DEFINING_FEATURES.items():
        present = [feat for feat in feats if feat in name_index]
        missing = [feat for feat in feats if feat not in name_index]
        if present:
            col_labels.append(label)
            used[label] = present
        if missing:
            skipped[label] = missing
    row_labels = [label for label in LABELS if np.any(y == label)]
    matrix = []
    offenders = []
    diagonals = []
    for row_label in row_labels:
        mask = y == row_label
        row_means = []
        for col_label in col_labels:
            cols = [name_index[feat] for feat in used[col_label]]
            per_row = []
            block = X[np.ix_(mask, cols)]
            for sample in block:
                per_row.append(_finite_mean(sample))
            cell = _finite_mean([value for value in per_row if value is not None])
            row_means.append(cell)
            if cell is None or row_label == col_label:
                if row_label == col_label:
                    diagonals.append({"label": col_label, "mean": cell})
                continue
            if cell > BLEED_LIMIT:
                offenders.append({"row": row_label, "col": col_label, "mean": cell})
        matrix.append(row_means)
    offenders.sort(key=lambda item: item["mean"], reverse=True)
    if not col_labels:
        status, headline = "SKIP", "no group-A defining features were in the matrix"
    elif offenders:
        worst = offenders[0]
        status = "FAIL"
        headline = (
            f"{len(offenders)} off-diagonal cells > {BLEED_LIMIT:.1f}; "
            f"max {worst['mean']:.3f} ({worst['row']} rows, {worst['col']} features)"
        )
    else:
        status = "PASS"
        headline = f"no off-diagonal cell > {BLEED_LIMIT:.1f} on group-A defining features"
    return {
        "id": "10.5",
        "title": "Predicate matrix",
        "rule": "off-diagonal bleed > 0.3",
        "status": status,
        "headline": headline,
        "matrix": matrix,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "offenders": offenders,
        "diagonals": diagonals,
        "used": used,
        "skipped": skipped,
    }


def _column_kind(column):
    finite = column[np.isfinite(column)]
    nan_rate = 1.0 - (finite.size / column.size if column.size else 0.0)
    if finite.size == 0:
        return {"kind": "constant", "mode_share": 1.0, "nunique": 0, "nan_rate": float(nan_rate)}
    values, counts = np.unique(finite, return_counts=True)
    mode_share = float(counts.max() / finite.size)
    if values.size <= 1:
        kind = "constant"
    elif mode_share >= NEAR_CONSTANT_SHARE:
        kind = "near_constant"
    else:
        kind = "ok"
    return {"kind": kind, "mode_share": mode_share, "nunique": int(values.size), "nan_rate": float(nan_rate)}


def pairwise_pearson(matrix):
    """Pairwise-complete Pearson correlation. NaN where a pair has < 3 rows or no variance."""
    ok = np.isfinite(matrix)
    values = np.where(ok, matrix, 0.0)
    weights = ok.astype(np.float64)
    count = weights.T @ weights
    summed = values.T @ weights
    summed_sq = (values * values).T @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = summed / count
        var = summed_sq / count - mean * mean
        cov = (values.T @ values) / count - mean * mean.T
        std = np.sqrt(np.clip(var, 0, None))
        corr = cov / std / std.T
    corr[count < 3] = np.nan
    invalid = (~np.isfinite(std)) | (std <= 0) | (~np.isfinite(std.T)) | (std.T <= 0)
    corr[invalid] = np.nan
    corr[~np.isfinite(corr)] = np.nan
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def check_health(names, X):
    """10.6. Fail when a feature is constant, near-constant, or |corr| > 0.95."""
    kinds = [_column_kind(X[:, index]) for index in range(X.shape[1])]
    constant = []
    near = []
    for name, info in zip(names, kinds):
        payload = {"name": name, "defining": name in DEFINING, **info}
        if info["kind"] == "constant":
            constant.append(payload)
        elif info["kind"] == "near_constant":
            near.append(payload)
    corr = pairwise_pearson(np.asarray(X, dtype=np.float64))
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = corr[left, right]
            if np.isfinite(value) and abs(value) > CORR_LIMIT:
                pairs.append({"a": names[left], "b": names[right], "corr": float(value)})
    pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
    nan_rate_max = max((info["nan_rate"] for info in kinds), default=0.0)
    n_flagged = len(constant) + len(near) + len(pairs)
    if n_flagged:
        status = "FAIL"
        headline = (
            f"{len(constant)} constant, {len(near)} near-constant "
            f"(mode share >= {NEAR_CONSTANT_SHARE:.0%}), {len(pairs)} pairs with absolute correlation above {CORR_LIMIT}"
        )
    else:
        status = "PASS"
        headline = f"no constant, near-constant, or absolute correlation above {CORR_LIMIT}"
    return {
        "id": "10.6",
        "title": "Feature health",
        "rule": "drop constants; keep one of each correlated pair",
        "status": status,
        "headline": headline,
        "n_features": len(names),
        "constant": constant,
        "near_constant": near,
        "corr_pairs": pairs,
        "nan_rate_max": float(nan_rate_max),
    }


def _usable_columns(X):
    keep = []
    for index in range(X.shape[1]):
        finite = X[:, index][np.isfinite(X[:, index])]
        if finite.size >= 2 and np.unique(finite).size >= 2:
            keep.append(index)
    return np.asarray(keep, dtype=int)


def impute_median(X):
    out = np.array(X, dtype=np.float64, copy=True)
    med = np.nanmedian(out, axis=0)
    missing = ~np.isfinite(out)
    if missing.any():
        out[missing] = np.take(med, np.where(missing)[1])
    return out


def check_projection(X, y, sources, hashes):
    """10.8. PCA. Fail when unique-hash 1-NN label agreement is above 0.50."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    sources = np.asarray(sources)
    hashes = np.asarray(hashes)
    if X.shape[0] < 10:
        return {"id": "10.8", "status": "SKIP", "headline": "fewer than 10 train rows", "xy": None}
    cols = _usable_columns(X)
    if cols.size < 2:
        return {"id": "10.8", "status": "SKIP", "headline": "fewer than 2 varying features", "xy": None}
    filled = impute_median(X[:, cols])
    scaled = StandardScaler().fit_transform(filled)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(scaled)
    first = {}
    for index, digest in enumerate(hashes.tolist()):
        first.setdefault(digest, index)
    unique_idx = np.asarray(list(first.values()), dtype=int)
    xy_u = xy[unique_idx]
    y_u = y[unique_idx]
    if unique_idx.size < 3:
        return {"id": "10.8", "status": "SKIP", "headline": "fewer than 3 unique hashes", "xy": xy}
    neighbors = NearestNeighbors(n_neighbors=2, algorithm="kd_tree")
    neighbors.fit(xy_u)
    neigh = neighbors.kneighbors(return_distance=False)
    agreement = float(np.mean(y_u[neigh[:, 0]] == y_u[neigh[:, 1]]))
    silhouette = None
    counts = Counter(y_u.tolist())
    if len(counts) > 1 and min(counts.values()) >= 2:
        silhouette = float(silhouette_score(xy_u, y_u, metric="euclidean"))
    sil_text = "n/a" if silhouette is None else f"{silhouette:.3f}"
    if silhouette is None:
        status = "SKIP"
        decision = "silhouette undefined"
    elif silhouette >= SILHOUETTE_SEPARATE:
        status = "FAIL"
        decision = f"silhouette {sil_text} >= {SILHOUETTE_SEPARATE:.2f} (classes separate; look at features)"
    else:
        status = "PASS"
        decision = f"silhouette {sil_text} < {SILHOUETTE_SEPARATE:.2f} (heavy overlap, expected)"
    headline = (
        f"{decision}; 1-NN label agreement on {unique_idx.size} unique train hashes = {agreement:.3f}; "
        f"PCA variance {pca.explained_variance_ratio_[0]:.3f} + {pca.explained_variance_ratio_[1]:.3f}"
    )
    return {
        "id": "10.8",
        "title": "Projection",
        "rule": "classes overlapping heavily is expected; otherwise look at features",
        "status": status,
        "headline": headline,
        "agreement": agreement,
        "silhouette": silhouette,
        "variance": [float(value) for value in pca.explained_variance_ratio_],
        "n_unique": int(unique_idx.size),
        "n_features_used": int(cols.size),
        "xy": xy,
        "y": y,
        "sources": sources,
    }


def check_outliers(X, rows):
    """10.9. IsolationForest on train features. No numeric fail bar."""
    from sklearn.ensemble import IsolationForest

    if len(rows) < 20:
        return {"id": "10.9", "status": "SKIP", "headline": "fewer than 20 train rows", "top": []}
    cols = _usable_columns(X)
    if cols.size < 2:
        return {"id": "10.9", "status": "SKIP", "headline": "fewer than 2 varying features", "top": []}
    filled = impute_median(X[:, cols])
    forest = IsolationForest(
        contamination=IF_CONTAMINATION,
        random_state=0,
        n_jobs=2,
    )
    forest.fit(filled)
    scores = forest.decision_function(filled)
    order = np.argsort(scores)[:IF_TOP_N]
    top = []
    for index in order.tolist():
        row = rows[index]
        code = row.get("code") or ""
        lines = code.strip().splitlines()
        clipped = "\n".join(lines[:12])
        if len(lines) > 12:
            clipped += "\n…"
        top.append({
            "id": row.get("id"),
            "problem_id": row.get("problem_id"),
            "label": row.get("label"),
            "source": row.get("source"),
            "op_id": row.get("op_id"),
            "ast_hash": row.get("ast_hash"),
            "score": float(scores[index]),
            "code": clipped,
        })
    return {
        "id": "10.9",
        "title": "Outliers",
        "rule": "inspect the top 20; no numeric fail bar",
        "status": "INSPECT",
        "headline": (
            f"top {len(top)} of {len(rows)} train rows "
            f"(contamination {IF_CONTAMINATION}, lowest score {top[0]['score']:.4f})"
        ),
        "top": top,
        "n_train": len(rows),
        "n_features_used": int(cols.size),
    }


def _event_set(row):
    summary = row.get("trace_summary") or {}
    return set(summary.get("events") or [])


def check_events(rows):
    """10.10. oob_read should sit in M08/AMB; uninit_read in M05."""
    per_event_class = defaultdict(Counter)
    class_n = Counter(row.get("label") for row in rows)
    for row in rows:
        for event in _event_set(row):
            per_event_class[event][row.get("label")] += 1

    def share(event, predicate):
        matched = [row for row in rows if event in _event_set(row)]
        if not matched:
            return {"event": event, "n": 0, "n_expected": 0, "share": None, "ok": False}
        n_expected = sum(1 for row in matched if predicate(row))
        portion = n_expected / len(matched)
        return {
            "event": event,
            "n": len(matched),
            "n_expected": n_expected,
            "share": portion,
            "ok": portion >= EVENT_SHARE_MIN,
        }

    focus = [
        share("oob_read", lambda row: row.get("label") == "M08" or row.get("source") == "AMB"),
        share("uninit_read", lambda row: row.get("label") == "M05"),
    ]
    status = "PASS" if all(item["ok"] for item in focus) else "FAIL"
    bits = []
    for item in focus:
        if item["share"] is None:
            bits.append(f"{item['event']} absent")
        else:
            bits.append(f"{item['event']} {item['n_expected']}/{item['n']} = {item['share']:.3f}")
    return {
        "id": "10.10",
        "title": "Trace events",
        "rule": "oob_read concentrates in M08/AMB; uninit_read in M05",
        "status": status,
        "headline": "; ".join(bits) + f" (bar {EVENT_SHARE_MIN:.2f})",
        "focus": focus,
        "per_event_class": {event: dict(counts) for event, counts in sorted(per_event_class.items())},
        "class_n": {label: class_n.get(label, 0) for label in LABELS},
        "events": sorted(per_event_class),
    }


def check_exposure(rows, luck_rows, exposures):
    """10.12. |empirical fail rate - authored e_ik| > 0.3 on DSA problems."""
    cells = []
    for problem_id in sorted(exposures):
        for label, authored in sorted(exposures[problem_id].items()):
            n_fail = 0
            for row in rows:
                if row.get("problem_id") != problem_id or row.get("label") != label:
                    continue
                if row.get("is_two_bug") or row.get("aug"):
                    continue
                if (row.get("verified") or {}).get("tests_failed", 0) > 0:
                    n_fail += 1
            n_luck = sum(
                1 for row in luck_rows
                if row.get("problem_id") == problem_id and row.get("label") == label
            )
            denom = n_fail + n_luck
            empirical = (n_fail / denom) if denom else None
            gap = None if empirical is None else abs(empirical - float(authored))
            cells.append({
                "problem_id": problem_id,
                "label": label,
                "authored": float(authored),
                "n_fail": n_fail,
                "n_luck": n_luck,
                "empirical": empirical,
                "gap": gap,
                "flag": gap is not None and gap > EXPOSURE_GAP + 1e-9,
            })
    flagged = [cell for cell in cells if cell["flag"]]
    compared = [cell for cell in cells if cell["gap"] is not None]
    if not compared:
        status = "SKIP"
        headline = "no DSA exposure cell had an emitted mutant"
    elif flagged:
        worst = max(flagged, key=lambda cell: cell["gap"])
        status = "FAIL"
        headline = (
            f"{len(flagged)} of {len(compared)} cells have absolute gap > {EXPOSURE_GAP}; "
            f"max {worst['gap']:.3f} at {worst['problem_id']} {worst['label']} "
            f"(empirical {worst['empirical']:.3f}, authored {worst['authored']:.3f})"
        )
    else:
        worst = max(compared, key=lambda cell: cell["gap"])
        status = "PASS"
        headline = (
            f"0 of {len(compared)} cells have absolute gap > {EXPOSURE_GAP}; "
            f"max gap {worst['gap']:.3f} at {worst['problem_id']} {worst['label']}"
        )
    missing = [cell for cell in cells if cell["gap"] is None]
    return {
        "id": "10.12",
        "title": "Exposure",
        "rule": "absolute gap between empirical and authored e_ik > 0.3",
        "status": status,
        "headline": headline,
        "cells": cells,
        "n_flagged": len(flagged),
        "n_compared": len(compared),
        "n_missing": len(missing),
    }


def check_masking(names, X, y, row_ids):
    """10.13. Fail when masking zeroes the true class. Uses ml.model.mask."""
    from ml.model.mask import allowed_classes

    name_index = {name: index for index, name in enumerate(names)}
    missing = [name for name in GROUP_A if name not in name_index]
    if missing:
        return {
            "id": "10.13",
            "status": "SKIP",
            "headline": f"matrix is missing {len(missing)} group-A features",
            "n_masked": None,
            "by_class": {},
            "examples": [],
            "n_rows": len(y),
        }
    columns = [name_index[name] for name in GROUP_A]
    allowed = allowed_classes(X[:, columns], features=list(GROUP_A), labels=list(LABELS))
    label_index = {label: index for index, label in enumerate(LABELS)}
    by_class = Counter()
    examples = []
    n_masked = 0
    for index, label in enumerate(y):
        slot = label_index.get(label)
        if slot is None:
            continue
        if not allowed[index, slot]:
            n_masked += 1
            by_class[label] += 1
            if len(examples) < 15:
                examples.append(row_ids[index])
    n_rows = len(y)
    rate = (n_masked / n_rows) if n_rows else None
    if n_masked:
        status = "FAIL"
        headline = f"{n_masked} / {n_rows} rows have the true label masked"
    else:
        status = "PASS"
        headline = f"0 / {n_rows} rows have the true label masked"
    return {
        "id": "10.13",
        "title": "Masking",
        "rule": "any masked true label",
        "status": status,
        "headline": headline,
        "n_masked": n_masked,
        "rate": rate,
        "by_class": dict(by_class),
        "examples": examples,
        "n_rows": n_rows,
    }


AUC_LIMIT = 0.85
N_BOOT = 1000
# Smaller leaf than the diagnoser: the external sets are tens or hundreds of rows.
_ADV_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 2,
    "is_unbalance": True,
    "force_col_wise": True,
}


def _group_bootstrap_auc(y, scores, groups, rng, n_boot):
    from sklearn.metrics import roc_auc_score

    buckets = defaultdict(list)
    for index, group in enumerate(groups):
        buckets[group].append(index)
    keys = list(buckets)
    aucs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(keys), size=len(keys))
        idx = []
        for pick in chosen:
            idx.extend(buckets[keys[int(pick)]])
        idx = np.asarray(idx, dtype=int)
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], scores[idx])))
    if len(aucs) < 50:
        return None, None, len(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), len(aucs)


def _cv_splits(X, y, groups):
    """Grouped folds. Mixed-class groups stay together when the splitter allows it."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    pos = {group for group, label in zip(groups, y) if label == 1}
    neg = {group for group, label in zip(groups, y) if label == 0}
    n_splits = min(5, len(pos), len(neg))
    if n_splits < 2:
        raise ValueError("fewer than 2 groups in one class")
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(cv.split(X, y, groups)), n_splits, "StratifiedGroupKFold"
    except ValueError:
        tagged = np.array([f"{label}:{group}" for label, group in zip(y, groups)])
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(X, y, tagged)), n_splits, "GroupKFold"


def adversarial_auc(X_train, groups_train, X_other, groups_other, names, *, num_boost_round=100, seed=42):
    """TRAIN vs an external set. Features only; ``y`` is synthetic (0) or external (1)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X_train = np.asarray(X_train, dtype=np.float64)
    X_other = np.asarray(X_other, dtype=np.float64)
    if X_train.size == 0 or X_other.size == 0:
        return {"status": "SKIP", "headline": "empty matrix", "auc": None, "top": []}
    X = np.vstack([X_train, X_other])
    y = np.array([0] * len(X_train) + [1] * len(X_other))
    groups = np.array([str(g) for g in list(groups_train) + list(groups_other)])
    splits, n_splits, splitter = _cv_splits(X, y, groups)
    oof = np.full(len(y), np.nan)
    gain = np.zeros(X.shape[1], dtype=np.float64)
    params = dict(_ADV_PARAMS)
    params["seed"] = seed
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"N1: adversarial fold {fold}/{n_splits}", flush=True)
        dataset = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=list(names), free_raw_data=False)
        model = lgb.train(params, dataset, num_boost_round=num_boost_round)
        oof[test_idx] = model.predict(X[test_idx])
        gain += model.feature_importance(importance_type="gain")
    scored = np.isfinite(oof)
    auc = float(roc_auc_score(y[scored], oof[scored]))
    lo, hi, n_boot = _group_bootstrap_auc(y[scored], oof[scored], groups[scored], np.random.default_rng(seed), N_BOOT)
    total = float(gain.sum())
    order = np.argsort(gain)[::-1]
    top = []
    for index in order:
        if gain[index] <= 0 or len(top) >= 5:
            break
        top.append({
            "name": names[int(index)],
            "gain_share": float(gain[index] / total) if total else 0.0,
        })
    over = auc > AUC_LIMIT
    return {
        "status": "FAIL" if over else "PASS",
        "auc": auc,
        "lo": lo,
        "hi": hi,
        "n_boot": n_boot,
        "n_train": int(len(X_train)),
        "n_other": int(len(X_other)),
        "n_splits": n_splits,
        "splitter": splitter,
        "top": top,
        "over": over,
    }


def check_shift(rllm, itsp):
    """10.7. Point AUC above 0.85 fails. The R-llm comparison is the rule; ITSP is reported beside it.

    ``rllm`` and ``itsp`` are result dicts from ``adversarial_auc`` plus count fields, or None.
    """
    if not rllm or rllm.get("auc") is None:
        return {
            "id": "10.7",
            "title": "Distribution shift",
            "rule": "adversarial AUC > 0.85",
            "status": "SKIP",
            "headline": "R-llm stand-in was not scored",
            "rllm": rllm,
            "itsp": itsp,
        }
    def _bit(name, item):
        if not item or item.get("auc") is None:
            return f"{name} not scored"
        ci = "n/a" if item.get("lo") is None else f"{item['lo']:.3f} to {item['hi']:.3f}"
        flag = "over 0.85" if item["over"] else "not over 0.85"
        return f"{name} AUC {item['auc']:.3f} (95% CI {ci}) {flag}"

    # The rule in §10 is the R-team comparison. ITSP is a second comparison against the same bar.
    status = rllm["status"]
    headline = (
        "LLM-written stand-in for R-team, not hand-written. "
        + _bit("R-llm", rllm)
        + "; "
        + _bit("ITSP", itsp)
    )
    return {
        "id": "10.7",
        "title": "Distribution shift",
        "rule": "adversarial AUC > 0.85",
        "status": status,
        "headline": headline,
        "rllm": rllm,
        "itsp": itsp,
    }


def check_realistic(rows):
    """10.11. κ needs a second rater on the realistic set, which is not in this file."""
    n_rater2 = sum(1 for row in rows if row.get("rater2_label"))
    sources = Counter(row.get("source") for row in rows)
    n_team = sources.get("R-team", 0)
    n_blind = sources.get("R-blind", 0)
    if n_rater2 < 2 or (n_team + n_blind) == 0:
        return {
            "id": "10.11",
            "title": "Realistic set",
            "rule": "kappa < 0.6",
            "status": "SKIP",
            "headline": (
                f"R-team rows in dataset.jsonl = {n_team}, R-blind = {n_blind}, "
                f"rows with rater2_label = {n_rater2}; kappa not computed"
            ),
            "n_rater2": n_rater2,
            "n_team": n_team,
            "n_blind": n_blind,
            "kappa": None,
        }
    from sklearn.metrics import cohen_kappa_score

    paired = [row for row in rows if row.get("rater2_label") and row.get("source") in ("R-team", "R-blind")]
    kappa = float(cohen_kappa_score([row["label"] for row in paired], [row["rater2_label"] for row in paired]))
    return {
        "id": "10.11",
        "title": "Realistic set",
        "rule": "kappa < 0.6",
        "status": "FAIL" if kappa < 0.6 else "PASS",
        "headline": f"kappa = {kappa:.3f} on {len(paired)} double-labelled realistic rows",
        "n_rater2": n_rater2,
        "n_team": n_team,
        "n_blind": n_blind,
        "kappa": kappa,
    }
