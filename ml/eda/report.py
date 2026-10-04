"""Markdown for docs/eda.md. Plots are linked from ml/eda/figures/."""
from __future__ import annotations

from ml.eda.checks import (
    BLEED_LIMIT,
    CORR_LIMIT,
    DROP_LIMIT,
    EVENT_SHARE_MIN,
    EXPOSURE_GAP,
    KS_D_MIN,
    KS_MIN_N,
    KS_P_MAX,
    MIN_CLASS_ROWS,
    NEAR_CONSTANT_SHARE,
    SILHOUETTE_SEPARATE,
    STUMP_ACC_LIMIT,
)


def _cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4g}"
    return str(value).replace("|", "/")


def _pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _p(value):
    if value is None:
        return "n/a"
    if value < 1e-6:
        return f"{value:.2e}"
    return f"{value:.4g}"


def scoring_notes():
    return f"""## How a rule is scored

Cutoffs written in 03 §10 are used as stated. Where §10 names the check and not the number, the bar used in this file is below. The same text is in `notes/N1.md`.

- **10.1** Fail if any of the 19 labels has fewer than {MIN_CLASS_ROWS} rows.
- **10.2** Attempt count is the number of operator sites on allowed correct variants (the generator's attempt loop, without re-running the interpreter). The file is after selection, so unique unaugmented base rows are a lower bound on mutants that passed verification. Lower bound on drop rate = unique passes-by-luck / attempts. Upper bound = (attempts − unique base rows) / attempts. Fail when the lower bound is greater than {DROP_LIMIT:.2f}. Pass when the upper bound is at most {DROP_LIMIT:.2f}. Otherwise the operator is inconclusive. Luck and kept keys that are the same mutant are removed from both counts. Counts that sum past the attempt total are inconsistent.
- **10.3** Fail when an `ast_hash` has more than one label and the label set is not exactly {{M01, M08}}.
- **10.4** Stumps are depth-1 trees, one feature at a time, on `split=train`. Macro-acc is the unweighted mean recall over classes present in that split. A non-defining feature trips the rule when macro-acc or overall accuracy is greater than {STUMP_ACC_LIMIT:.2f}. Defining means the name is in `CLASS_DEFINING_FEATURES` for some class. Length uses `len(code)`. A class separates from CORRECT when both sides have at least {KS_MIN_N} rows, the two-sample KS statistic is at least {KS_D_MIN:.2f}, and p < {KS_P_MAX:.2f}. §10 does not name the KS cutoff. The pass/fail uses every CORRECT row. A second KS, against CORRECT rows on the same problems only, is reported and does not change the status.
- **10.5** A cell is the mean, over rows of the row-class, of the per-row mean of the column-class's defining features that sit in group A. Group B and R defining features are omitted. Fail when any off-diagonal cell is greater than {BLEED_LIMIT:.1f}.
- **10.6** Train rows, groups A and C. Constant means at most one finite value. Near-constant means the most common finite value covers at least {NEAR_CONSTANT_SHARE:.0%} of finite rows. Correlation is pairwise-complete Pearson. Fail when any constant, near-constant, or pair with absolute correlation above {CORR_LIMIT} is present.
- **10.7** Adversarial validation is a binary LightGBM (grouped CV, 2 threads) on groups A and C only. The target is synthetic train versus the external file. Class labels of the external rows are not read. Fail when the out-of-fold AUC is greater than 0.85. The interval is a 95% group bootstrap of that AUC (1,000 resamples of `problem_id`). The R-llm file is an LLM-written stand-in for R-team, not hand-written. R-blind is not read. ITSP is a second comparison against the same bar and does not replace the R-llm verdict.
- **10.8** PCA on median-imputed, standardised train features after constant columns are dropped. Fail (look at features) when the silhouette of one row per `ast_hash` is at least {SILHOUETTE_SEPARATE:.2f}. Below that is the heavy overlap §10 expects. 1-NN label agreement on those rows is reported and is not the pass/fail number: copies of one template sit next to each other even when the class clouds overlap. UMAP is not installed.
- **10.9** IsolationForest, contamination 0.02, `random_state=0`, on the same train matrix. No numeric fail bar. Status is INSPECT.
- **10.10** A row counts once per event name in `trace_summary.events`. Fail unless at least {EVENT_SHARE_MIN:.0%} of `oob_read` rows are label M08 or source AMB, at least {EVENT_SHARE_MIN:.0%} of `uninit_read` rows are label M05, and both events occur. §10 says "concentrate" without a fraction.
- **10.11** Cohen's kappa needs a second rater on R-team or R-blind. Those rows are not in this file.
- **10.12** For each DSA problem and each authored exposure class, empirical = unaugmented non-two-bug rows in `dataset.jsonl` with that label and `tests_failed > 0`, divided by those plus passes-by-luck rows with that label. Fail when the absolute gap is greater than {EXPOSURE_GAP}. A class with no emitted mutant is listed and not compared.
- **10.13** Fail when `ml.model.mask.allowed_classes` on group A zeroes the row's true label. Domain counts have no numeric bar.
"""


def body_counts(result):
    rows = [[label, result["per_class"][label], "yes" if result["per_class"][label] < MIN_CLASS_ROWS else "no"]
            for label in result["per_class"]]
    parts = [
        _table(["class", "rows", f"under {MIN_CLASS_ROWS}"], rows),
        "",
        f"Two-bug rows: {result['n_two_bug']} ({_pct(result['two_bug_share'])}). "
        f"Soft-label rows: {result['n_soft_rows']} in {result['n_soft_groups']} groups.",
        "",
    ]
    if result["soft_sets"]:
        parts.append(_table(
            ["label set", "groups"],
            [[name, count] for name, count in sorted(result["soft_sets"].items())],
        ))
        parts.append("")
    parts.append("Class × source:")
    parts.append("")
    headers = ["class", *result["sources"]]
    table_rows = []
    for index, label in enumerate(result["per_class"]):
        table_rows.append([label, *result["class_source"][index]])
    parts.append(_table(headers, table_rows))
    parts.append("")
    parts.append("Non-zero class × problem cells:")
    parts.append("")
    nonzero = []
    for index, label in enumerate(result["per_class"]):
        bits = []
        for problem, count in zip(result["problems"], result["class_problem"][index]):
            if count:
                bits.append(f"{problem}:{count}")
        nonzero.append([label, " ".join(bits)])
    parts.append(_table(["class", "problem:rows"], nonzero))
    return "\n".join(parts)


def body_drops(result):
    parts = [
        "Passes-by-luck rate per class is luck rows / (luck rows + dataset rows). "
        "That rate is not the operator drop rate.",
        "",
        _table(
            ["class", "luck", "dataset", "luck rate"],
            [[item["label"], item["luck"], item["dataset"], _pct(item["rate"])] for item in result["per_class_luck"]],
        ),
        "",
        "Operator bounds. `fail` means the lower bound is above 40%. "
        "`pass` means the upper bound is at most 40%.",
        "",
        _table(
            ["op_id", "attempts", "unique kept", "luck", "lower", "upper", "verdict"],
            [[
                item["op_id"], item["attempts"], item["unique_kept"], item["luck"],
                _pct(item["lower"]), _pct(item["upper"]), item["verdict"],
            ] for item in result["operators"]],
        ),
    ]
    return "\n".join(parts)


def body_duplicates(result):
    parts = [
        f"`n_unique(problem, variant, op_id)` = {result['n_unique_pvo']}.",
        "",
        "Largest hash groups:",
        "",
        _table(
            ["ast_hash", "rows", "labels", "n problems"],
            [[item["ast_hash"], item["n"], ",".join(item["labels"]), item["n_problems"]] for item in result["largest"]],
        ),
    ]
    if result["bad"]:
        parts.extend(["", "Cross-label collisions:", ""])
        parts.append(_table(
            ["ast_hash", "rows", "labels", "problems", "example"],
            [[
                item["ast_hash"], item["n"], ",".join(item["labels"]),
                ",".join(item["problems"]), item["example_id"],
            ] for item in result["bad"][:40]],
        ))
        if len(result["bad"]) > 40:
            parts.append("")
            parts.append(f"{len(result['bad']) - 40} further collisions are omitted from the table; the count above includes them.")
    else:
        parts.extend(["", "No cross-label hash collision outside {M01, M08}."])
    return "\n".join(parts)


def body_length(result):
    if result["status"] == "SKIP":
        return result["headline"]
    rows = []
    for item in result["classes"]:
        rows.append([
            item["label"], item["n"],
            None if item["median"] is None else round(item["median"], 1),
            None if item["ks"] is None else round(item["ks"], 3),
            _p(item["p"]),
            "yes" if item["separates"] else "no",
            None if item.get("within_ks") is None else round(item["within_ks"], 3),
            "yes" if item.get("within_separates") else "no",
        ])
    return "\n".join([
        f"CORRECT: n={result['correct_n']}, median length {result['correct_median']:.1f}, "
        f"mean {result['correct_mean']:.1f}.",
        "",
        "The status uses the KS against every CORRECT row. "
        "`same-problem D` is the same test restricted to problems that contain the class.",
        "",
        _table(
            ["class", "n", "median len", "KS D", "p", "separates", "same-problem D", "same-problem separates"],
            rows,
        ),
    ])


def body_stumps(result, limit=25):
    if not result["rows"]:
        return "No stump was fit."
    shown = result["rows"][:limit]
    tripped = [item for item in result["rows"] if item["trips"]]
    lines = [
        f"Scored {result['n_scored']} features on classes {', '.join(result['labels'])}. "
        f"Showing the top {len(shown)} by max(macro-acc, accuracy).",
        "",
        _table(
            ["feature", "macro-acc", "accuracy", "defining", "over 0.60"],
            [[
                item["name"], round(item["macro_acc"], 3), round(item["accuracy"], 3),
                "yes" if item["defining"] else "no",
                "yes" if item["trips"] else "no",
            ] for item in shown],
        ),
    ]
    extra = [item for item in tripped if item not in shown]
    if extra:
        lines.extend(["", "Further non-defining features over 0.60:", ""])
        lines.append(_table(
            ["feature", "macro-acc", "accuracy"],
            [[item["name"], round(item["macro_acc"], 3), round(item["accuracy"], 3)] for item in extra],
        ))
    return "\n".join(lines)


def body_predicates(result):
    if result["status"] == "SKIP":
        return result["headline"]
    parts = ["Diagonal (mean of the class's own group-A defining features):", ""]
    parts.append(_table(
        ["class", "diagonal mean"],
        [[item["label"], None if item["mean"] is None else round(item["mean"], 3)] for item in result["diagonals"]],
    ))
    parts.extend(["", "Group-A features used / trace features not in this matrix:", ""])
    used_rows = []
    labels = sorted(set(result["used"]) | set(result["skipped"]))
    for label in labels:
        used_rows.append([
            label,
            ", ".join(result["used"].get(label, [])) or "—",
            ", ".join(result["skipped"].get(label, [])) or "—",
        ])
    parts.append(_table(["class", "used", "not in matrix"], used_rows))
    if result["offenders"]:
        parts.extend(["", f"Off-diagonal cells above {BLEED_LIMIT:.1f}:", ""])
        parts.append(_table(
            ["row class", "feature class", "mean"],
            [[item["row"], item["col"], round(item["mean"], 3)] for item in result["offenders"]],
        ))
    else:
        parts.extend(["", "No off-diagonal cell is above 0.3."])
    return "\n".join(parts)


def body_health(result, parse_fail, n_rows):
    parts = [
        f"Features in the matrix: {result['n_features']}. "
        f"Parse failures: {parse_fail} / {n_rows}. Max NaN rate: {_pct(result['nan_rate_max'])}.",
        "",
    ]
    if result["constant"]:
        names = ", ".join(
            f"{item['name']}" + (" (defining)" if item["defining"] else "")
            for item in result["constant"]
        )
        parts.append(f"Constant ({len(result['constant'])}): {names}")
        parts.append("")
    else:
        parts.append("No constant features.")
        parts.append("")
    if result["near_constant"]:
        parts.append(_table(
            ["feature", "mode share", "n unique", "defining"],
            [[
                item["name"], f"{item['mode_share']:.4f}", item["nunique"],
                "yes" if item["defining"] else "no",
            ] for item in result["near_constant"]],
        ))
        parts.append("")
    else:
        parts.append("No near-constant features.")
        parts.append("")
    if result["corr_pairs"]:
        parts.append(_table(
            ["feature", "feature", "corr"],
            [[item["a"], item["b"], round(item["corr"], 3)] for item in result["corr_pairs"]],
        ))
    else:
        parts.append("No pair with absolute correlation above 0.95.")
    return "\n".join(parts)


def _shift_table(item):
    if not item or item.get("auc") is None:
        return item.get("headline", "Not scored.") if item else "Not scored."
    ci = "n/a" if item.get("lo") is None else f"{item['lo']:.3f} to {item['hi']:.3f}"
    lines = [
        _table(
            ["n train", "n external", "AUC", "95% CI", "splits", "over 0.85"],
            [[
                item["n_train"], item["n_other"], round(item["auc"], 3), ci,
                f"{item['n_splits']} {item['splitter']}",
                "yes" if item["over"] else "no",
            ]],
        ),
        "",
        "Top features by total split gain:",
        "",
    ]
    if item.get("top"):
        lines.append(_table(
            ["feature", "share of gain"],
            [[row["name"], _pct(row["gain_share"])] for row in item["top"]],
        ))
    else:
        lines.append("The model put no gain on any feature.")
    return "\n".join(lines)


def body_shift(result):
    rllm = result.get("rllm") or {}
    itsp = result.get("itsp") or {}
    parts = [
        "LLM-written stand-in for R-team, not hand-written. "
        "Inputs are groups A and C. Labels on the external rows are dropped before the model is fit. "
        "R-blind is not read.",
        "",
        "### R-llm",
        "",
        _shift_table(rllm),
    ]
    counts = rllm.get("counts") or {}
    if counts:
        parts.extend([
            "",
            f"Rows: {counts.get('n')}. Problems: {counts.get('n_problems')}. "
            f"AST parse failures: {counts.get('parse_fail')}. "
            f"Rows whose problem is in the bank: {counts.get('n_bank')}.",
        ])
    parts.extend(["", "### ITSP", ""])
    if itsp.get("auc") is None and not itsp.get("counts"):
        parts.append(itsp.get("headline") or "ITSP slice was not found.")
    else:
        parts.append(_shift_table(itsp))
        counts = itsp.get("counts") or {}
        if counts:
            parts.extend([
                "",
                f"Path: `{counts.get('path')}`. Rows: {counts.get('n')}. "
                f"Problems: {counts.get('n_problems')}. "
                f"AST parse failures: {counts.get('parse_fail')}. "
                f"Signatures read from the code (not in the problem bank): {counts.get('n_signature')}.",
                "",
            ])
            families = counts.get("families") or []
            if families:
                parts.append(_table(["family", "rows"], families))
    return "\n".join(parts)


def body_projection(result):
    if result.get("xy") is None and result["status"] == "SKIP":
        return result["headline"]
    sil = result.get("silhouette")
    var = result.get("variance") or []
    var_text = " + ".join(f"{value:.3f}" for value in var) if var else "n/a"
    return (
        f"Features used after dropping constants: {result.get('n_features_used')}. "
        f"Unique train hashes: {result.get('n_unique')}. "
        f"Explained variance: {var_text}. "
        f"Silhouette on the unique-hash plane: {'n/a' if sil is None else f'{sil:.3f}'} "
        f"(reported, not the pass/fail number)."
    )


def body_outliers(result):
    if not result.get("top"):
        return result["headline"]
    parts = [
        f"Train rows: {result['n_train']}. Varying features: {result['n_features_used']}. "
        "Score is `decision_function` (lower is more anomalous). Code is clipped to 12 lines.",
        "",
    ]
    for rank, item in enumerate(result["top"], start=1):
        parts.append(
            f"### {rank}. `{item['id']}` score {item['score']:.4f} "
            f"{item['problem_id']} {item['label']} op `{item['op_id']}` source {item['source']}"
        )
        parts.append("")
        parts.append("```c")
        parts.append(item["code"])
        parts.append("```")
        parts.append("")
    return "\n".join(parts)


def body_events(result):
    parts = []
    for item in result["focus"]:
        parts.append(
            f"- `{item['event']}`: {item['n_expected']} / {item['n']} "
            f"in the expected bucket ({_pct(item['share'])})"
        )
    parts.append("")
    headers = ["class", "rows", *result["events"]]
    table_rows = []
    for label in result["class_n"]:
        table_rows.append([
            label,
            result["class_n"][label],
            *[result["per_event_class"].get(event, {}).get(label, 0) for event in result["events"]],
        ])
    parts.append(_table(headers, table_rows))
    return "\n".join(parts)


def body_exposure(result):
    parts = [
        "Empirical rate uses unaugmented, non-two-bug failures in `dataset.jsonl` "
        "over those plus `passes_by_luck.jsonl`. "
        f"Cells with no mutant: {result['n_missing']}.",
        "",
        _table(
            ["problem", "class", "authored", "n fail", "n luck", "empirical", "|gap|", "over 0.3"],
            [[
                cell["problem_id"], cell["label"], cell["authored"], cell["n_fail"], cell["n_luck"],
                None if cell["empirical"] is None else round(cell["empirical"], 3),
                None if cell["gap"] is None else round(cell["gap"], 3),
                "yes" if cell["flag"] else "no",
            ] for cell in result["cells"]],
        ),
    ]
    return "\n".join(parts)


def body_domain(counts, masking):
    parts = [
        f"Main rows: {counts['n_main']}. DSA rows: {counts['n_dsa']}.",
        "",
        _table(
            ["class", "main", "dsa", "other"],
            [[label, *counts["class_domain"][index]] for index, label in enumerate(counts["per_class"])],
        ),
        "",
        f"Masked true labels: {masking['n_masked']} / {masking['n_rows']} ({_pct(masking.get('rate'))}).",
    ]
    if masking.get("by_class"):
        parts.extend(["", _table(
            ["class", "masked rows"],
            [[label, count] for label, count in sorted(masking["by_class"].items())],
        )])
    if masking.get("examples"):
        parts.extend(["", "Example rows:", ""])
        parts.append(_table(
            ["id", "label", "problem"],
            [[item.get("id"), item.get("label"), item.get("problem_id")] for item in masking["examples"]],
        ))
    return "\n".join(parts)


def render(scope, sections):
    lines = [
        "# EDA",
        "",
        "Generated by `python -m ml.eda` from `ml/data/dataset.jsonl` (03 §10).",
        "",
        scope,
        "",
        scoring_notes().rstrip(),
        "",
        "## Red rules",
        "",
        _table(
            ["#", "Rule", "Status", "Number"],
            [[section["id"], section["rule"], section["status"], section["headline"]] for section in sections],
        ),
        "",
    ]
    for section in sections:
        lines.append(f"## {section['id']} {section['title']}")
        lines.append("")
        lines.append(f"**Rule.** {section['rule']}")
        lines.append("")
        lines.append(f"**{section['status']}.** {section['headline']}")
        lines.append("")
        body = section.get("body") or ""
        if body:
            lines.append(body.rstrip())
            lines.append("")
        for filename, caption in section.get("plots") or []:
            lines.append(f"![{caption}](../ml/eda/figures/{filename})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
