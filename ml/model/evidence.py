"""Evidence generation (plans/03 §5.6), package M1.

1. `booster.predict(x, pred_contrib=True)` gives SHAP-style contributions per class; the slice
   for the top class is taken.
2. The top positive contributors that have a sentence for their current value become items,
   filled from the metadata `ml.features.extract.extract` returned. A contributor with no
   sentence (or a missing placeholder) is skipped and the next one is tried.
3. At most one RUN item summarising the run is added, then the fixer sentence when a fix
   description is given, then the Bayes layer's items (YOU PREDICTED / PROBE / HISTORY).
4. For a HARD twin set with status "ambiguous" the fixed "identical code" item is added.

`weight` is the feature's share of all positive contributions toward the class (0–1).
"""
from __future__ import annotations

import numpy as np

from ml.contracts.classes import LABELS, TWIN_SETS
from ml.contracts.feature_names import FEATURES
from ml.features.extract import load_templates, render_evidence

TOP_K = 3
HARD_TWIN_TEXT = "This code is identical for both explanations. Asking one question."
RUN_SUMMARY_FEATURES = ["b_status_timeout", "b_status_runtime_error", "b_pass_frac"]


def class_contributions(contributions, cls, labels=LABELS):
    """Contribution of each feature toward `cls` (bias column dropped)."""
    return np.asarray(contributions)[labels.index(cls), :-1]


def top_contributors(contrib, features=FEATURES):
    """(feature, contribution, share) for every positive contributor, largest first."""
    contrib = np.asarray(contrib, dtype=np.float64)
    positive = contrib > 0
    total = contrib[positive].sum()
    order = [i for i in np.argsort(-contrib, kind="stable") if positive[i]]
    return [(features[i], float(contrib[i]), float(contrib[i] / total)) for i in order]


def model_items(contributions, cls, meta, *, k=TOP_K, labels=LABELS, features=FEATURES, templates=None):
    """Up to `k` CODE/RUN items for the strongest contributors toward `cls`."""
    items = []
    for name, _, share in top_contributors(class_contributions(contributions, cls, labels), features):
        item = render_evidence(name, meta, templates)
        if item is None:
            continue
        items.append(dict(item, weight=round(share, 3)))
        if len(items) >= k:
            break
    return items


def run_item(meta, already=(), templates=None):
    """One sentence about how the run went, unless the model items already say it."""
    for name in RUN_SUMMARY_FEATURES:
        if name in already:
            return None
        item = render_evidence(name, meta, templates)
        if item is not None:
            return {"type": "RUN", "text": item["text"], "feature": name}
    return None


def fixer_item(fix_desc, line=None):
    """The post-model fixer sentence (03 §4.4): not a model feature."""
    text = load_templates()["f_fix"]["text"].format(fix_desc=fix_desc)
    item = {"type": "CODE", "text": text, "feature": "f_fix"}
    if line is not None:
        item["line"] = line
    return item


def build_evidence(contributions, cls, meta, *, status=None, twin_set=None, fix_desc=None, fix_line=None,
                   extra_items=(), k=TOP_K, labels=LABELS, features=FEATURES):
    """The `evidence` list of a diagnosis.

    contributions  (n_labels, n_features + 1) from `Diagnoser.contributions(row)`
    cls            the class to explain (top-1)
    meta           the metadata `extract` returned for the same row
    extra_items    items the Bayes layer produced, appended unchanged
    """
    items = model_items(contributions, cls, meta, k=k, labels=labels, features=features)
    if meta.get("has_trace"):
        summary = run_item(meta, already={item["feature"] for item in items})
        if summary is not None:
            items.append(summary)
    if fix_desc:
        items.append(fixer_item(fix_desc, fix_line))
    items += [dict(item) for item in extra_items]
    if status == "ambiguous" and twin_set and TWIN_SETS.get(twin_set, {}).get("type") == "HARD":
        items.append({"type": "RUN", "text": HARD_TWIN_TEXT})
    return [{key: value for key, value in item.items() if value is not None} for item in items]
