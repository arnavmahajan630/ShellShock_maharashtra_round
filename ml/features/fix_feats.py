"""Group F fix-probe features (ml_plan/03 §4.4). Package R1.

``fix_features(problem, code, classes)`` runs the verified fixer for each class
named in ``classes``. That list is the top-3 after the model. It is not all 17
classes, and these values are not columns of the training matrix.

For each class that is run:

* ``f_fix_<class>`` is 1 when any of its candidates (at most five) passes every
  test, and 0 when none does.
* ``f_fixgain_<class>`` is the best change in pass fraction among the candidates
  that were actually run. The search stops at the first candidate that passes,
  and a full pass is the largest gain those candidates can have.

Classes that are not in ``classes`` are absent from the returned dict. They are
not filled in as 0.
"""
from __future__ import annotations

from ml.learner.fixer import probe


def fix_features(problem, code, classes):
    """Group F names for ``classes`` only. Other ``GROUP_F`` names are omitted."""
    out = {}
    for cls in classes:
        info = probe(problem, code, cls)
        out[f"f_fix_{cls}"] = 1 if info["hit"] else 0
        out[f"f_fixgain_{cls}"] = info["gain"]
    return out
