"""Operator registry for the dataset generator (package C3).

C1 and C2 own the operators. This module only routes ``sites``, ``apply`` and
``holds`` and matches ``allowed_ops`` globs. ``op_variant`` is ``op_id``.
"""
from __future__ import annotations

import fnmatch

from ml.generate.ops_dsa import all_ops as dsa_ops
from ml.generate.ops_dsa import apply as dsa_apply
from ml.generate.ops_dsa import sites as dsa_sites
from ml.generate.ops_main import all_ops as main_ops
from ml.generate.ops_main import apply as main_apply
from ml.generate.ops_main import sites as main_sites
from ml.generate.predicates_dsa import holds as holds_dsa
from ml.generate.predicates_main import holds as holds_main

# DSA surfaces that live in ops_dsa / holds_dsa, plus every D01–D08 operator.
_DSA_IDS = frozenset(op.op_id for op in dsa_ops())
AMB_OPS = frozenset({"amb_le_array", "amb_start1_array", "amb_pair_bound"})


def operators():
    """Main operators, then DSA operators. Test-only U1/U2 stay out."""
    return [op for op in list(main_ops()) + list(dsa_ops()) if not op.test_only]


def class_op_ids():
    """Every operator whose predicate can reject an OTHER row."""
    return [op.op_id for op in operators() if not op.op_id.startswith("oth_")]


def is_dsa_op(op_id):
    return op_id in _DSA_IDS


def allowed(op, globs):
    return any(fnmatch.fnmatch(op.op_id, pattern) for pattern in globs or [])


def find_sites(op, code):
    try:
        if is_dsa_op(op.op_id):
            return list(dsa_sites(op, code))
        return list(main_sites(op, code))
    except Exception:
        return []


def apply_op(op, code, site):
    if is_dsa_op(op.op_id):
        return dsa_apply(op, code, site)
    return main_apply(op, code, site)


def predicate_holds(op_id, code):
    """Route a DSA op_id to ``holds_dsa`` and a main op_id to ``holds_main``."""
    try:
        if is_dsa_op(op_id):
            return bool(holds_dsa(op_id, code))
        return bool(holds_main(op_id, code))
    except Exception:
        return False


def primary_label(op):
    return op.labels[0]
