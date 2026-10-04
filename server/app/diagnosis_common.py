"""Shared helpers for the rule-based diagnoser stand-ins.

`conditions_rules.py` and `loops_rules.py` fill the same `ml/contracts/schemas.Diagnosis` /
`Intervention` contracts with deterministic pattern-matching — the trained model (`ml/model/`)
isn't ready. This module holds what both planets share: the posterior/card builders, the
class→intervention-modality table (`design_plan/02_DESIGN_SCOPE_v3.md` D06), and the trap-item
bank (keyed by class, not by planet — the same misconception gets the same trap question
wherever it's diagnosed).
"""
from ml.contracts.classes import CLASS_INFO, band

FLOOR = 0.003
ALL_CLASSES = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10",
    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08",
]

# Which D06 intervention panel is primary for a class. D01/D02/.../D08 are DSA classes
# (out of scope for Conditions/Loops) and aren't listed here.
CLASS_MODALITY = {
    "M01": "trace_timeline",
    "M02": "trace_timeline",
    "M03": "trace_timeline",
    "M06": "trace_timeline",
    "M07": "trace_timeline",
    "M10": "trace_timeline",
    "M08": "memory_strip",
    "M05": "value_meter",
    "M04": "value_meter",
}


def to_internal_problem(public_problem):
    """The harness/gate want `tests`; the public fixture shape calls them `sample_tests`."""
    p = dict(public_problem)
    p["tests"] = public_problem.get("tests") or public_problem.get("sample_tests") or []
    p.setdefault("display_test", 0)
    p.setdefault("forbid", public_problem.get("forbid") or [])
    return p


def card(class_id, p):
    info = CLASS_INFO[class_id]
    return {"id": class_id, "p": p, "name": info["name"], "subtitle": info["subtitle"], "band": band(p)}


def posterior(primary=None, p_primary=0.0, secondary=None, p_secondary=0.0, correct=FLOOR, other=FLOOR):
    post = {c: FLOOR for c in ALL_CLASSES}
    post["CORRECT"], post["OTHER"] = correct, other
    if primary:
        post[primary] = p_primary
    if secondary:
        post[secondary] = p_secondary
    return post


# ---------------------------------------------------------------- trap items (D07, no execution)

TRAP_ITEMS = {
    "M01": {
        "prompt": "`for (int i = 0; i <= n; i++)` reads `a[i]` each pass, over an array of n cells. How many cells does this read?",
        "code": "int n = 3;\nfor (int i = 0; i <= n; i++) {\n    // read a[i]\n}",
        "options": ["3", "4 — one past the end", "n"],
        "correct": 1,
    },
    "M02": {
        "prompt": "`i` is never changed inside the loop body. What happens?",
        "code": "int i = 0;\nwhile (i < n) {\n    fire();\n}",
        "options": ["Fires n times then stops", "Fires forever — i never changes", "Fires 0 times"],
        "correct": 1,
    },
    "M03": {
        "prompt": "`total` starts at 0 before the loop; inside the loop it's reassigned (not `+=`) every pass. What is `total` after?",
        "code": "int total = 0;\nfor (int i = 0; i < n; i++) {\n    total = cells[i];\n}",
        "options": ["The sum of all cells", "Just the last cell's value — each pass overwrites it", "0"],
        "correct": 1,
    },
    "M05": {
        "prompt": "`int m;` then `if (b > m)` on the very next line — what is `m` the first time?",
        "code": "int m;\nif (b > m) {\n    m = b;\n}",
        "options": ["0", "Whatever was already in that memory — unpredictable", "b"],
        "correct": 1,
    },
    "M06": {
        "prompt": "`if (open = 1)` — what does this `if` do?",
        "code": "int open = 0;\nif (open = 1) {\n    // ...\n}",
        "options": ["Checks whether open equals 1", "Sets open to 1, and the if always runs", "Does nothing"],
        "correct": 1,
    },
    "M07": {
        "prompt": "`if (energy < 30);` — what runs inside this `if`?",
        "code": "if (energy < 30);\n{\n    return 0;\n}",
        "options": ["Nothing — the `;` ends the if right there", "The block below, only when true", "A syntax error"],
        "correct": 0,
    },
    "M08": {
        "prompt": "`int a[5];` — what is the index of the last valid cell?",
        "code": "int a[5];",
        "options": ["5", "4", "depends on values"],
        "correct": 1,
    },
    "M10": {
        "prompt": "A function `printf`s the answer but never hits a `return` — what does the caller receive?",
        "code": "int f(int n) {\n    printf(\"%d\", n);\n}",
        "options": ["The printed value", "An unpredictable value — printing isn't returning", "0"],
        "correct": 1,
    },
}


def grade_trap(class_id, answer):
    item = TRAP_ITEMS.get(class_id)
    if item is None:
        return False
    try:
        return int(answer) == item["correct"]
    except (TypeError, ValueError):
        return False
