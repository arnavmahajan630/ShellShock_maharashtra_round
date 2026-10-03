"""The frozen C subset, interpreter limits, trace events and gate messages.

Contract: plans/03 §2.1, §2.2, §2.4, §3.7.1.
"""

GARBAGE = -858993460        # value of an uninitialised read or an out-of-bounds read
STEP_CAP = 5000             # executed statements before status "timeout"
DEPTH_CAP = 100             # recursion depth before status "timeout"
MAX_RECORDED_STEPS = 2000   # steps kept in a trace; beyond this, truncated = true
FLOAT_TOLERANCE = 1e-3
MAX_LINES = 120             # gate G7
MAX_BYTES = 4096            # gate G7

TRACE_STATUSES = ["ok", "timeout", "runtime_error", "parse_error", "unsupported"]

WORLD_BUILTINS = ["fire", "launch", "open_door", "close_door", "scan"]
LIB_BUILTINS = ["printf", "strlen"]
PRINTF_FORMATS = ["%d", "%i", "%f", "%.Nf", "%c", "%s", "%%", "%lf", "%ld"]

SUPPORTED = [
    "int, float, double (as float), char (as int, char literals)",
    "1D arrays, array initialisers, array parameters (shared with the caller)",
    "+ - * / %, compound assignment, ++/-- (pre and post)",
    "relational operators, && || !, ternary, casts (int) (float) (double)",
    "if/else, for (C99 declaration in init), while, do-while, break, continue, return",
    "user functions, recursion (depth cap 100)",
    "char arrays and string literals with a '\\0' terminator, printf(\"%s\")",
    "strlen (unless the problem forbids it)",
    "#define NAME literal (text substitution), #include lines (stripped)",
    "// and /* */ comments (replaced by spaces before parsing; line numbers kept)",
]

# Construct name -> the word shown in the G4 message.
REJECTED = {
    "pointer": "pointers",
    "address_of": "&x",
    "struct": "struct",
    "malloc": "malloc",
    "string_h": "<string.h> functions other than strlen",
    "scanf": "scanf",
    "goto": "goto",
    "multi_dim_array": "multi-dimensional arrays",
    "switch": "switch",
}

# Event types in trace["events"] and steps[].events. Every event has "type" and "line".
EVENT_FIELDS = {
    "uninit_read": ["var"],
    "oob_read": ["arr", "idx", "size"],
    "oob_write": ["arr", "idx", "size"],
    "intdiv": ["remainder_nonzero", "into_float"],
    "div_zero": [],
    "assign_in_cond": ["var", "value"],
    "empty_body": ["kind"],                 # kind: if | for | while
    "step_cap_hit": [],
    "missing_return": [],
    "overflow": [],
    "str_literal_compare": [],
    "array_compare": [],
    "depth_cap_hit": ["fn", "last_args"],
    "discarded_call_value": ["fn"],
}

# Effects are strings in steps[].effects: "<name>" or "<name>:<fields joined by ':'>".
# 03 fixes the names; the field order below is the W0 decision for what 03 leaves open.
EFFECT_FIELDS = {
    "fire": [],
    "launch": [],
    "door_open": [],
    "door_closed": [],
    "scan": ["x"],
    "read_cell": ["i"],
    "read_void": ["i"],
    "write_cell": ["i", "v"],
    "compare": ["i", "j"],
    "call": ["fn", "depth", "args"],        # args joined by ","  e.g. "call:factorial:2:3"
    "ret": ["fn", "depth", "value"],        # e.g. "ret:factorial:2:6"
}

# Keys always present in trace["effects_count"] (03 §2.4); other effect names are added when seen.
EFFECTS_COUNT_KEYS = ["fire", "read_cell", "read_void", "write_cell", "compare", "call"]

# Step rules (W0 decision where 03 is silent):
# - one step per executed statement; a for-header yields a step for its init, each test and each update
# - "vars" holds the scalar locals and parameters of the current frame after the step
# - a variable with no value yet is null; arrays are not in "vars" (rebuild them from the
#   test's args plus write_cell effects)
# - a step is recorded when its statement finishes. A "call" effect sits on the first step
#   inside the called function and a "ret" effect on that function's return step, so the
#   caller's own step comes after the callee's steps
# - trace["events"] lists every occurrence over all tests, in order; each carries "test",
#   the index of the test it happened in. steps[].events carry no "test"

GATE_MESSAGES = {
    "G0": "",
    "G1": "Navigation core is empty.",
    "G2": "No changes detected in the core.",
    "G3a": "Core rejected line {n}: {reason}.",
    "G3b": "That's Python. This ship only speaks C.",
    "G3c": "That's C++. This ship only speaks C.",
    "G4": "This ship doesn't understand `{construct}` yet.",
    "G4b": "Trial rules: count it yourself, no `strlen`.",
    "G5": "Mission needs `{signature}`.",
    "G5b": "This trial needs a warp gate: solve it with recursion.",
    "G6": "Your core gives the same answer whatever the input. The mission changes its inputs.",
    "G7": "Core too large for this mission.",
}
# G8 (smart quotes, non-ASCII punctuation) is fixed silently and reported as G0.
