"""Class ids and their display text.

Contract: plans/03 §3.1 (ids, wrong beliefs), plans/02 §2.3 (ship names, subtitles).
Do not rename or reorder anything here.
"""

MAIN_CLASSES = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10"]
DSA_CLASSES = ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"]
MISCONCEPTIONS = MAIN_CLASSES + DSA_CLASSES          # 17

# Model output order (03 §5.1).
LABELS = MISCONCEPTIONS + ["CORRECT", "OTHER"]       # 19

UNSEEN_CLASSES = ["U1", "U2"]                        # test only, never trained
STRONG_CLASSES = ["M09"]                             # only if Strong S4 lands

# Sentence reader labels (05 §6).
REASON_LABELS = MISCONCEPTIONS + ["CORRECT_REASON"]  # 18

STATES = ["UNSEEN", "ACTIVE", "TREATING", "PROBATION", "STABLE", "MASTERED", "RELAPSED"]

CLASS_INFO = {
    "M01": {
        "code_name": "LOOP_BOUND_OFF_BY_ONE",
        "name": "Boundary Drift",
        "subtitle": "Loop runs one step too many or too few",
        "belief": "`i<=n` (from 0) runs n times; whether the bound is the last value included or excluded is mixed up",
    },
    "M02": {
        "code_name": "LOOP_NO_PROGRESS",
        "name": "Stalled Thruster",
        "subtitle": "Loop variable never moves toward the exit",
        "belief": "The loop variable advances by itself and the loop ends by itself",
    },
    "M03": {
        "code_name": "ACCUMULATOR_RESET",
        "name": "Memory Wipe",
        "subtitle": "Running total is reset inside the loop",
        "belief": "The total keeps its value across passes even if it is set again inside the loop",
    },
    "M04": {
        "code_name": "INT_DIVISION",
        "name": "Fraction Shear",
        "subtitle": "int ÷ int drops the decimals",
        "belief": "int / int gives the exact decimal",
    },
    "M05": {
        "code_name": "UNINITIALIZED_VAR",
        "name": "Static Signal",
        "subtitle": "Variable used before it was given a value",
        "belief": "Local variables start at 0",
    },
    "M06": {
        "code_name": "ASSIGN_IN_CONDITION",
        "name": "Sensor Overwrite",
        "subtitle": "`=` stores a value; `==` compares",
        "belief": "`=` compares",
    },
    "M07": {
        "code_name": "STRAY_SEMICOLON",
        "name": "Ghost Semicolon",
        "subtitle": "A stray `;` gives the if/loop an empty body",
        "belief": "A `;` after the header is harmless",
    },
    "M08": {
        "code_name": "ARRAY_INDEX_BASE",
        "name": "Index Origin Fault",
        "subtitle": "Arrays start at 0; last cell is n−1",
        "belief": "Arrays are indexed 1..n",
    },
    "M10": {
        "code_name": "PRINT_NOT_RETURN",
        "name": "Silent Messenger",
        "subtitle": "`printf` shows a value; `return` hands it back",
        "belief": "Printing the value hands it back to the caller",
    },
    "D01": {
        "code_name": "SEARCH_EARLY_EXIT",
        "name": "Premature Abort",
        "subtitle": "The search gives up after the first miss",
        "belief": "The else branch means 'not this one, keep looking'; it doesn't end or overwrite anything",
    },
    "D02": {
        "code_name": "BSEARCH_NO_SHRINK",
        "name": "Frozen Window",
        "subtitle": "Binary search window never shrinks (`low = mid`)",
        "belief": "mid is excluded from the window automatically",
    },
    "D03": {
        "code_name": "SWAP_OVERWRITE",
        "name": "Cargo Overwrite",
        "subtitle": "Swapping without a temp loses a value",
        "belief": "Two assignments happen at the same time",
    },
    "D04": {
        "code_name": "SINGLE_PASS_SORT",
        "name": "Half-Sorted Hold",
        "subtitle": "One pass doesn't sort the whole array",
        "belief": "One pass over the array sorts it",
    },
    "D05": {
        "code_name": "MISSING_BASE_CASE",
        "name": "Endless Warp",
        "subtitle": "Recursion has no reachable base case",
        "belief": "Recursion stops by itself when the work is done",
    },
    "D06": {
        "code_name": "RECURSION_NO_SHRINK",
        "name": "Static Warp",
        "subtitle": "The recursive call doesn't move toward the base case",
        "belief": "The recursive call moves toward the base case on its own",
    },
    "D07": {
        "code_name": "RECURSIVE_RESULT_DISCARDED",
        "name": "Lost Echo",
        "subtitle": "The recursive call's result is thrown away",
        "belief": "The recursive call's result is combined automatically",
    },
    "D08": {
        "code_name": "STRING_EQ_COMPARE",
        "name": "Signal Mismatch",
        "subtitle": "`==` doesn't compare text in C",
        "belief": "`==` compares text",
    },
    "CORRECT": {"code_name": "CORRECT", "name": "Correct", "subtitle": "Passes every test", "belief": None},
    "OTHER": {
        "code_name": "OTHER",
        "name": "Unclassified Fault",
        "subtitle": "A bug we don't have a lesson for yet",
        "belief": None,
    },
}

# Shown when status == "novel" (not a model output).
NOVEL_DISPLAY = {"name": "Unknown Anomaly", "subtitle": "Doesn't match any known pattern"}

# 03 §3.2. T5 is Strong (M09 vs M10) and is left out.
TWIN_SETS = {
    "T1": {"type": "HARD", "members": ["M01", "M08"]},
    "T2": {"type": "STRUCTURAL", "members": ["M02", "M07"]},
    "T3": {"type": "STRUCTURAL", "members": ["M06", "M07"]},
    "T4": {"type": "STRUCTURAL", "members": ["M03", "M05"]},
    "T6": {"type": "STRUCTURAL", "members": ["D02", "M02"]},
    "T7": {"type": "STRUCTURAL", "members": ["D05", "D06"]},
    "T8": {"type": "STRUCTURAL", "members": ["D07", "M10"]},
    "T9": {"type": "STRUCTURAL", "members": ["D01", "M03"]},
}

SECTORS = ["searching", "sorting", "array_tech", "strings", "recursion"]
PLANETS = ["conditions", "loops", "arrays", "variables", "functions"]


def band(p: float) -> str:
    """Confidence word shown next to a probability (02 §2.4)."""
    if p >= 0.75:
        return "Likely"
    if p >= 0.45:
        return "Possible"
    return "Unsure"


def twin_set_of(a: str, b: str) -> str | None:
    """The twin set containing both classes, if any."""
    for set_id, info in TWIN_SETS.items():
        if a in info["members"] and b in info["members"] and a != b:
            return set_id
    return None
