"""Check the problem bank against the schema, the coverage rules and both backends.

    python -m ml.problems.check main
    python -m ml.problems.check dsa

`main` is the 16 main-game problems (package B1). `dsa` is the 18 Deep Space
problems (package B2) and reuses the same rules, plus the D-class coverage line.
A problem is kept only when every correct variant passes every test on gcc and
on the interpreter.
"""
import json
import sys
from pathlib import Path

from ml.contracts.classes import DSA_CLASSES, MAIN_CLASSES, MISCONCEPTIONS

# OTHER is a model label, not one of the 17 misconception ids, but P16 and Q09 target it.
EXPOSURE_KEYS = set(MISCONCEPTIONS) | {"OTHER"}
from ml.contracts.schemas import Problem
from ml.runner import run_tests

ROOT = Path(__file__).parent
KINDS = {"main": ROOT / "main", "dsa": ROOT / "dsa"}

MAIN_IDS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08",
    "P09", "P10", "P11", "P12", "P13", "P14", "P16", "P17",
]
DSA_IDS = [f"Q{n:02d}" for n in range(1, 19)]
MAIN_HOLDOUT = {"P04", "P09", "P13"}
DSA_HOLDOUT = {"Q04", "Q09", "Q16"}

# 03 §3.3.1 puts M04 on P07 (train) and P13 (holdout) only, so it cannot
# appear on two training problems. Every other main class can, and must.
M04_TRAIN = "P07"
M04_HOLDOUT = "P13"

# 03 §3.3.2 coverage line. Every id here is a training problem.
DSA_COVERAGE = {
    "D01": ["Q01", "Q08", "Q15"],
    "D02": ["Q03", "Q05"],
    "D03": ["Q06", "Q07", "Q10", "Q11"],
    "D04": ["Q06", "Q07"],
    "D05": ["Q17", "Q18"],
    "D06": ["Q17", "Q18"],
    "D07": ["Q17", "Q18"],
    "D08": ["Q14", "Q15"],
}

LOOP_FAMILIES = {
    "count_loop", "countdown_loop", "array_accumulate", "array_count_if",
    "while_progress", "accumulate_product", "average", "array_max", "prefix_loop",
    "index_access", "linear_search", "count_match", "binary_search", "flag_search",
    "guess_halving", "bubble_sort", "selection_sort", "pairwise_check",
    "running_max2", "two_pointer_swap", "shift", "two_pointer_scan",
    "string_scan", "string_count", "string_two_pointer",
}
RECURSION_IDS = {"Q16", "Q17", "Q18"}
IN_PLACE_IDS = {"Q06", "Q07", "Q10", "Q11"}
SORT_IDS = {"Q06", "Q07"}


def load_problems(kind):
    folder = KINDS[kind]
    problems = []
    for path in sorted(folder.glob("*.json")):
        problems.append(json.loads(path.read_text(encoding="utf-8")))
    return problems


def _exposure_counts(problems, split):
    counts = {}
    for problem in problems:
        if problem["split"] != split:
            continue
        for key in problem["exposure"]:
            counts[key] = counts.get(key, 0) + 1
    return counts


def structure_errors(kind, problems):
    """Rules that do not run any code."""
    errors = []
    expected = MAIN_IDS if kind == "main" else DSA_IDS
    holdout = MAIN_HOLDOUT if kind == "main" else DSA_HOLDOUT
    found = [p["problem_id"] for p in problems]
    if found != expected:
        errors.append(f"ids are {found}, expected {expected}")
        return errors

    for problem in problems:
        pid = problem["problem_id"]
        try:
            Problem.model_validate(problem)
        except Exception as exc:
            errors.append(f"{pid}: schema: {exc}")
            continue
        if (problem.get("planet") is None) == (problem.get("sector") is None):
            errors.append(f"{pid}: exactly one of planet and sector must be set")
        if kind == "main" and problem.get("sector") is not None:
            errors.append(f"{pid}: a main problem has no sector")
        if kind == "dsa" and problem.get("planet") is not None:
            errors.append(f"{pid}: a DSA problem has no planet")
        want_split = "holdout_problem" if pid in holdout else "train"
        if problem["split"] != want_split:
            errors.append(f"{pid}: split is {problem['split']}, expected {want_split}")
        if len(problem["correct_variants"]) < 3:
            errors.append(f"{pid}: needs at least 3 correct variants")
        n_tests = len(problem["tests"])
        if not 4 <= n_tests <= 6:
            errors.append(f"{pid}: needs 4 to 6 tests, has {n_tests}")
        if sum(1 for t in problem["tests"] if t.get("sample")) != 2:
            errors.append(f"{pid}: needs exactly 2 sample tests")
        if not any(t.get("adversarial") for t in problem["tests"]):
            errors.append(f"{pid}: needs an adversarial test")
        if not problem["allowed_ops"]:
            errors.append(f"{pid}: allowed_ops is empty")
        bad_keys = set(problem["exposure"]) - EXPOSURE_KEYS
        if bad_keys:
            errors.append(f"{pid}: exposure keys {sorted(bad_keys)} are not classes")
        for key, value in problem["exposure"].items():
            if value not in (0.4, 0.7):
                errors.append(f"{pid}: exposure {key} is {value}, expected 0.4 or 0.7")
        if kind == "main":
            item = problem.get("predict_item")
            if not item or not item.get("belief"):
                errors.append(f"{pid}: predict_item needs a belief map")
            elif item["correct"] not in item["options"]:
                errors.append(f"{pid}: predict correct is not an option")
            else:
                for cls, answer in item["belief"].items():
                    if answer not in item["options"] or answer == item["correct"]:
                        errors.append(f"{pid}: belief {cls}={answer} is not a wrong option")
        if problem["family"] in LOOP_FAMILIES:
            text = "\n".join(problem["correct_variants"])
            if "for" not in text or "while" not in text:
                errors.append(f"{pid}: loop family needs a for variant and a while variant")
        if pid in RECURSION_IDS and "recursion" not in problem["prompt"].lower():
            errors.append(f"{pid}: prompt must say to use recursion")
        if pid == "Q13":
            if "strlen" not in problem["forbid"]:
                errors.append("Q13: forbid must list strlen")
            if "strlen" not in problem["prompt"].lower():
                errors.append("Q13: prompt must say not to use strlen")
        if pid in SORT_IDS and not problem.get("one_pass"):
            errors.append(f"{pid}: sorts need a one_pass helper")
        if pid in IN_PLACE_IDS and not any("array0" in t["expect"] for t in problem["tests"]):
            errors.append(f"{pid}: an in-place problem needs an array0 expectation")
        if pid in RECURSION_IDS and not any("max_depth_le" in t["expect"] for t in problem["tests"]):
            errors.append(f"{pid}: recursion tests need max_depth_le")

    if errors:
        return errors

    if kind == "main":
        train = _exposure_counts(problems, "train")
        for cls in MAIN_CLASSES:
            if cls == "M04":
                continue
            if train.get(cls, 0) < 2:
                errors.append(f"{cls} is on {train.get(cls, 0)} training problems, needs 2")
        by_id = {p["problem_id"]: p for p in problems}
        if "M04" not in by_id[M04_TRAIN]["exposure"] or by_id[M04_TRAIN]["split"] != "train":
            errors.append("M04 must be on training problem P07")
        if "M04" not in by_id[M04_HOLDOUT]["exposure"] or by_id[M04_HOLDOUT]["split"] != "holdout_problem":
            errors.append("M04 must be on held-out problem P13")
    else:
        by_id = {p["problem_id"]: p for p in problems}
        for cls, ids in DSA_COVERAGE.items():
            for pid in ids:
                problem = by_id[pid]
                if problem["split"] != "train":
                    errors.append(f"{pid} covers {cls} but is not a training problem")
                elif cls not in problem["exposure"]:
                    errors.append(f"{pid} must expose {cls}")
            if cls not in DSA_CLASSES:
                errors.append(f"{cls} is not a DSA class")
    return errors


def execution_errors(problems, backends=("gcc", "interp")):
    errors = []
    for backend in backends:
        for problem in problems:
            pid = problem["problem_id"]
            for index, code in enumerate(problem["correct_variants"], start=1):
                try:
                    result = run_tests(problem, code, backend=backend)
                except Exception as exc:
                    errors.append(f"{pid} variant {index} on {backend}: {type(exc).__name__}: {exc}")
                    continue
                for nth, test in enumerate(result["tests"]["results"]):
                    if test["pass"]:
                        continue
                    errors.append(
                        f"{pid} variant {index} on {backend}, test {nth} args {test['args']}: "
                        f"expected {test['expected']} got {test['got']}"
                    )
    return errors


def check(kind, backends=("gcc", "interp")):
    problems = load_problems(kind)
    errors = structure_errors(kind, problems)
    if not errors:
        errors = execution_errors(problems, backends)
    return problems, errors


def main(argv):
    if len(argv) != 2 or argv[1] not in KINDS:
        print("usage: python -m ml.problems.check main|dsa", file=sys.stderr)
        return 2
    kind = argv[1]
    problems, errors = check(kind)
    if errors:
        print(f"{kind}: {len(errors)} problem(s)")
        for error in errors:
            print(f"  {error}")
        return 1
    variants = sum(len(p["correct_variants"]) for p in problems)
    print(f"{kind}: {len(problems)} problems, {variants} variants, gcc ok, interpreter ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
