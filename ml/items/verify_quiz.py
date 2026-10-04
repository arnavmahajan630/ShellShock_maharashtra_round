"""Checks the quiz bank (plans/05 §3.1).

    python -m ml.items.verify_quiz                    interpreter and gcc
    python -m ml.items.verify_quiz --backend interp

Reads ml/data/quiz_items.json and ml/data/quiz_checks.json. It does not trust how the items
were built: every answer is worked out again by running the code.

Rules (05 §3.1):
  - 3 or 4 different options; `correct` is one of them; every belief answer is an option and
    differs from `correct`; belief classes are listed in `classes`
  - `next_state` items carry a `state_query`; `reasoning` items have no options and expect
    CORRECT_REASON
  - an item with code runs on the interpreter and gives `correct`; gcc agrees wherever C defines
    what the code does; `verified` says exactly that
  - a belief answer equals what the believed program (quiz_checks.json) gives
  - an item with no check is a concept question: it must be marked `manual` and is listed
  - no item shares its code with a probe, a trap or an exam trace item
  - targets: per class 2 mcq, 2 predict_output, 1 next_state, 1 reasoning; 12 collision items
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ml.contracts.classes import MISCONCEPTIONS, PLANETS, SECTORS
from ml.contracts.schemas import QuizItem
from ml.items import snippet

DATA = Path(__file__).resolve().parents[1] / "data"
PER_CLASS = {"mcq": 2, "predict_output": 2, "next_state": 1, "reasoning": 1}
MIN_COLLISIONS = 12
OTHER_BANKS = {"probes.json": "probe_id", "items.json": "item_id", "exam_items.json": "item_id"}


def load(folder=DATA):
    items = json.loads((folder / "quiz_items.json").read_text(encoding="utf-8"))
    checks = json.loads((folder / "quiz_checks.json").read_text(encoding="utf-8"))
    return items, checks


def _flat(code):
    return " ".join(code.split())


def collision_answers(item):
    """Wrong options that are the belief answer of two or more classes."""
    counts = Counter(item.get("belief", {}).values())
    return [answer for answer, n in counts.items() if n >= 2 and answer != item.get("correct")]


def structure_problems(items, checks, other_code=None):
    """Everything that can be checked without running code."""
    out, seen = [], set()
    for item in items:
        item_id = item.get("item_id", "?")
        try:
            QuizItem.model_validate(item)
        except ValidationError as error:
            out.append(f"{item_id}: does not fit QuizItem ({error.errors()[0]['msg']})")
            continue
        if item_id in seen:
            out.append(f"{item_id}: id used twice")
        seen.add(item_id)
        if item["concept"] not in PLANETS + SECTORS:
            out.append(f"{item_id}: unknown concept {item['concept']}")
        if not item["explain"].strip():
            out.append(f"{item_id}: no explanation")
        if not item["classes"] or any(c not in MISCONCEPTIONS for c in item["classes"]):
            out.append(f"{item_id}: classes must be a non-empty list of known classes")
        if item["type"] == "reasoning":
            if item["options"] or item["correct"] is not None or item["belief"]:
                out.append(f"{item_id}: a reasoning item has no options, correct answer or belief answers")
            if item["expected"] != "CORRECT_REASON":
                out.append(f"{item_id}: a reasoning item expects CORRECT_REASON")
        else:
            options, correct, belief = item["options"], item["correct"], item["belief"]
            if not 3 <= len(options) <= 4 or len(set(options)) != len(options):
                out.append(f"{item_id}: needs 3 or 4 different options")
            if correct not in options:
                out.append(f"{item_id}: correct answer {correct!r} is not an option")
            if not belief:
                out.append(f"{item_id}: no belief answer")
            for cls, value in belief.items():
                if cls not in item["classes"]:
                    out.append(f"{item_id}: belief class {cls} is not in classes")
                if value not in options:
                    out.append(f"{item_id}: belief answer {value!r} of {cls} is not an option")
                if value == correct:
                    out.append(f"{item_id}: belief answer of {cls} equals the correct answer")
        if (item["type"] == "next_state") != (item["state_query"] is not None):
            out.append(f"{item_id}: state_query belongs to next_state items, and only to them")
        if item_id not in checks and not item["verified"]["manual"]:
            out.append(f"{item_id}: no check and not marked manual")
        if item_id in checks and item["verified"]["manual"]:
            out.append(f"{item_id}: marked manual but has a check")
        if item_id in checks and not item["code"].strip():
            out.append(f"{item_id}: has a check but no code")
    for extra in sorted(set(checks) - seen):
        out.append(f"{extra}: check without an item")

    codes = Counter(_flat(item["code"]) for item in items if item.get("code", "").strip())
    for item in items:
        code = _flat(item.get("code", ""))
        if code and codes[code] > 1:
            out.append(f"{item.get('item_id')}: same code as another quiz item")
        if code and other_code and code in other_code:
            out.append(f"{item.get('item_id')}: same code as {other_code[code]}")
    return out


def target_problems(items):
    """Counts the bank must reach (05 §3.1)."""
    out = []
    have = Counter((cls, item["type"]) for item in items for cls in item["classes"])
    for cls in MISCONCEPTIONS:
        for kind, wanted in PER_CLASS.items():
            if have[(cls, kind)] < wanted:
                out.append(f"target: {cls} has {have[(cls, kind)]} {kind} items, needs {wanted}")
    collisions = sum(1 for item in items if collision_answers(item))
    if collisions < MIN_COLLISIONS:
        out.append(f"target: {collisions} collision items, needs {MIN_COLLISIONS}")
    return out


def other_bank_code(folder=DATA):
    out = {}
    for name, key in OTHER_BANKS.items():
        path = folder / name
        if path.exists():
            for item in json.loads(path.read_text(encoding="utf-8")):
                out[_flat(item["code"])] = item[key]
    return out


def run_problems(items, checks, backends=("interp", "gcc")):
    """Run every check. Returns (problems, summary)."""
    jobs, labels = [], []
    for item in items:
        how = checks.get(item["item_id"])
        if how is None:
            continue
        common = {"tail": how.get("tail"), "kind": how["kind"], "says": how.get("says")}
        expected = how.get("expect") if item["type"] == "reasoning" else item["correct"]
        jobs.append({"fragment": item["code"], "state_query": item.get("state_query"), **common})
        labels.append((item, "correct", expected))
        believed = how.get("belief_answers") if item["type"] == "reasoning" else item["belief"]
        for cls, fragment in (how.get("belief_code") or {}).items():
            jobs.append({"fragment": fragment, "state_query": how.get("belief_query") or item.get("state_query"),
                         **common})
            labels.append((item, f"belief of {cls}", (believed or {}).get(cls)))
    results = snippet.check_many(jobs, backends)

    out = []
    for (item, what, expected), got in zip(labels, results):
        item_id = item["item_id"]
        if got["text"] is None:
            out.append(f"{item_id}: {what}: the interpreter cannot run it ({got['error']})")
        elif got["text"] != expected:
            out.append(f"{item_id}: {what}: file says {expected!r}, the interpreter says {got['text']!r}")
        elif got["error"]:
            out.append(f"{item_id}: {what}: {got['error']}")
        if what == "correct":
            flags = item["verified"]
            if not flags["interp"]:
                out.append(f"{item_id}: runs on the interpreter but verified.interp is false")
            if "gcc" in backends and flags["gcc"] != (got["gcc"] is not None and got["error"] is None):
                out.append(f"{item_id}: verified.gcc is {flags['gcc']}, but gcc "
                           f"{'was not asked (C leaves this undefined)' if got['gcc'] is None else 'answered'}")
            if item["type"] == "reasoning" and expected is not None and expected not in item["question"]:
                if expected not in ("never stops", "unpredictable"):
                    out.append(f"{item_id}: the question does not state the real output {expected!r}")

    by_type = Counter(item["type"] for item in items)
    summary = {
        "items": len(items), "by_type": dict(by_type),
        "run": sum(1 for _, what, _ in labels if what == "correct"),
        "on_gcc": sum(1 for (_, what, _), got in zip(labels, results) if what == "correct" and got["gcc"] is not None),
        "interp_only": sum(1 for (_, what, _), got in zip(labels, results)
                           if what == "correct" and got["text"] is not None and not got["defined"]),
        "belief_run": sum(1 for _, what, _ in labels if what != "correct"),
        "manual": [item["item_id"] for item in items if item["verified"]["manual"]],
        "collisions": sum(1 for item in items if collision_answers(item)),
    }
    return out, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["both", "interp"], default="both")
    args = parser.parse_args(argv)
    backends = ("interp",) if args.backend == "interp" else ("interp", "gcc")

    items, checks = load()
    problems = structure_problems(items, checks, other_bank_code()) + target_problems(items)
    ran, summary = run_problems(items, checks, backends)
    problems += ran

    print(f"{summary['items']} quiz items: " + ", ".join(f"{n} {kind}" for kind, n in sorted(summary["by_type"].items())))
    print(f"run on the interpreter: {summary['run']}")
    if "gcc" in backends:
        print(f"also agreed by gcc: {summary['on_gcc']}  "
              f"(interpreter only, because C leaves them undefined: {summary['interp_only']})")
    print(f"belief answers confirmed by running the believed program: {summary['belief_run']}")
    print(f"collision items: {summary['collisions']}")
    print(f"marked manual (no code to run; skim by hand): {len(summary['manual'])} {summary['manual']}")
    for line in problems:
        print("PROBLEM", line)
    print("VERIFY FAILED" if problems else "VERIFY PASSED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
