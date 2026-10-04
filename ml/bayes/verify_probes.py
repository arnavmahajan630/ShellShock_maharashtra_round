"""Checks the probe, trap and exam-item banks (plans/03 §6.3, §6.6, §8.3, §8.5.1).

    python -m ml.bayes.verify_probes                 interpreter and gcc
    python -m ml.bayes.verify_probes --backend interp

Files:
    ml/data/probes.json        14 probes, one object per probe (schemas.Probe)
    ml/data/items.json         17 traps, one per mistake (schemas.TrapItem)
    ml/data/exam_items.json    exam trace items (schemas.ExamTraceItem)
    ml/data/item_checks.json   how each item is checked, keyed by its id

A check is either {"manual": "<why nothing can be run>"} or
    {"kind", "tail", "says", "state_query", "belief_code": {class: fragment}}
(see ml/items/snippet.py). The item's own `code` is what runs. Its answer must equal `correct`
on the interpreter, and on gcc too whenever C defines what the code does. A `belief_code`
fragment is the program a learner with that mistake thinks they are reading; its answer must
equal the item's belief answer for that class.
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from ml.contracts.classes import MISCONCEPTIONS, SECTORS, TWIN_SETS
from ml.contracts.schemas import ExamTraceItem, Probe, TrapItem
from ml.items import snippet

DATA = Path(__file__).resolve().parents[1] / "data"
BANKS = {"probes.json": (Probe, "probe_id"), "items.json": (TrapItem, "item_id"),
         "exam_items.json": (ExamTraceItem, "item_id")}
MIN_PER_SECTOR = 4
MIN_PROBES_PER_TWIN_SET = 2


def load(folder=DATA):
    """({file: [items]}, checks)"""
    banks = {name: json.loads((folder / name).read_text(encoding="utf-8")) for name in BANKS}
    checks = json.loads((folder / "item_checks.json").read_text(encoding="utf-8"))
    return banks, checks


def _flat(code):
    return " ".join(code.split())


def option_problems(item_id, item):
    """Rules every choice item must meet."""
    out = []
    options, correct, belief = item["options"], item["correct"], item["belief"]
    if not 3 <= len(options) <= 4 or len(set(options)) != len(options):
        out.append(f"{item_id}: needs 3 or 4 different options")
    if correct not in options:
        out.append(f"{item_id}: correct answer {correct!r} is not an option")
    for cls, value in belief.items():
        if cls not in MISCONCEPTIONS:
            out.append(f"{item_id}: unknown class {cls}")
        if value not in options:
            out.append(f"{item_id}: belief answer {value!r} of {cls} is not an option")
    if belief and all(value == correct for value in belief.values()):
        out.append(f"{item_id}: no belief answer differs from the correct one")
    return out


def core_twin_sets():
    """Twin sets whose members are all model classes. A set with a Strong class (T5 with M09)
    needs no probes until that class is trained."""
    return [set_id for set_id, info in TWIN_SETS.items() if set(info["members"]) <= set(MISCONCEPTIONS)]


def structure_problems(banks, checks):
    """Everything that can be checked without running code. A list of sentences; empty means sound."""
    out, seen_ids = [], set()
    for name, (model, key) in BANKS.items():
        for item in banks[name]:
            item_id = item.get(key, "?")
            try:
                model.model_validate(item)
            except ValidationError as error:
                out.append(f"{item_id}: does not fit {model.__name__} ({error.errors()[0]['msg']})")
                continue
            if item_id in seen_ids:
                out.append(f"{item_id}: id used twice")
            seen_ids.add(item_id)
            out += option_problems(item_id, item)
            if item_id not in checks:
                out.append(f"{item_id}: no entry in item_checks.json")
    for extra in sorted(set(checks) - seen_ids):
        out.append(f"{extra}: check without an item")

    per_set = dict.fromkeys(core_twin_sets(), 0)
    for probe in banks["probes.json"]:
        for set_id in probe.get("twin_sets", []):
            if set_id not in TWIN_SETS:
                out.append(f"{probe['probe_id']}: unknown twin set {set_id}")
                continue
            per_set[set_id] = per_set.get(set_id, 0) + 1
            a, b = TWIN_SETS[set_id]["members"]
            belief = probe["belief"]
            if a not in belief or b not in belief:
                out.append(f"{probe['probe_id']}: no belief answer for both members of {set_id}")
            elif belief[a] == belief[b]:
                out.append(f"{probe['probe_id']}: {a} and {b} give the same answer, so it cannot split {set_id}")
    for set_id, n in per_set.items():
        if n < MIN_PROBES_PER_TWIN_SET:
            out.append(f"twin set {set_id}: {n} probes, needs {MIN_PROBES_PER_TWIN_SET}")

    targets = [trap.get("target") for trap in banks["items.json"]]
    for cls in MISCONCEPTIONS:
        if targets.count(cls) != 1:
            out.append(f"traps: {targets.count(cls)} for {cls}, needs exactly 1")
    for trap in banks["items.json"]:
        belief = trap.get("belief", {})
        if list(belief) != [trap.get("target")] or belief.get(trap.get("target")) == trap.get("correct"):
            out.append(f"{trap.get('item_id')}: a trap needs one belief answer, for its target, that is wrong")

    per_sector, covered = dict.fromkeys(SECTORS, 0), set()
    for item in banks["exam_items.json"]:
        item_id = item.get("item_id")
        if item.get("sector") not in SECTORS:
            out.append(f"{item_id}: unknown sector {item.get('sector')}")
        else:
            per_sector[item["sector"]] += 1
        if item.get("difficulty") not in (1, 2, 3):
            out.append(f"{item_id}: difficulty must be 1, 2 or 3")
        belief = item.get("belief", {})
        if not 1 <= len(belief) <= 3:
            out.append(f"{item_id}: needs belief answers for 1 to 3 classes")
        if any(value == item.get("correct") for value in belief.values()):
            out.append(f"{item_id}: a belief answer equals the correct answer")
        if len(item.get("options", [])) != 3:
            out.append(f"{item_id}: an exam trace item has exactly 3 options")
        covered |= set(belief)
    for sector, n in per_sector.items():
        if n < MIN_PER_SECTOR:
            out.append(f"exam items: {n} in {sector}, needs {MIN_PER_SECTOR}")
    for cls in MISCONCEPTIONS:
        if cls not in covered:
            out.append(f"exam items: no item has a belief answer for {cls}")

    # the exam must not leak into reassessment (03 §6.6): no shared code with traps or probes
    used = {_flat(trap["code"]): trap["item_id"] for trap in banks["items.json"]}
    used.update({_flat(probe["code"]): probe["probe_id"] for probe in banks["probes.json"]})
    for item in banks["exam_items.json"]:
        if _flat(item["code"]) in used:
            out.append(f"{item['item_id']}: same code as {used[_flat(item['code'])]}")
    return out


def _jobs(banks, checks):
    """One job per fragment to run: (item_id, what, fragment, check, expected answer)."""
    jobs, manual = [], []
    for name, (_, key) in BANKS.items():
        for item in banks[name]:
            item_id, how = item[key], checks.get(item[key])
            if how is None:
                continue
            if "manual" in how:
                manual.append((item_id, how["manual"]))
                continue
            jobs.append((item_id, "correct", item["code"], how, item["correct"]))
            for cls, fragment in (how.get("belief_code") or {}).items():
                jobs.append((item_id, f"belief of {cls}", fragment, how, item["belief"].get(cls)))
    return jobs, manual


def run_problems(banks, checks, backends=("interp", "gcc")):
    """Run every check. Returns (problems, summary)."""
    jobs, manual = _jobs(banks, checks)
    results = snippet.check_many(
        [{"fragment": fragment, "tail": how.get("tail"), "kind": how["kind"], "says": how.get("says"),
          "state_query": how.get("state_query")} for _, _, fragment, how, _ in jobs], backends)
    out = []
    for (item_id, what, _, _, expected), got in zip(jobs, results):
        if got["text"] is None:
            out.append(f"{item_id}: {what}: the interpreter cannot run it ({got['error']})")
        elif got["text"] != expected:
            out.append(f"{item_id}: {what}: file says {expected!r}, the interpreter says {got['text']!r}")
        elif got["error"]:
            out.append(f"{item_id}: {what}: {got['error']}")

    belief_total = sum(len(item["belief"]) for name in BANKS for item in banks[name])
    summary = {
        "items": sum(len(banks[name]) for name in BANKS),
        "probes": len(banks["probes.json"]), "traps": len(banks["items.json"]), "exam": len(banks["exam_items.json"]),
        "run": sum(1 for job in jobs if job[1] == "correct"),
        "manual": manual,
        "on_gcc": sum(1 for got in results if got["gcc"] is not None),
        "interp_only": sum(1 for got in results if got["text"] is not None and not got["defined"]),
        "belief_answers": belief_total,
        "belief_answers_run": sum(1 for job in jobs if job[1] != "correct"),
    }
    return out, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["both", "interp"], default="both")
    args = parser.parse_args(argv)
    backends = ("interp",) if args.backend == "interp" else ("interp", "gcc")

    banks, checks = load()
    problems = structure_problems(banks, checks)
    ran, summary = run_problems(banks, checks, backends)
    problems += ran

    print(f"{summary['probes']} probes, {summary['traps']} traps, {summary['exam']} exam items")
    print(f"correct answers run on the interpreter: {summary['run']} of {summary['items']}")
    if "gcc" in backends:
        print(f"fragments also run on gcc: {summary['on_gcc']}  "
              f"(interpreter only, because C leaves them undefined: {summary['interp_only']})")
    print(f"belief answers confirmed by running the believed program: "
          f"{summary['belief_answers_run']} of {summary['belief_answers']}")
    for item_id, why in summary["manual"]:
        print(f"not run, check by hand: {item_id} ({why})")
    for line in problems:
        print("PROBLEM", line)
    print("VERIFY FAILED" if problems else "VERIFY PASSED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
