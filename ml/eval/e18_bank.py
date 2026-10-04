"""E18: is the item bank sound? (ml_plan/05 §9), package E-c.

    python -m ml.eval.e18_bank            # writes ml/eval/e18_card.json and ml/eval/e18_card.md

Quiz bank: the whole of `ml.items.verify_quiz` is run again over every item in
ml/data/quiz_items.json (structure rules, the 05 §3.1 targets, and every answer re-worked by running
the code on the interpreter and on gcc, plus every believed program). Nothing is trusted from the
`verified` flags: the flags are compared with the fresh run.

Code bank: `python -m ml.items.build_code_items --check` over ml/data/code_items.json, but only if
package C4 has built them. If it has not, the card says so and reports no code-item numbers. When
the files are there, the card keeps the checker's own output and the counts against the 05 §3.2
targets (1–2 complete_snippet per problem; 2 fix_bug and 2 debug_line per class; about 20% of
debug_line items have no bug).

Extra descriptive counts (not in the plan, no pass/fail): items per class and type against the
05 §3.1 targets, the position of the correct option (a bank whose answer is usually "B" teaches
the wrong thing), option counts, difficulty, and where each item came from.
"""
from __future__ import annotations

import ml.eval._ec_common as ec  # noqa: F401  (thread limits)

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

from ml.contracts.classes import MISCONCEPTIONS
from ml.eval._ec_common import md_table, write_card
from ml.items import verify_quiz as vq

QUIZ = ec.ROOT / "ml" / "data" / "quiz_items.json"
CODE_ITEMS = ec.ROOT / "ml" / "data" / "code_items.json"
CODE_TARGETS = {"complete_snippet": "1-2 per problem", "fix_bug": "2 per class", "debug_line": "2 per class"}


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def source_of(item_id, checks):
    """Hand-written items come from ml/items/quiz_manual.json; their draft ids start with col_ or fill_."""
    draft = (checks.get(item_id) or {}).get("draft", "")
    return "hand-written" if draft.startswith(("col_", "fill_")) else "DeepSeek draft, checked by running"


# ---------------------------------------------------------------- quiz bank

def check_quiz(backends=("interp", "gcc"), log=lambda message: None):
    items, checks = vq.load()
    started = time.perf_counter()
    structure = vq.structure_problems(items, checks, vq.other_bank_code())
    targets = vq.target_problems(items)
    log(f"structure: {len(structure)} problems; targets: {len(targets)} problems; running {len(items)} items ...")
    ran, summary = vq.run_problems(items, checks, backends)
    seconds = time.perf_counter() - started

    problems = structure + targets + ran
    failing = {line.split(":")[0] for line in structure + ran if ":" in line}
    n = len(items)
    flags = [item["verified"] for item in items]
    claimed_gcc = sum(1 for f in flags if f["gcc"])
    claimed_interp = sum(1 for f in flags if f["interp"])

    by_type = Counter(item["type"] for item in items)
    have = Counter((cls, item["type"]) for item in items for cls in item["classes"])
    matrix = {cls: {kind: have[(cls, kind)] for kind in vq.PER_CLASS} for cls in MISCONCEPTIONS}
    short = {cls: {kind: max(0, wanted - have[(cls, kind)]) for kind, wanted in vq.PER_CLASS.items()} for cls in MISCONCEPTIONS}

    positions = Counter(item["options"].index(item["correct"]) for item in items if item["type"] != "reasoning")
    n_choice = sum(positions.values())
    width = max(positions) + 1 if positions else 0
    chi_p = None
    try:
        from scipy.stats import chisquare
        if n_choice:
            # Items have 3 or 4 options, so position 3 can only come from 4-option items:
            # the expected count at each position is the sum of 1/len(options) over items that have it.
            expected = [sum(1.0 / len(item["options"]) for item in items
                            if item["type"] != "reasoning" and len(item["options"]) > i) for i in range(width)]
            chi_p = round(float(chisquare([positions.get(i, 0) for i in range(width)], f_exp=expected).pvalue), 4)
    except Exception:
        pass
    longest = sum(1 for item in items if item["type"] != "reasoning"
                  and len(item["correct"]) == max(len(o) for o in item["options"])
                  and sum(len(o) == len(item["correct"]) for o in item["options"]) == 1)

    return {
        "items": n,
        "by_type": dict(by_type),
        "backends": list(backends),
        "seconds": round(seconds, 1),
        "verify_passed": not problems,
        "problems": problems,
        "items_with_a_problem": len(failing),
        "run_on_interpreter": summary["run"],
        "agreed_by_gcc": summary["on_gcc"],
        "interpreter_only_c_undefined": summary["interp_only"],
        "belief_programs_run": summary["belief_run"],
        "pct_verified_interpreter": pct(n - len(failing), n),
        "pct_verified_gcc": pct(summary["on_gcc"] - len(failing & {i["item_id"] for i in items if i["verified"]["gcc"]}), n),
        "flags_in_file": {"interp_true": claimed_interp, "gcc_true": claimed_gcc, "manual_true": sum(1 for f in flags if f["manual"])},
        "flags_match_fresh_run": not any("verified." in line for line in ran),
        "manual": summary["manual"],
        "n_manual": len(summary["manual"]),
        "collision_items": summary["collisions"],
        "collision_target": vq.MIN_COLLISIONS,
        "per_class_type": matrix,
        "per_class_type_target": dict(vq.PER_CLASS),
        "per_class_type_short": {c: {k: v for k, v in row.items() if v} for c, row in short.items() if any(row.values())},
        "descriptive": {
            "correct_option_position": {str(i): positions.get(i, 0) for i in range(width)},
            "correct_option_position_chi2_p": chi_p,   # against uniform position within each item's own option count
            "correct_is_the_unique_longest_option": longest,
            "choice_items": n_choice,
            "options_per_item": dict(Counter(len(item["options"]) for item in items if item["type"] != "reasoning")),
            "difficulty": dict(sorted(Counter(item["difficulty"] for item in items).items())),
            "concept": dict(sorted(Counter(item["concept"] for item in items).items())),
            "source": dict(Counter(source_of(item["item_id"], checks) for item in items)),
        },
    }


# ---------------------------------------------------------------- code bank

def versus_targets(items):
    """Counts in the file against 05 §3.2. A problem with no snippet is not in this file, so only --check can see it."""
    snippets = Counter(item["problem_id"] for item in items if item["type"] == "complete_snippet")
    outside = {pid: n for pid, n in sorted(snippets.items()) if not 1 <= n <= 2}
    fix_off, debug_off = {}, {}
    fix_n = debug_n = 0
    for cls in MISCONCEPTIONS:
        fix = sum(1 for item in items if item["type"] == "fix_bug" and item.get("planted") == cls)
        debug = sum(1 for item in items if item["type"] == "debug_line" and item.get("planted") == cls)
        fix_n += fix
        debug_n += debug
        if fix != 2:
            fix_off[cls] = fix
        if debug != 2:
            debug_off[cls] = debug
    debug_items = [item for item in items if item["type"] == "debug_line"]
    no_bug = sum(1 for item in debug_items if item.get("planted") is None)
    snippet_met = bool(snippets) and not outside
    return {
        "complete_snippet": {
            "target": CODE_TARGETS["complete_snippet"],
            "items": sum(snippets.values()),
            "problems": len(snippets),
            "with_1": [pid for pid, n in sorted(snippets.items()) if n == 1],
            "with_2": sum(1 for n in snippets.values() if n == 2),
            "outside": outside,
            "met": snippet_met,
        },
        "fix_bug": {"target": CODE_TARGETS["fix_bug"], "items": fix_n, "not_equal_to_2": fix_off, "met": not fix_off},
        "debug_line": {
            "target": CODE_TARGETS["debug_line"],
            "planted_items": debug_n,
            "not_equal_to_2": debug_off,
            "met": not debug_off,
            "no_bug": no_bug,
            "no_bug_of": len(debug_items),
            "no_bug_share": round(no_bug / len(debug_items), 3) if debug_items else None,
        },
        "counts_meet_targets": snippet_met and not fix_off and not debug_off,
    }


def check_code_items(log=lambda message: None):
    have_builder = importlib.util.find_spec("ml.items.build_code_items") is not None
    have_file = CODE_ITEMS.is_file()
    if not (have_builder and have_file):
        missing = [name for name, ok in (("ml/items/build_code_items.py", have_builder), ("ml/data/code_items.json", have_file)) if not ok]
        return {
            "status": "not built",
            "why": "package C4 has not produced " + " and ".join(missing) + " (notes/C4.md says the sources it needs, the generator's verified dataset, did not exist when it looked)",
            "targets_05_3_2": CODE_TARGETS,
            "checked": 0,
        }
    log("running build_code_items --check ...")
    proc = subprocess.run([sys.executable, "-m", "ml.items.build_code_items", "--check"], cwd=ec.ROOT,
                          capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    items = json.loads(CODE_ITEMS.read_text(encoding="utf-8"))
    lines = (proc.stdout or "").strip().splitlines()
    check_line = next((line for line in reversed(lines) if line.startswith(("CHECK ", "FAIL:"))), "")
    versus = versus_targets(items)
    out = {
        "status": "checked",
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "check_line": check_line,
        "output_tail": lines[-25:],
        "errors_tail": (proc.stderr or "").strip().splitlines()[-10:],
        "checked": len(items),
        "by_type": dict(Counter(item["type"] for item in items)),
        "targets_05_3_2": CODE_TARGETS,
        "versus_targets": versus,
        "counts_meet_targets": versus["counts_meet_targets"],
    }
    planted = Counter(item.get("planted") for item in items if item["type"] in ("fix_bug", "debug_line"))
    out["planted_per_class_and_type"] = {
        cls: {kind: sum(1 for i in items if i["type"] == kind and i.get("planted") == cls) for kind in ("fix_bug", "debug_line")}
        for cls in MISCONCEPTIONS}
    debug = [i for i in items if i["type"] == "debug_line"]
    out["debug_line_no_bug_share"] = round(sum(1 for i in debug if i.get("planted") is None) / len(debug), 3) if debug else None
    out["complete_snippet_per_problem"] = dict(Counter(i["problem_id"] for i in items if i["type"] == "complete_snippet"))
    out["planted_overall"] = {str(k): v for k, v in planted.items()}
    return out


# ---------------------------------------------------------------- card

def build_card(quiz, code, log=lambda message: None):
    n = quiz["items"]
    return {
        "id": "E18",
        "title": "Is the item bank sound?",
        "slice": (f"ml/data/quiz_items.json ({n} items, re-run on the interpreter and gcc); "
                  f"code items: {code['check_line'] if code['status'] == 'checked' else code['status']}"),
        "n": n,
        "metrics": {
            "quiz_items": n,
            "pct_verified_interpreter": quiz["pct_verified_interpreter"],
            "pct_verified_gcc": quiz["pct_verified_gcc"],
            "pct_interpreter_only_because_c_undefined": pct(quiz["interpreter_only_c_undefined"], n),
            "n_manual": quiz["n_manual"],
            "verify_passed": quiz["verify_passed"],
            "collision_items": quiz["collision_items"],
            "code_items_checked": code["checked"],
            "code_check_passed": code.get("passed", False),
            "code_counts_meet_targets": code.get("counts_meet_targets", False),
        },
        "per_class": quiz["per_class_type"],
        "plots": [],
        "generated_at": ec.today(),
        "caveat": ("'Verified' means: re-running the code gives the stated answer, and re-running the believed program gives the "
                   "stated belief answer. It does not mean a student holding that mistake would choose that option; no student has "
                   "answered these items. Nobody has skimmed the DeepSeek-drafted items (06 §5). Counts are for the files on disk "
                   "at the time of the run."),
        "quiz": quiz,
        "code_items": code,
        "scope": ("The plan asks for verify_quiz.py + build_code_items.py --check over every item. The quiz half ran in full. "
                  + (f"The code-item half ran ({code['check_line']})." if code["status"] == "checked"
                     else "The code-item half could not run: " + code["why"] + ".")),
    }


def to_markdown(card):
    quiz, code = card["quiz"], card["code_items"]
    out = ["# E18: is the item bank sound?", "", f"Generated {card['generated_at']} by `python -m ml.eval.e18_bank`.", "",
           f"**Caveat.** {card['caveat']}", "", f"**Scope.** {card['scope']}", ""]
    out += ["## Quiz bank (`quiz_items.json`)", ""]
    out += md_table(["Measure", "Value"], [
        ["Items", f"{quiz['items']} ({', '.join(f'{k} {v}' for k, v in sorted(quiz['by_type'].items()))})"],
        ["`verify_quiz` over every item, both backends", "PASSED" if quiz["verify_passed"] else f"FAILED ({len(quiz['problems'])} problems)"],
        ["Verified by the interpreter", f"{quiz['pct_verified_interpreter']}% (all {quiz['items']} run, {quiz['items_with_a_problem']} with a problem)"],
        ["Also agreed by gcc", f"{quiz['agreed_by_gcc']} of {quiz['items']} = {quiz['pct_verified_gcc']}%"],
        ["Interpreter only (C leaves it undefined, so gcc is not asked)", f"{quiz['interpreter_only_c_undefined']}"],
        ["Believed programs run (behind the belief answers)", str(quiz["belief_programs_run"])],
        ["`verified` flags in the file match the fresh run", "yes" if quiz["flags_match_fresh_run"] else "NO"],
        ["Marked manual", f"{quiz['n_manual']} {quiz['manual'] or ''}".strip()],
        ["Collision items (target 12)", str(quiz["collision_items"])],
        ["Run time", f"{quiz['seconds']} s"],
    ])
    if quiz["problems"]:
        out += ["Problems found:", ""] + [f"- {line}" for line in quiz["problems"]] + [""]
    out += ["### Items per mistake and type (05 §3.1 target: 2 mcq, 2 predict_output, 1 next_state, 1 reasoning)", ""]
    kinds = list(quiz["per_class_type_target"])
    lines = [[cls] + [quiz["per_class_type"][cls][k] for k in kinds] for cls in quiz["per_class_type"]]
    lines.append(["target"] + [quiz["per_class_type_target"][k] for k in kinds])
    out += md_table(["Mistake"] + kinds, lines)
    out += [("All targets met." if not quiz["per_class_type_short"] else "Short of target: " + json.dumps(quiz["per_class_type_short"])),
            "A collision item counts toward each mistake in its `classes`.", ""]
    d = quiz["descriptive"]
    out += ["### Descriptive counts (no pass/fail)", ""]
    out += md_table(["Count", "Value"], [
        ["Position of the correct option (0 = first)", json.dumps(d["correct_option_position"]) + f" over {d['choice_items']} choice items; chi-square p = {d['correct_option_position_chi2_p']}"],
        ["Correct option is the unique longest", f"{d['correct_is_the_unique_longest_option']} of {d['choice_items']}"],
        ["Options per item", json.dumps(d["options_per_item"])],
        ["Difficulty (drafter's estimate, unpiloted)", json.dumps(d["difficulty"])],
        ["Concept", json.dumps(d["concept"])],
        ["Source", json.dumps(d["source"])],
    ])
    out += ["## Code bank (`code_items.json`)", ""]
    if code["status"] != "checked":
        out += [f"**Not built.** {code['why']}.", "", "Targets (05 §3.2), not yet checkable: "
                + "; ".join(f"{k}: {v}" for k, v in code["targets_05_3_2"].items()) + ".", ""]
    else:
        versus = code["versus_targets"]
        snip, fix, debug = versus["complete_snippet"], versus["fix_bug"], versus["debug_line"]
        share_pct = pct(debug["no_bug"], debug["no_bug_of"])
        gcc_line = next((line for line in code["output_tail"] if line.startswith("gcc ")), "")
        rows = [
            ["`build_code_items --check`", "PASSED" if code["passed"] else f"FAILED (exit {code['exit_code']})"],
            ["Check line", f"`{code['check_line']}`"],
            ["Items", f"{code['checked']} ({', '.join(f'{k} {v}' for k, v in sorted(code['by_type'].items()))})"],
            ["debug_line items with no bug (05 §3.2: about 20%)", f"{debug['no_bug']}/{debug['no_bug_of']} = {share_pct}%"],
            ["Counts meet the 05 §3.2 targets", "yes" if code["counts_meet_targets"] else "NO"],
        ]
        if gcc_line:
            rows.append(["gcc", gcc_line.removeprefix("gcc cross-check ")])
        out += md_table(["Measure", "Value"], rows)
        out += ["### Counts versus targets (05 §3.2)", ""]
        snippet_bits = [f"{snip['with_2']} problems with 2"]
        if snip["with_1"]:
            snippet_bits.append(f"{', '.join(snip['with_1'])} with 1")
        if snip["outside"]:
            snippet_bits.append("outside 1-2: " + json.dumps(snip["outside"]))
        out += md_table(["Kind", "Count", "Target", "Versus target"], [
            ["complete_snippet", f"{snip['items']} over {snip['problems']} problems ({'; '.join(snippet_bits)})", snip["target"], "met" if snip["met"] else "not met"],
            ["fix_bug", str(fix["items"]) + ("" if fix["met"] else "; " + json.dumps(fix["not_equal_to_2"])), fix["target"], "met" if fix["met"] else "not met"],
            ["debug_line", f"{debug['planted_items']} planted, plus {debug['no_bug']} with planted null", debug["target"], "met" if debug["met"] else "not met"],
        ])
        planted = code["planted_per_class_and_type"]
        lines = [[cls, planted[cls]["fix_bug"], planted[cls]["debug_line"]] for cls in planted]
        lines.append(["target", 2, 2])
        out += ["### Planted items per mistake (target: 2 fix_bug and 2 debug_line)", ""]
        out += md_table(["Mistake", "fix_bug", "debug_line"], lines)
        missing = (" This run exited 0, so none were missing." if code["passed"]
                   else " This run did not exit 0, so a missing problem may be among the failures.")
        out += ["All three 05 §3.2 targets met." if code["counts_meet_targets"] else
                "Short of a 05 §3.2 target: " + json.dumps({k: versus[k] for k in ("complete_snippet", "fix_bug", "debug_line") if not versus[k]["met"]}),
                "A problem with no `complete_snippet` is absent from the file, so only `--check` can see it." + missing,
                ""]
        out += ["Checker output:", "", "```", *code["output_tail"], "```", ""]
        if code.get("errors_tail"):
            out += ["stderr:", "", "```", *code["errors_tail"], "```", ""]
    return "\n".join(out)


def run(log=print, backends=("interp", "gcc")):
    quiz = check_quiz(backends, log)
    code = check_code_items(log)
    return build_card(quiz, code, log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["both", "interp"], default="both")
    args = parser.parse_args(argv)
    card = run(backends=("interp",) if args.backend == "interp" else ("interp", "gcc"))
    path = write_card(card, "e18", to_markdown(card))
    q = card["quiz"]
    print(f"wrote {path}")
    print(f"{q['items']} quiz items: verify {'PASSED' if q['verify_passed'] else 'FAILED'}; "
          f"interpreter {q['pct_verified_interpreter']}%, gcc {q['agreed_by_gcc']}/{q['items']} ({q['pct_verified_gcc']}%), "
          f"interp-only {q['interpreter_only_c_undefined']}, manual {q['n_manual']}, collisions {q['collision_items']}")
    print("code items:", card["code_items"]["status"], card["code_items"].get("check_line", ""))
    for line in q["problems"][:20]:
        print("PROBLEM", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
