"""Builds ml/data/quiz_items.json from drafts by running them (ml_plan/05 §3.1).

    python -m ml.items.build_quiz

Input:
    ml/items/quiz_drafts.json    candidates written by DeepSeek (draft_quiz_llm.py)
    ml/items/quiz_manual.json    candidates written by hand (the collision items and gap fillers)
    ml/items/quiz_explain.json   explanations from the second DeepSeek pass, if it has been run
    ml/items/quiz_rejected.json  drafts that ran fine but were thrown out by a human reader, with the reason

A candidate has the code the learner sees and, per mistake, the program a learner with that
mistake thinks they are reading. Nothing a candidate claims about answers is used: the correct
answer and each belief answer come from running the fragments on the interpreter, and a
candidate is kept only if gcc gives the same answers wherever C defines them.

Output:
    ml/data/quiz_items.json      the items (schemas.QuizItem)
    ml/data/quiz_checks.json     how each item is checked again by verify_quiz.py
"""
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS
from ml.contracts.schemas import QuizItem
from ml.items import snippet
from ml.items.draft_quiz_llm import DRAFTS, EXPLAIN, GUIDE, explain_key

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MANUAL = HERE / "quiz_manual.json"
REJECTED = HERE / "quiz_rejected.json"       # {draft_id: why}, drafts thrown out after reading them

WANTED = {"predict_output": 2, "mcq": 2, "next_state": 1, "reasoning": 1}     # per class (05 §3.1)
SHORT = {"predict_output": "po", "mcq": "mcq", "next_state": "ns", "reasoning": "rs"}
MAX_LINES, MAX_WIDTH, MAX_ANSWER = 14, 70, 24
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}

# An event the interpreter must report when the shown code runs: proof that the code really
# contains the mistake the item is about.
MUST_SEE = {"M04": "intdiv", "M05": "uninit_read", "M06": "assign_in_cond", "M07": "empty_body",
            "D07": "discarded_call_value", "D08": "str_literal_compare"}
MUST_NOT_END = {"M02", "D05", "D06"}


def flat(code):
    return " ".join(code.split())


def _clean(code):
    return code.replace("\r\n", "\n").replace("\t", "    ").strip("\n")


def normalise(draft):
    """A draft in one shape, or (None, reason)."""
    kind = draft.get("type")
    if kind not in ("predict_output", "mcq", "next_state"):
        return None, "unknown type"
    classes = draft.get("classes") or [draft.get("class")]
    belief_code = draft.get("belief_code")
    if isinstance(belief_code, str):
        belief_code = {classes[0]: belief_code}
    if not isinstance(draft.get("code"), str) or not isinstance(belief_code, dict):
        return None, "no code"
    out = {
        "draft_id": draft["draft_id"], "type": kind, "classes": classes, "manual": "classes" in draft,
        "concept": draft.get("concept") or GUIDE[classes[0]]["concept"],
        "code": _clean(draft["code"]), "belief_code": {c: _clean(v) for c, v in belief_code.items()},
        "says": draft.get("says"), "tail": "", "state_query": None, "belief_query": None,
        "distractors": [str(d).replace("\\n", "").strip() for d in draft.get("distractors") or []],
        "difficulty": draft.get("difficulty") if draft.get("difficulty") in (1, 2, 3) else 1,
    }
    lines = out["code"].split("\n")
    if len(lines) > MAX_LINES or max(len(line) for line in lines) > MAX_WIDTH:
        return None, "code too long"
    if set(belief_code) != set(classes) or any(c not in MISCONCEPTIONS for c in classes):
        return None, "classes and belief_code do not match"
    if kind == "predict_output":
        out["kind"], out["question"] = draft.get("kind") or "printed", "What does this print?"
        if "printf" not in out["code"]:
            return None, "predict_output code prints nothing"
    elif kind == "mcq":
        out["kind"] = draft.get("kind") or "tail"
        out["question"] = " ".join(str(draft.get("question") or "").split())
        out["tail"] = str(draft.get("tail") or "").replace("\\n", "").strip()
        if not out["question"]:
            return None, "mcq without a question"
        if out["kind"] == "tail" and not re.fullmatch(r"(?:[^\n;]*;\n)?printf\([^\n]*\);", out["tail"]):
            return None, "mcq tail is not one printf"
        if "printf" in out["code"] and "M10" not in classes:     # M10 is the mistake of printing a result
            return None, "mcq code prints"
        if "printf" in out["code"] and re.search(r"[A-Za-z_]\w*\s*\(", out["tail"].replace("printf(", "", 1)):
            return None, "mcq tail calls a function that prints, so the value cannot be told apart"
    else:
        out["kind"] = "state"
        var, line, hit = draft.get("var"), draft.get("after_line"), draft.get("hit")
        other = draft.get("belief_after_line", line)
        if not (isinstance(var, str) and re.fullmatch(r"[A-Za-z_]\w*", var) and isinstance(hit, int)
                and 1 <= hit <= 6 and isinstance(line, int) and isinstance(other, int)):
            return None, "next_state fields missing"
        believed = next(iter(out["belief_code"].values())).split("\n")
        if not (1 <= line <= len(lines) and 1 <= other <= len(believed)):
            return None, "next_state line outside the code"
        if re.search(r"\b(for|while|if|else)\b", lines[line - 1]) or not lines[line - 1].rstrip().endswith(";"):
            return None, "next_state line is not a plain statement"
        out["state_query"] = {"var": var, "after_line": line, "hit": hit}
        out["belief_query"] = {"var": var, "after_line": other, "hit": hit}
        out["question"] = f"What is {var} right after line {line} has run for the {ORDINAL[hit]} time?"
    for cls, believed in out["belief_code"].items():
        if difflib.SequenceMatcher(None, flat(out["code"]), flat(believed)).ratio() < 0.5:
            return None, "belief_code is a different program"
    return out, None


def run_all(candidates):
    """Adds "correct", "belief", "gcc" and "problem" to each candidate."""
    jobs, owners = [], []
    for index, cand in enumerate(candidates):
        common = {"tail": cand["tail"], "kind": cand["kind"], "says": cand["says"]}
        jobs.append({"fragment": cand["code"], "state_query": cand["state_query"], **common})
        owners.append((index, None))
        for cls, believed in cand["belief_code"].items():
            jobs.append({"fragment": believed, "state_query": cand["belief_query"], **common})
            owners.append((index, cls))
    results = snippet.check_many(jobs)
    for cand in candidates:
        cand.update(correct=None, belief={}, gcc=True, problem=None)
    for (index, cls), got in zip(owners, results):
        cand = candidates[index]
        what = "the shown code" if cls is None else f"the believed program of {cls}"
        if got["text"] is None:
            cand["problem"] = cand["problem"] or f"{what} does not run ({got['error']})"
        elif got["error"]:
            cand["problem"] = cand["problem"] or f"{what}: {got['error']}"
        elif cls is None:
            cand["correct"], cand["gcc"] = got["text"], got["gcc"] is not None
            cand["events"], cand["status"] = got["events"], got["status"]
        else:
            cand["belief"][cls] = got["text"]
    for cand in candidates:
        if cand["problem"]:
            continue
        answers = [cand["correct"], *cand["belief"].values()]
        if any(not a or len(a) > MAX_ANSWER or "\n" in a for a in answers):
            cand["problem"] = "an answer is empty or too long"
        elif any(value == cand["correct"] for value in cand["belief"].values()):
            cand["problem"] = "the believed program gives the same answer as the shown code"
        elif cand["manual"] and len(set(cand["belief"].values())) != 1:
            cand["problem"] = f"collision item whose classes disagree: {cand['belief']}"
        else:
            cand["problem"] = _mistake_missing(cand)
    return candidates


def _mistake_missing(cand):
    """A reason when the shown code does not show the mistake it is meant to (drafts only)."""
    if cand["manual"]:
        return None
    cls = cand["classes"][0]
    if cls in MUST_SEE and MUST_SEE[cls] not in cand["events"]:
        return f"the shown code never triggers {MUST_SEE[cls]}"
    if cls in MUST_NOT_END and cand["status"] != "timeout":
        return "the shown code ends, but this mistake should make it run forever"
    return None


def _same(a, b):
    """True when two option texts are the same text or the same number."""
    if a.strip().lower() == b.strip().lower():
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def options_for(cand):
    """3 or 4 options in a fixed shuffled order, or None."""
    options = [cand["correct"]]
    for value in cand["belief"].values():
        if value not in options:
            options.append(value)
    for value in cand["distractors"]:           # "4" next to "4.0" would be a trick, not a distractor
        if value and len(value) <= MAX_ANSWER and len(options) < 4 and not any(_same(value, o) for o in options):
            options.append(value)
    if len(options) < 3 and re.fullmatch(r"-?\d+", cand["correct"]):
        for guess in (int(cand["correct"]) + 1, int(cand["correct"]) - 1, 0):
            if str(guess) not in options and len(options) < 3:
                options.append(str(guess))
    if len(options) < 3:
        return None
    random.Random(hashlib.md5(cand["code"].encode("utf-8")).hexdigest()).shuffle(options)
    return options


def reasoning_question(correct, believed):
    if correct == "never stops":
        return (f'This never stops. A classmate expected it to print "{believed}". '
                f"In one sentence: why does it not stop?")
    if correct == "unpredictable":
        return (f'What this prints cannot be predicted. A classmate expected "{believed}". '
                f"In one sentence: why can it not be predicted?")
    return f'This prints "{correct}". A classmate expected "{believed}". In one sentence: why?'


def _too_close(code, chosen):
    return any(difflib.SequenceMatcher(None, flat(code), flat(other)).ratio() > 0.8 for other in chosen)


def choose(candidates, taken_code):
    """Pick the items: per class 2 predict_output, 2 mcq, 1 next_state, 1 reasoning; plus every collision item."""
    picked, gaps = [], []
    used = set(taken_code)
    pools = {}
    for cand in candidates:
        if cand["problem"] or flat(cand["code"]) in used:
            continue
        used.add(flat(cand["code"]))
        key = ("collision", cand["type"]) if len(cand["classes"]) > 1 else (cand["classes"][0], cand["type"])
        pools.setdefault(key, []).append(cand)

    for kind in ("predict_output", "mcq"):
        for cand in pools.get(("collision", kind), []):
            picked.append((kind, cand))
    for cls in MISCONCEPTIONS:
        chosen = []
        for kind in ("predict_output", "mcq", "next_state", "reasoning"):
            pool = pools.get((cls, "predict_output" if kind == "reasoning" else kind), [])
            got = 0
            for cand in pool:
                if got == WANTED[kind]:
                    break
                if cand.get("used") or _too_close(cand["code"], chosen) or (kind != "reasoning" and not options_for(cand)):
                    continue
                cand["used"] = True
                chosen.append(cand["code"])
                picked.append((kind, cand))
                got += 1
            if got < WANTED[kind]:
                gaps.append(f"{cls}: {got} of {WANTED[kind]} {kind}")
    return picked, gaps


def to_item(kind, cand, number, explanations):
    classes = cand["classes"]
    tag = "".join(c.lower() for c in classes)
    item = {
        "item_id": f"qz_{tag}_{SHORT[kind]}_{number:02d}", "type": kind, "concept": cand["concept"],
        "classes": classes, "code": cand["code"], "question": cand["question"], "options": [], "correct": None,
        "belief": {}, "state_query": None, "expected": None, "explain": "", "difficulty": cand["difficulty"],
        "verified": {"interp": True, "gcc": cand["gcc"], "manual": False},
    }
    check = {"draft": cand["draft_id"], "kind": cand["kind"], "belief_code": cand["belief_code"]}
    for key in ("tail", "says", "belief_query"):
        if cand[key]:
            check[key] = cand[key]
    if kind == "reasoning":
        believed = cand["belief"][classes[0]]
        item.update(question=reasoning_question(cand["correct"], believed), expected="CORRECT_REASON")
        check.update(expect=cand["correct"], belief_answers=cand["belief"])
    else:
        item.update(options=options_for(cand), correct=cand["correct"], belief=cand["belief"],
                    state_query=cand["state_query"])
    item["explain"] = explanations.get(explain_key(item)) or f"{CLASS_INFO[classes[0]]['subtitle']}."
    QuizItem.model_validate(item)
    return item, check


def main():
    drafts = json.loads(DRAFTS.read_text(encoding="utf-8")) if DRAFTS.exists() else []
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))
    explanations = json.loads(EXPLAIN.read_text(encoding="utf-8")) if EXPLAIN.exists() else {}

    rejected = json.loads(REJECTED.read_text(encoding="utf-8")) if REJECTED.exists() else {}

    candidates, dropped = [], Counter()
    for draft in manual + drafts:               # hand-written first, so they win a tie on equal code
        if draft.get("draft_id") in rejected:
            dropped["rejected by a human reader (quiz_rejected.json)"] += 1
            continue
        cand, reason = normalise(draft)
        if cand is None:
            dropped[reason] += 1
        else:
            candidates.append(cand)
    run_all(candidates)
    for cand in candidates:
        if cand["problem"]:
            dropped[re.sub(r"\(.*", "", cand["problem"]).strip()[:70]] += 1
            if cand["manual"]:
                print(f"HAND-WRITTEN ITEM FAILED {cand['draft_id']}: {cand['problem']}")

    taken = []
    for name in ("probes.json", "items.json", "exam_items.json"):
        if (DATA / name).exists():
            taken += [flat(item["code"]) for item in json.loads((DATA / name).read_text(encoding="utf-8"))]
    picked, gaps = choose(candidates, taken)

    items, checks, counter = [], {}, Counter()
    for kind, cand in picked:
        key = ("".join(cand["classes"]), kind)
        counter[key] += 1
        item, check = to_item(kind, cand, counter[key], explanations)
        items.append(item)
        checks[item["item_id"]] = check
    items.sort(key=lambda item: item["item_id"])
    (DATA / "quiz_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")
    (DATA / "quiz_checks.json").write_text(json.dumps(dict(sorted(checks.items())), indent=2, ensure_ascii=False)
                                           + "\n", encoding="utf-8", newline="\n")

    sound = sum(1 for cand in candidates if not cand["problem"])
    print(f"{len(manual) + len(drafts)} candidates ({len(manual)} hand-written), {sound} ran and held up")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:3d}: {reason}")
    print(f"{len(items)} items written; with explanations from the second pass: "
          f"{sum(1 for item in items if explain_key(item) in explanations)}")
    for gap in gaps:
        print("GAP", gap)
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
