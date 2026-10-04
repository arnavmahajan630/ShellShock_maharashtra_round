"""Code items for the practice screens (package C4, 05 section 3.2).

    python -m ml.items.build_code_items            # (re)write ml/data/code_items.json
    python -m ml.items.build_code_items --check    # re-run every item; exit 1 on any failure

Three kinds of item:

``fix_bug`` / ``debug_line``
    One verified unaugmented single-bug row of ``ml/data/dataset.jsonl`` (C3). ``op_id`` and
    ``planted`` are the row's own; ``bug_lines`` come from a line diff between the mutant and
    the reprinted correct variant it was made from. About a fifth of the ``debug_line`` items
    are a correct program with ``planted: null`` ("no bug" is a real answer).

``complete_snippet``
    A correct variant of the problem with a few ``____`` holes. Each hole sits where a mutation
    operator applies (loop start, loop test, update, index, accumulator start, base case,
    recursive argument, bound update, return value, branch test, float cast). Every wrong choice
    is run: filled in alone it must fail a test, and exactly one combination of choices passes.
    ``fix`` holds the reference fill: ``{"code": ..., "answers": {"h1": ...}}``.

Code shown to the learner (``code``, ``starter`` of ``fix_bug``) is the generator's reprint put
into K&R braces and 4-space indent (``tidy``), the same for buggy and correct programs so the
layout never gives the answer away. ``complete_snippet`` starters keep the problem author's own
text.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "code_items.json"
DATASET = ROOT / "ml" / "data" / "dataset.jsonl"
HOLE = "____"
SEED = 4
MAX_HOLES = 3

# Strings (Q13-Q15, D08) are built last and dropped first (06 section 2). Nothing is dropped now.
DROPPED_PROBLEMS: set = set()
DROPPED_CLASSES: set = set()

NULL_DEBUG_SHARE = 0.20            # share of debug_line items with planted null (05 section 3.2)
PER_CLASS = 2                      # fix_bug and debug_line per class
PARSE_STATUS = ("parse_error", "unsupported")

# ---------------------------------------------------------------- one-line reasons per operator
EXPLAIN = {
    "m01_le": "The loop test uses <= where < is needed, so the loop runs one time too many.",
    "m01_while_le": "The while test uses <= where < is needed, so the loop runs one time too many.",
    "m01_nminus1": "The loop stops one step early, so the last item is never visited.",
    "m01_start1_count": "The loop starts at 1 instead of 0, so the first item is skipped.",
    "m01_half_bound": "The loop bound is wrong, so too many or too few items are handled.",
    "m01_countdown_ge0": "The countdown test goes one step too far.",
    "m02_drop_update": "The loop variable is never changed, so the loop never ends.",
    "m02_pointer_stuck": "A pointer that should move is never changed, so the loop never ends.",
    "m02_reverse": "The loop variable moves the wrong way, so the test never becomes false.",
    "m02_wrong_var": "The wrong variable is updated, so the one in the test never changes.",
    "m03_assign_in_loop": "The accumulator is set again inside the loop, so only the last pass counts.",
    "m03_decl_in_loop": "A new copy of the variable is made on every pass, so the running total is lost.",
    "m04_cast_late": "The division happens on integers first and is converted to a float afterwards.",
    "m04_drop_cast": "Both operands are int, so the division drops the fraction.",
    "m05_drop_init_acc": "The accumulator is never given a starting value, so it begins with garbage.",
    "m05_drop_init_counter": "The counter is never given a starting value, so it begins with garbage.",
    "m05_drop_init_max": "The running best is never given a starting value, so it begins with garbage.",
    "m06_if_assign": "= assigns; == compares. The condition overwrites the variable.",
    "m07_for_semi": "A stray semicolon ends the loop, so its body runs only once, after the loop.",
    "m07_if_semi": "A stray semicolon ends the if, so the body always runs.",
    "m07_while_semi": "A stray semicolon ends the while, so its body is outside the loop.",
    "m08_first1": "Counting starts at 1, but the first cell of an array is 0.",
    "m08_index_n": "Index n is one past the last cell; the last cell is n - 1.",
    "m08_iplus1": "The index is one cell off, so the code reads past the end of the array.",
    "m08_mirror": "The mirrored index is off by one.",
    "m08_onebased": "The loop counts from 1 to n, but the cells are numbered 0 to n - 1.",
    "m10_print_bool": "The function prints the answer instead of returning it.",
    "m10_printf_noreturn": "The function prints the answer instead of returning it.",
    "m10_printf_return0": "The function prints the answer and returns 0 instead of returning it.",
    "d01_else_flag_reset": "The else branch clears the flag, so one non-match undoes an earlier match.",
    "d01_else_return": "The else branch answers after the first item, so the rest are never looked at.",
    "d02_high_mid": "high = mid never shrinks the range, so the search can stay in one place forever.",
    "d02_low_mid": "low = mid never shrinks the range, so the search can stay in one place forever.",
    "d03_drop_temp": "Without the temporary, the first assignment overwrites a value the swap still needs.",
    "d03_rotate_no_save": "The first element is not saved before the shift overwrites it.",
    "d04_drop_outer": "The outer loop does not repeat, so one pass is all the sort does.",
    "d04_outer_once": "The outer loop stops after one pass, so the array is not sorted.",
    "d05_drop_base": "There is no base case, so the recursion never stops.",
    "d05_unreachable_base": "The base case can be skipped over, so the recursion never stops.",
    "d06_grow_arg": "The recursive call moves away from the base case.",
    "d06_same_arg": "The recursive call passes the same value, so it never gets closer to the base case.",
    "d07_discard": "The result of the recursive call is thrown away.",
    "d08_str_literal": "== compares addresses, not text; use a loop over the characters.",
}
CLASS_EXPLAIN = {
    "M01": "The loop runs the wrong number of times.", "M02": "The loop never ends.",
    "M03": "The running value is reset or hidden inside the loop.", "M04": "Integer division loses the fraction.",
    "M05": "A variable is used before it has a value.", "M06": "= is used where == is meant.",
    "M07": "A stray semicolon changes what the statement controls.", "M08": "An array index is one off.",
    "M10": "The function prints a value instead of returning it.",
    "D01": "The search gives up or resets too early.", "D02": "The search range does not shrink.",
    "D03": "A swap or shift loses a value.", "D04": "The sort does not repeat enough.",
    "D05": "The recursion has no reachable base case.", "D06": "The recursive call does not get closer to the base case.",
    "D07": "The recursive result is not used.", "D08": "Text is compared with == instead of character by character.",
}


# ---------------------------------------------------------------- loading
def load_problems():
    found = {}
    for path in sorted(glob.glob(str(ROOT / "ml" / "problems" / "*" / "*.json"))):
        problem = json.loads(Path(path).read_text(encoding="utf-8"))
        found[problem["problem_id"]] = problem
    return found


def load_dataset():
    with DATASET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def op_table():
    """op_id -> labels, from the same registry the generator used."""
    from ml.generate.registry import operators
    return {op.op_id: tuple(op.labels) for op in operators()}


# ---------------------------------------------------------------- printing and diffing
def reprint(code):
    from ml.generate.ambiguity import execution_key
    return execution_key(code)


def tidy(text):
    """CGenerator layout -> K&R braces, 4 spaces, no blank lines. Same syntax tree."""
    out = []
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        body = raw.lstrip()
        indent = (len(raw) - len(body)) * 2
        if body == "{" and out and re.search(r"(\)|\belse|\bdo)$", out[-1].rstrip()):
            out[-1] = out[-1].rstrip() + " {"
            continue
        if body.startswith("else") and out and out[-1].strip() == "}":
            out[-1] = out[-1].rstrip() + " " + body
            continue
        out.append(" " * indent + body.rstrip())
    return "\n".join(out)


def same_tree(a, b):
    ka, kb = reprint(a), reprint(b)
    return ka is not None and ka == kb


def diff_lines(correct, buggy):
    """(bug_lines in buggy, changed_lines in correct), both 1-based."""
    a, b = correct.split("\n"), buggy.split("\n")
    bug, fixed = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if j2 > j1:
            bug.update(range(j1 + 1, j2 + 1))
        else:                                       # lines only deleted: point at the gap
            if j1 >= 1:
                bug.add(j1)
            if j1 < len(b) and b[j1].strip() != "}":
                bug.add(j1 + 1)
        if i2 > i1:
            fixed.update(range(i1 + 1, i2 + 1))
        elif a:                                     # lines only inserted in buggy: point at the gap
            fixed.add(min(max(i1, 1), len(a)))
    return sorted(bug), sorted(fixed)


# ---------------------------------------------------------------- running
def _tests(problem, code, backend="interp"):
    from ml.runner import run_tests
    try:
        result = run_tests(problem, code, backend=backend)
    except Exception as error:                      # an unparsable fill is "does not pass"
        return {"status": "crash", "passed": 0, "total": 0, "error": str(error)}
    tests = result.get("tests") or {}
    return {"status": result.get("status") or "", "passed": int(tests.get("passed") or 0),
            "total": int(tests.get("total") or 0)}


def passes(problem, code, backend="interp"):
    t = _tests(problem, code, backend)
    return t["status"] == "ok" and t["total"] > 0 and t["passed"] == t["total"]


def fails_cleanly(problem, code):
    """Parses and runs (or runs forever), and fails at least one test."""
    t = _tests(problem, code)
    return t["status"] not in PARSE_STATUS + ("crash",) and t["total"] > 0 and t["passed"] < t["total"]


# ---------------------------------------------------------------- explanations
def explain_text(op_id, label, bug_text, fix_text):
    base = EXPLAIN.get(op_id) or CLASS_EXPLAIN.get(label, "")
    if bug_text and fix_text and bug_text.strip() != fix_text.strip():
        return f"{base} It should read: {fix_text.strip()}".strip()
    return base


# ---------------------------------------------------------------- fix_bug and debug_line
def mutant_rows(problems, rows, labels_of):
    """Unaugmented single-bug source-A rows whose op_id is a real operator, with diff data."""
    out = []
    for row in rows:
        if row["aug"] or row["is_two_bug"] or row["source"] != "A" or not row["op_id"]:
            continue
        if row["label"] in ("CORRECT", "OTHER") or row["op_id"] not in labels_of:
            continue
        if row["label"] in DROPPED_CLASSES or row["problem_id"] in DROPPED_PROBLEMS:
            continue
        problem = problems.get(row["problem_id"])
        if not problem or row["op_id"].startswith("amb_") or row["op_id"] not in labels_of:
            continue
        index = int(row["variant"][2:]) - 1
        correct = tidy(reprint(problem["correct_variants"][index]))
        buggy = tidy(row["code"])
        if not same_tree(buggy, row["code"]) or not same_tree(correct, problem["correct_variants"][index]):
            continue
        bug_lines, fix_lines = diff_lines(correct, buggy)
        if not bug_lines or len(bug_lines) > 4:
            continue
        out.append({"row": row, "correct": correct, "buggy": buggy, "bug_lines": bug_lines,
                    "fix_lines": fix_lines})
    return out


def pick_per_class(cands, rng):
    """Four distinct rows per class, spread over problems and operators."""
    chosen = {}
    by_class = defaultdict(list)
    for cand in cands:
        by_class[cand["row"]["label"]].append(cand)
    for label, bucket in by_class.items():
        rng.shuffle(bucket)
        bucket.sort(key=lambda c: len(c["bug_lines"]))          # stable: short diffs first
        picked, used_po, used_p, used_code = [], set(), set(), set()
        for rule in ("new_problem_and_op", "new_op", "any"):
            for cand in bucket:
                if len(picked) == 2 * PER_CLASS:
                    break
                row = cand["row"]
                key = (row["problem_id"], row["op_id"])
                if cand["buggy"] in used_code:
                    continue
                if rule == "new_problem_and_op" and (row["problem_id"] in used_p or key in used_po):
                    continue
                if rule == "new_op" and key in used_po:
                    continue
                picked.append(cand)
                used_po.add(key)
                used_p.add(row["problem_id"])
                used_code.add(cand["buggy"])
        chosen[label] = picked
    return chosen


def make_fix_debug(problems, rows, labels_of, rng):
    cands = mutant_rows(problems, rows, labels_of)
    chosen = pick_per_class(cands, rng)
    fix_items, debug_items = [], []
    counters = Counter()
    for label in sorted(chosen):
        picks = chosen[label]
        for position, cand in enumerate(picks):
            row = cand["row"]
            kind = "fix_bug" if position % 2 == 0 else "debug_line"
            prefix = "fb" if kind == "fix_bug" else "dl"
            counters[(prefix, row["problem_id"], label)] += 1
            item_id = f"{prefix}_{row['problem_id']}_{label.lower()}_{counters[(prefix, row['problem_id'], label)]:02d}"
            if kind == "fix_bug":
                fix_items.append({
                    "item_id": item_id, "type": "fix_bug", "problem_id": row["problem_id"],
                    "planted": label, "op_id": row["op_id"], "starter": cand["buggy"],
                    "bug_lines": cand["bug_lines"],
                })
                continue
            buggy_lines = cand["buggy"].split("\n")
            correct_lines = cand["correct"].split("\n")
            bug_text = " ".join(buggy_lines[n - 1].strip() for n in cand["bug_lines"])
            fix_text = " ".join(correct_lines[n - 1].strip() for n in cand["fix_lines"])
            debug_items.append({
                "item_id": item_id, "type": "debug_line", "problem_id": row["problem_id"],
                "planted": label, "op_id": row["op_id"], "code": cand["buggy"],
                "bug_lines": cand["bug_lines"], "allow_no_bug": True,
                "explain": explain_text(row["op_id"], label, bug_text, fix_text),
                "fix": {"code": cand["correct"], "changed_lines": cand["fix_lines"]},
            })
    return fix_items, debug_items


def make_null_debug(problems, n_planted, rng):
    """Correct programs for "no bug": about NULL_DEBUG_SHARE of all debug_line items."""
    want = max(1, round(n_planted * NULL_DEBUG_SHARE / (1 - NULL_DEBUG_SHARE)))
    pids = sorted(pid for pid in problems if pid not in DROPPED_PROBLEMS)
    rng.shuffle(pids)
    items = []
    for pid in pids:
        if len(items) >= want:
            break
        problem = problems[pid]
        variants = list(range(len(problem["correct_variants"])))
        rng.shuffle(variants)
        for index in variants:
            code = tidy(reprint(problem["correct_variants"][index]))
            if same_tree(code, problem["correct_variants"][index]) and passes(problem, code):
                items.append({
                    "item_id": f"dl_{pid}_ok_01", "type": "debug_line", "problem_id": pid,
                    "planted": None, "op_id": None, "code": code, "bug_lines": [], "allow_no_bug": True,
                    "explain": "There is no bug: this program passes every test.", "fix": None,
                })
                break
    return items


# ---------------------------------------------------------------- complete_snippet: finding holes
def _match_paren(text, open_at):
    depth, i = 0, open_at
    while i < len(text):
        ch = text[i]
        if ch in "'\"":
            quote, i = ch, i + 1
            while i < len(text) and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top(text, start, end, sep):
    """[(a, b)] spans of text[start:end] split at top-level `sep`."""
    spans, depth, here, i = [], 0, start, start
    while i < end:
        ch = text[i]
        if ch in "'\"":
            quote, i = ch, i + 1
            while i < end and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == sep and depth == 0:
            spans.append((here, i))
            here = i + 1
        i += 1
    spans.append((here, end))
    return spans


def _strip_span(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


_REL_RE = re.compile(r"\s*([A-Za-z_][\w\[\]]*)\s*(<=|>=|==|!=|<|>)\s*(-?[\w\[\]]+)\s*")
_INT_RE = re.compile(r"-?\d+")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _header_parts(line, keyword):
    """Spans of the parenthesised part after `keyword (`, as (a, b) or None."""
    m = re.search(rf"\b{keyword}\s*\(", line)
    if not m:
        return None
    close = _match_paren(line, m.end() - 1)
    if close < 0:
        return None
    return m.end(), close


def find_holes(code):
    """Candidate holes: dicts {kind, line (0-based), a, b} with the text span of the replaced code."""
    lines = code.split("\n")
    name_m = re.match(r"\s*\w+\s+(\w+)\s*\(", lines[0])
    fname = name_m.group(1) if name_m else None
    loopvars, holes = set(), []

    def rel_op(line, a, b):
        cond = line[a:b]
        if "&&" in cond or "||" in cond:
            return None
        m = _REL_RE.fullmatch(cond)
        if not m:
            return None
        return m, a + m.start(2), a + m.end(2)

    for li, line in enumerate(lines[1:], start=1):                     # pass 1: loop variables
        for key in ("for", "while"):
            span = _header_parts(line, key)
            if not span:
                continue
            if key == "for":
                parts = _split_top(line, span[0], span[1], ";")
                if len(parts) != 3:
                    continue
                a, b = _strip_span(line, *parts[0])
                init = re.fullmatch(r"(?:int\s+)?([A-Za-z_]\w*)\s*=\s*.+", line[a:b])
                cond = _strip_span(line, *parts[1])
                found = rel_op(line, *cond)
                if init and found:
                    loopvars.add(init.group(1))
                if found and _IDENT_RE.fullmatch(found[0].group(1)):
                    loopvars.add(found[0].group(1))
            else:
                found = rel_op(line, *_strip_span(line, *span))
                if found and _IDENT_RE.fullmatch(found[0].group(1)):
                    loopvars.add(found[0].group(1))

    def add(kind, li, a, b):
        if b > a:
            holes.append({"kind": kind, "line": li, "a": a, "b": b})

    for li, line in enumerate(lines):
        if li == 0:
            continue
        for_span = _header_parts(line, "for")
        while_span = _header_parts(line, "while")
        if for_span:
            parts = _split_top(line, for_span[0], for_span[1], ";")
            if len(parts) == 3:
                a, b = _strip_span(line, *parts[0])
                init = re.fullmatch(r"((?:int\s+)?([A-Za-z_]\w*)\s*=\s*)(.+)", line[a:b])
                if init and init.group(2) in loopvars:
                    add("loop_init", li, a + len(init.group(1)), b)
                a, b = _strip_span(line, *parts[1])
                found = rel_op(line, a, b)
                if found and found[0].group(1) in loopvars:
                    add("loop_rel", li, found[1], found[2])
                a, b = _strip_span(line, *parts[2])
                if re.fullmatch(r"[A-Za-z_]\w*\s*(\+\+|--|[-+]=\s*\d+)", line[a:b]):
                    add("loop_update", li, a, b)
            continue
        if while_span:
            a, b = _strip_span(line, *while_span)
            found = rel_op(line, a, b)
            if found and found[0].group(1) in loopvars:
                add("loop_rel", li, found[1], found[2])
            continue
        decl = re.fullmatch(r"(\s*(?:int|float)\s+([A-Za-z_]\w*)\s*=\s*)([^;]+);\s*", line)
        if decl:
            var, rhs = decl.group(2), decl.group(3).strip()
            a = len(decl.group(1))
            elsewhere = [other for k, other in enumerate(lines) if k != li]
            changed = any(re.search(rf"\b{var}\s*(\+\+|--|[-+*]?=(?!=))", other) for other in elsewhere)
            if var in loopvars and re.fullmatch(r"-?\w+(\s*[-+]\s*\d+)?", rhs):
                add("loop_init", li, a, a + len(rhs))
            elif var not in loopvars and changed and (_INT_RE.fullmatch(rhs) or re.fullmatch(r"\w+\[0\]", rhs)):
                add("acc_init", li, a, a + len(rhs))
            continue
        m = re.fullmatch(r"\s*([A-Za-z_]\w*(?:\+\+|--))\s*;\s*", line)
        if m and re.match(r"\s*([A-Za-z_]\w*)", line).group(1) in loopvars:
            add("loop_update", li, m.start(1), m.end(1))
        m = re.search(r"\b[A-Za-z_]\w*\s*=\s*(mid\s*[-+]\s*1)\s*;", line)
        if m:
            add("bound_update", li, m.start(1), m.end(1))
        if "return" in line and "(float)" in line:
            r = re.search(r"\breturn\s+([^;]+);", line)
            if r:
                add("cast_expr", li, r.start(1), r.end(1))
        else:
            for r in re.finditer(r"\breturn\s+(-?\d+)\s*;", line):
                add("return_value", li, r.start(1), r.end(1))
        if fname:
            for call in re.finditer(rf"\b{fname}\s*\(", line):
                close = _match_paren(line, call.end() - 1)
                if close > 0:
                    last = _strip_span(line, *_split_top(line, call.end(), close, ",")[-1])
                    add("rec_arg", li, *last)
        if_span = _header_parts(line, "if")
        if if_span:
            a, b = _strip_span(line, *if_span)
            base = fname and re.fullmatch(r"\s*[A-Za-z_]\w*\s*(<=|<|==)\s*(-?\d+)\s*", line[a:b])
            if base and re.search(rf"\b{fname}\s*\(", "\n".join(lines[1:])):
                add("base_cond", li, a + line[a:b].rfind(base.group(2)), a + line[a:b].rfind(base.group(2)) + len(base.group(2)))
            else:
                pos = a
                for clause in re.split(r"(&&|\|\|)", line[a:b]):
                    if clause not in ("&&", "||"):
                        inner = clause.strip().strip("()")
                        start = pos + clause.find(inner) if inner else pos
                        found = rel_op(line, start, start + len(inner)) if inner else None
                        if found:
                            add("branch_rel", li, found[1], found[2])
                    pos += len(clause)
        if not (for_span or while_span):
            for ix in re.finditer(r"\b[A-Za-z_]\w*\[([A-Za-z_]\w*(?:\s*[-+]\s*\w+)?)\]", line):
                if not re.match(r"\s*(?:int|float|char)\b", line):
                    add("index", li, ix.start(1), ix.end(1))
    return holes


# ---------------------------------------------------------------- complete_snippet: wrong choices
_REL_ORDER = {
    "<": ["<=", ">", ">="], "<=": ["<", ">=", ">"], ">": [">=", "<", "<="], ">=": [">", "<=", "<"],
    "==": ["=", "!=", "<"], "!=": ["==", "<", "="],
}


def candidates(kind, text):
    """Wrong-looking replacements for `text`, most tempting first."""
    text = text.strip()
    out = []
    if kind in ("loop_rel", "branch_rel"):
        out = [r for r in _REL_ORDER.get(text, []) if kind == "branch_rel" or r != "="]
    elif kind == "loop_update":
        m = re.fullmatch(r"(\w+)(\+\+|--)", text)
        if m:
            v, op = m.groups()
            out = [f"{v}--" if op == "++" else f"{v}++", f"{v} += 2" if op == "++" else f"{v} -= 2", v]
        m = re.fullmatch(r"(\w+)\s*([-+])=\s*(\d+)", text)
        if m:
            v, sign, d = m.group(1), m.group(2), int(m.group(3))
            out = [f"{v} {'-' if sign == '+' else '+'}= {d}", f"{v} {sign}= {d + 1}"]
    elif kind == "cast_expr":
        drop = re.sub(r"\(float\)\s*", "", text)
        late = re.sub(r"\(float\)\s*(\w+)\s*/\s*(\w+)", r"(float)(\1 / \2)", text)
        late_all = re.sub(r"\(float\)\s*(\w+)\s*/\s*\(float\)\s*(\w+)", r"(float)(\1 / \2)", text)
        out = [late_all if late_all != text else late, drop]
        out = [o for o in out if o != text and "(float)" in o or o == drop]
    elif _INT_RE.fullmatch(text):
        v = int(text)
        out = [str(v + 1), str(v - 1), "0", "1", "2"]
    elif re.fullmatch(r"(\w+)\[0\]", text):
        arr = text.split("[")[0]
        out = ["0", f"{arr}[1]", f"{arr}[n - 1]"]
    elif re.fullmatch(r"\w+\[[^\]]+\]", text):
        out = []
    else:
        m = re.fullmatch(r"(\w+)\s*([-+/])\s*(\d+)", text)
        if m:
            w, op, d = m.group(1), m.group(2), int(m.group(3))
            if op == "-":
                out = [w, f"{w} - {d + 1}", f"{w} + {d}", "0", "1"]
            elif op == "+":
                out = [w, f"{w} + {d + 1}", f"{w} - {d}"]
            else:
                out = [w, f"{w} - 1", f"{w} / 2"]
        elif _IDENT_RE.fullmatch(text):
            out = [f"{text} + 1", f"{text} - 1", "0", "1"]
        else:
            m = re.fullmatch(r"(\w+)", text)
            out = []
    seen, uniq = {text}, []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq


def fill(code, holes, values):
    """Replace each hole's span (right to left so earlier spans stay valid)."""
    lines = code.split("\n")
    for hole, value in sorted(zip(holes, values), key=lambda p: (-p[0]["line"], -p[0]["a"])):
        line = lines[hole["line"]]
        lines[hole["line"]] = line[:hole["a"]] + value + line[hole["b"]:]
    return "\n".join(lines)


PRIORITY = ["loop_init", "loop_rel", "acc_init", "index", "loop_update", "bound_update", "rec_arg",
            "base_cond", "branch_rel", "cast_expr", "return_value"]
# only used when no wrong choice of a hole matches any operator predicate
STATIC_EXPOSES = {
    "loop_init": ["M01", "M08"], "loop_rel": ["M01", "M08"], "acc_init": ["M05"], "index": ["M08"],
    "loop_update": ["M02"], "bound_update": ["D02"], "rec_arg": ["D06"], "base_cond": ["D05"],
    "branch_rel": ["M06"], "cast_expr": ["M04"], "return_value": [],
}


def _spans_clash(a, b):
    return a["line"] == b["line"] and not (a["b"] <= b["a"] or b["b"] <= a["a"])


def _same_cond(a, b):
    return a["line"] == b["line"] and {a["kind"], b["kind"]} <= {"branch_rel", "base_cond"}


def build_snippet(problem, variant_index, avoid_kinds, nth):
    """One complete_snippet item from a correct variant, or None."""
    code = problem["correct_variants"][variant_index]
    if not passes(problem, code):
        return None
    all_holes = find_holes(code)
    order = sorted(PRIORITY, key=lambda k: (k in avoid_kinds, PRIORITY.index(k)))
    chosen = []                                                         # [(hole, [wrong choices])]
    for kind in order:
        if len(chosen) == MAX_HOLES:
            break
        for hole in [h for h in all_holes if h["kind"] == kind]:
            if any(_spans_clash(hole, c[0]) or _same_cond(hole, c[0]) for c in chosen):
                continue
            right = code.split("\n")[hole["line"]][hole["a"]:hole["b"]].strip()
            good = []
            for wrong in candidates(kind, right):
                if fails_cleanly(problem, fill(code, [hole], [wrong])):
                    good.append(wrong)
                if len(good) == 2:
                    break
            if good:
                chosen.append((hole, good))
                break
    if not chosen:
        return None
    chosen.sort(key=lambda c: (c[0]["line"], c[0]["a"]))
    holes = [c[0] for c in chosen]
    right = [code.split("\n")[h["line"]][h["a"]:h["b"]].strip() for h in holes]
    wrongs = [list(c[1]) for c in chosen]
    # exactly one passing combination: drop a wrong choice that takes part in another passing one
    for _ in range(12):
        extra = None
        for combo in itertools.product(*[[r] + w for r, w in zip(right, wrongs)]):
            if list(combo) != right and passes(problem, fill(code, holes, list(combo))):
                extra = combo
                break
        if extra is None:
            break
        for k, value in enumerate(extra):
            if value != right[k]:
                wrongs[k].remove(value)
                break
    keep = [k for k in range(len(holes)) if wrongs[k]]
    if not keep:
        return None
    holes_kept = [holes[k] for k in keep]
    right_kept = [right[k] for k in keep]
    wrongs_kept = [wrongs[k] for k in keep]
    # a hole left without a wrong choice stays as the author's text; spans refer to the original text
    starter = fill(code, holes, [HOLE if k in keep else right[k] for k in range(len(holes))])
    item_id = f"cs_{problem['problem_id']}_{nth:02d}"
    rng = random.Random(item_id)
    # exposes: classes of the operators whose predicate fires on a wrong fill (never on the right one)
    from ml.generate.registry import class_op_ids, predicate_holds
    labels_of = op_table()
    reference_key = reprint(code)
    firing_ref = {o for o in class_op_ids() if predicate_holds(o, reference_key)}
    exposes = set()
    for hole, wrong in zip(holes_kept, wrongs_kept):
        for w in wrong:
            key = reprint(fill(code, [hole], [w]))
            for o in class_op_ids():
                if o not in firing_ref and key and predicate_holds(o, key):
                    exposes.update(label for label in labels_of.get(o, ()) if label.startswith(("M", "D")))
    from ml.contracts.classes import MISCONCEPTIONS
    # plus what each kind of hole is for, limited to the classes this problem is built to expose
    allowed_classes = set(problem.get("exposure") or {})
    for hole in holes_kept:
        exposes.update(c for c in STATIC_EXPOSES.get(hole["kind"], []) if c in allowed_classes)
    if not exposes:
        exposes.update(allowed_classes)
    exposes = sorted(c for c in exposes if c in MISCONCEPTIONS)
    starter_lines = starter.split("\n")
    out_holes, answers = [], {}
    for number, (hole, r, w) in enumerate(zip(holes_kept, right_kept, wrongs_kept), start=1):
        choices = [r] + w
        rng.shuffle(choices)
        out_holes.append({"id": f"h{number}", "line": hole["line"] + 1, "kind": hole["kind"], "choices": choices})
        answers[f"h{number}"] = r
    assert sum(line.count(HOLE) for line in starter_lines) == len(out_holes)
    return {
        "item_id": item_id, "type": "complete_snippet", "problem_id": problem["problem_id"],
        "starter": starter, "holes": out_holes, "exposes": exposes,
        "fix": {"code": code, "answers": answers},
    }


def make_snippets(problems):
    items = []
    for pid in sorted(problems):
        if pid in DROPPED_PROBLEMS:
            continue
        problem = problems[pid]
        used_kinds, made = set(), []
        for index in range(len(problem["correct_variants"])):
            if len(made) == 2:
                break
            item = build_snippet(problem, index, used_kinds, len(made) + 1)
            if item is None:
                continue
            if any(item["starter"] == other["starter"] for other in made):
                continue
            made.append(item)
            used_kinds.update(h["kind"] for h in item["holes"])
        items.extend(made)
    return items


# ---------------------------------------------------------------- build
def build():
    rng = random.Random(SEED)
    problems = load_problems()
    rows = load_dataset()
    labels_of = op_table()
    snippets = make_snippets(problems)
    fix_items, debug_items = make_fix_debug(problems, rows, labels_of, rng)
    nulls = make_null_debug(problems, len(debug_items), rng)
    items = snippets + sorted(fix_items, key=lambda i: i["item_id"]) + sorted(debug_items + nulls, key=lambda i: i["item_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return items


# ---------------------------------------------------------------- check
def _summary(items):
    kinds = Counter(i["type"] for i in items)
    print("items " + " ".join(f"{k}={kinds[k]}" for k in ("complete_snippet", "fix_bug", "debug_line")))
    from ml.contracts.classes import MISCONCEPTIONS
    for kind in ("fix_bug", "debug_line"):
        per = Counter(i["planted"] for i in items if i["type"] == kind and i.get("planted"))
        print(f"{kind} per class " + " ".join(f"{c}={per[c]}" for c in MISCONCEPTIONS))
    nulls = sum(1 for i in items if i["type"] == "debug_line" and i.get("planted") is None)
    total = sum(1 for i in items if i["type"] == "debug_line")
    print(f"debug_line no-bug {nulls}/{total}")
    per_problem = Counter(i["problem_id"] for i in items if i["type"] == "complete_snippet")
    print("complete_snippet per problem " + " ".join(f"{p}={n}" for p, n in sorted(per_problem.items())))


def check(verbose=True):
    from pydantic import ValidationError
    from ml.contracts.classes import MISCONCEPTIONS
    from ml.contracts.schemas import CodeItem, PublicCodeItem
    from ml.generate.registry import predicate_holds
    from ml.runner import available

    problems = load_problems()
    labels_of = op_table()
    errors = []

    def bad(item, message):
        errors.append(f"{item.get('item_id')}: {message}")

    if not OUT.is_file():
        print(f"FAIL: {OUT} does not exist (run without --check first)")
        return 1
    items = json.loads(OUT.read_text(encoding="utf-8"))
    use_gcc = available("gcc")
    ids = Counter(i.get("item_id") for i in items)
    for item_id, n in ids.items():
        if n > 1:
            errors.append(f"{item_id}: duplicate item_id")

    provenance = defaultdict(set)
    wanted = {(i["problem_id"], i["op_id"]) for i in items if i.get("op_id")}
    for row in load_dataset():
        if row["aug"] or row["is_two_bug"] or (row["problem_id"], row["op_id"]) not in wanted:
            continue
        key = reprint(row["code"])
        if key:
            provenance[(row["problem_id"], row["op_id"])].add(key)

    for item in items:
        try:
            CodeItem.model_validate(item)
            PublicCodeItem.model_validate({k: item[k] for k in PublicCodeItem.model_fields if k in item})
        except ValidationError as error:
            bad(item, f"schema: {error.errors()[0]['msg']}")
            continue
        problem = problems.get(item["problem_id"])
        if problem is None:
            bad(item, "unknown problem_id")
            continue
        kind = item["type"]
        if kind in ("fix_bug", "debug_line"):
            code = item["starter"] if kind == "fix_bug" else item["code"]
            planted, op_id = item.get("planted"), item.get("op_id")
            if kind == "debug_line" and planted is None:
                if op_id or item["bug_lines"] or item.get("fix") is not None:
                    bad(item, "a no-bug item must have no op_id, bug_lines or fix")
                if not passes(problem, code):
                    bad(item, "no-bug code does not pass every test")
                if not any(same_tree(code, v) for v in problem["correct_variants"]):
                    bad(item, "no-bug code is not a correct variant of the problem")
                continue
            if planted not in MISCONCEPTIONS:
                bad(item, f"planted {planted!r} is not one of the 17 classes")
            if op_id not in labels_of:
                bad(item, f"op_id {op_id!r} is not a real operator")
                continue
            if planted not in labels_of[op_id]:
                bad(item, f"{op_id} does not produce {planted}")
            if not fails_cleanly(problem, code):
                bad(item, "buggy code does not fail a test")
            if not predicate_holds(op_id, reprint(code)):
                bad(item, f"predicate {op_id} does not hold on the code")
            if reprint(code) not in provenance.get((item["problem_id"], op_id), set()):
                bad(item, "code is not an unaugmented dataset row for this problem and op_id")
            lines = code.split("\n")
            if not item["bug_lines"] or any(not 1 <= n <= len(lines) for n in item["bug_lines"]):
                bad(item, "bug_lines are missing or outside the code")
            correct = item["fix"]["code"] if kind == "debug_line" and item.get("fix") else None
            if kind == "debug_line":
                if correct is None:
                    bad(item, "debug_line with a bug needs fix.code")
                    continue
                if not item.get("explain"):
                    bad(item, "debug_line needs explain")
                if not item["fix"].get("changed_lines"):
                    bad(item, "fix.changed_lines is empty")
                if not item.get("allow_no_bug"):
                    bad(item, "allow_no_bug must be true")
                for backend in ("interp", "gcc") if use_gcc else ("interp",):
                    if not passes(problem, correct, backend):
                        bad(item, f"fix.code does not pass every test on {backend}")
                if diff_lines(correct, code)[0] != item["bug_lines"]:
                    bad(item, "bug_lines do not match the line diff against fix.code")
                if not any(same_tree(correct, v) for v in problem["correct_variants"]):
                    bad(item, "fix.code is not a correct variant")
            else:
                # fix_bug has no stored fix: some correct variant must differ from it on the bug lines only
                variants = [tidy(reprint(v)) for v in problem["correct_variants"]]
                if not any(diff_lines(v, code)[0] == item["bug_lines"] for v in variants):
                    bad(item, "bug_lines do not match the line diff against any correct variant")
        else:
            starter = item["starter"]
            holes = item["holes"]
            if starter.count(HOLE) != len(holes) or not holes:
                bad(item, "number of ____ differs from number of holes")
                continue
            fix = item.get("fix") or {}
            answers, correct = fix.get("answers") or {}, fix.get("code")
            if not correct or set(answers) != {h["id"] for h in holes}:
                bad(item, "fix.code / fix.answers missing")
                continue
            starter_lines = starter.split("\n")
            for hole in holes:
                if not 1 <= hole["line"] <= len(starter_lines) or HOLE not in starter_lines[hole["line"] - 1]:
                    bad(item, f"hole {hole['id']} line {hole['line']} has no ____")
                if answers[hole["id"]] not in hole["choices"] or len(set(hole["choices"])) != len(hole["choices"]):
                    bad(item, f"hole {hole['id']}: answer missing from choices or choices repeat")
                if len(hole["choices"]) < 2:
                    bad(item, f"hole {hole['id']} has no wrong choice")
            order = [h["line"] for h in holes]
            if order != sorted(order):
                bad(item, "holes are not in reading order")
            pieces = starter.split(HOLE)
            def filled(values):
                return "".join(p + (values[k] if k < len(values) else "") for k, p in enumerate(pieces))
            right = [answers[h["id"]] for h in holes]
            if filled(right) != correct:
                bad(item, "starter with the answers filled in is not fix.code")
            if not any(same_tree(correct, v) for v in problem["correct_variants"]):
                bad(item, "fix.code is not a correct variant")
            for backend in ("interp", "gcc") if use_gcc else ("interp",):
                if not passes(problem, correct, backend):
                    bad(item, f"fix.code does not pass every test on {backend}")
            passing = []
            for combo in itertools.product(*[h["choices"] for h in holes]):
                if passes(problem, filled(list(combo))):
                    passing.append(combo)
            if passing != [tuple(right)]:
                bad(item, f"{len(passing)} choice combinations pass (exactly the intended one should)")
            for k, hole in enumerate(holes):
                for wrong in hole["choices"]:
                    if wrong == right[k]:
                        continue
                    values = list(right)
                    values[k] = wrong
                    if not fails_cleanly(problem, filled(values)):
                        bad(item, f"wrong choice {wrong!r} of {hole['id']} does not fail a test")
            unknown = [c for c in item["exposes"] if c not in MISCONCEPTIONS]
            if unknown or not item["exposes"]:
                bad(item, f"exposes {item['exposes']} is empty or has unknown classes")

    # targets (05 section 3.2)
    for pid in sorted(problems):
        if pid in DROPPED_PROBLEMS:
            continue
        n = sum(1 for i in items if i["type"] == "complete_snippet" and i["problem_id"] == pid)
        if not 1 <= n <= 2:
            errors.append(f"{pid}: {n} complete_snippet items (target 1-2)")
    for cls in MISCONCEPTIONS:
        if cls in DROPPED_CLASSES:
            continue
        for kind in ("fix_bug", "debug_line"):
            n = sum(1 for i in items if i["type"] == kind and i.get("planted") == cls)
            if n != PER_CLASS:
                errors.append(f"{cls}: {n} {kind} items (target {PER_CLASS})")
    debug = [i for i in items if i["type"] == "debug_line"]
    share = sum(1 for i in debug if i.get("planted") is None) / max(1, len(debug))
    if not 0.12 <= share <= 0.25:
        errors.append(f"debug_line no-bug share {share:.2f} is not about 0.20")

    if verbose:
        _summary(items)
        print(f"gcc cross-check {'on' if use_gcc else 'off (gcc not available)'}")
    if errors:
        print(f"CHECK FAILED: {len(errors)} problem(s)")
        for line in errors[:60]:
            print("  " + line)
        return 1
    print(f"CHECK PASSED: {len(items)} items")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify ml/data/code_items.json and exit 0/1")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    items = build()
    _summary(items)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(items)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
