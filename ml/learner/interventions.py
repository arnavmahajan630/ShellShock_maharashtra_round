"""Intervention policy and package builder (ml_plan/03 §7.1, §11.2). Package R2.

``build_package`` is the one call the /intervene route needs:

    build_package(problem, code, cls, tried=("trace_timeline",), modality=None) -> dict

The dict has the shape of 03 §11.2 (``class``, ``modality``, ``next_modalities``, ``copy``,
``question``, ``timeline``, one panel for the class, ``counterexample``, ``fix``). The
route adds ``model_version`` and ``latency_ms``.

Policy (03 §7.1): each class has an ordered list of modalities. The next one is the first
that was not tried yet for this (learner, class); a tried one is never offered again. A
modality is either a panel (``trace_timeline``, ``memory_strip``, ``window_strip``,
``value_meter``, ``call_stack``) or ``counterexample`` / ``minimal_fix``. Every package carries
all three parts (the class's panel, the counterexample and the fix); the modality only
says which one is primary and which question is asked.

When every modality of the class has been tried, ``minimal_fix`` is offered as a last
resort (if it was not tried), and after that the first modality comes back with
``exhausted: true`` so the route can see it is a repeat.

The log of (learner, class, modality, outcome) for a future bandit is the knowledge
store's job (D2 / S1); this module keeps no state.
"""
from __future__ import annotations

import difflib
import re

from pycparser import c_generator, c_parser

from ml.c_interp.preprocess import preprocess
from ml.learner import counterexample as _counterexample
from ml.learner import fixer as _fixer
from ml.learner import timeline as _timeline

MODALITIES = ["trace_timeline", "memory_strip", "window_strip", "value_meter", "call_stack",
              "counterexample", "minimal_fix"]
PANELS = ["trace_timeline", "memory_strip", "window_strip", "value_meter", "call_stack"]

_LOOP = ["trace_timeline", "counterexample", "minimal_fix"]

# 03 §7.1, one row per class. M07 has a second row for an `if`.
POLICY = {
    "M01": _LOOP,
    "M02": _LOOP,
    "M03": ["trace_timeline", "minimal_fix"],
    "M04": ["value_meter", "counterexample"],
    "M05": ["trace_timeline", "minimal_fix"],
    "M06": ["trace_timeline", "minimal_fix"],
    "M07": _LOOP,
    "M08": ["memory_strip", "counterexample", "minimal_fix"],
    "M10": ["trace_timeline", "minimal_fix"],
    "D01": _LOOP,
    "D02": ["window_strip", "minimal_fix"],
    "D03": ["memory_strip", "counterexample", "minimal_fix"],
    "D04": ["memory_strip", "counterexample"],
    "D05": ["call_stack", "counterexample", "minimal_fix"],
    "D06": ["call_stack", "counterexample", "minimal_fix"],
    "D07": ["call_stack", "minimal_fix"],
    "D08": ["value_meter", "minimal_fix"],
}
POLICY_M07_IF = ["trace_timeline", "minimal_fix"]
POLICY_OTHER = ["trace_timeline", "minimal_fix"]       # novel, OTHER and anything unlisted

# The panel each class shows (also the primary panel when its modality is chosen).
PANEL_OF = {
    "M04": "value_meter", "D08": "value_meter",
    "M08": "memory_strip", "D03": "memory_strip", "D04": "memory_strip",
    "D02": "window_strip",
    "D05": "call_stack", "D06": "call_stack", "D07": "call_stack",
}

COPY = {
    "M01": ["A loop runs while its condition is true. Check where it stops, not where it starts.",
            "`i < n` stops before n; `i <= n` runs one more time."],
    "M02": ["A loop only ends if something inside it moves toward the exit.",
            "Nothing changes the loop variable, so the condition stays true forever."],
    "M03": ["A variable set inside the loop is set again on every pass.",
            "Put the starting value before the loop so the total can keep growing."],
    "M04": ["Dividing two ints gives an int: the decimals are dropped before anything else happens.",
            "Make one side a float (`2.0`, or `(float)a`) to keep them."],
    "M05": ["A local variable has no value until you give it one.",
            "Reading it first returns whatever was in memory: static."],
    "M06": ["`=` stores a value; `==` asks if two values are equal.",
            "Inside an `if`, `=` overwrites the variable and is true whenever the value is not 0."],
    "M07": ["A `;` right after `if (...)`, `for (...)` or `while (...)` is an empty body.",
            "The next line then runs on its own, whatever the condition said."],
    "M08": ["Arrays start at cell 0.",
            "An array of n cells ends at cell n−1; cell n is the void."],
    "M10": ["`printf` shows a value on screen; `return` hands it back to the caller.",
            "The caller only sees what is returned."],
    "D01": ["Searching is not finished at the first miss.",
            "Only give up after every cell has been checked."],
    "D02": ["After checking `mid`, the window has to leave `mid` behind.",
            "`low = mid + 1` or `high = mid - 1`; with `low = mid` the same window can come back."],
    "D03": ["Two assignments run one after the other, not together.",
            "The first one overwrites a value the second still needs. Keep it in a temp first."],
    "D04": ["One pass moves only the biggest value into place.",
            "Repeat the pass until the whole array is in order."],
    "D05": ["Every recursion needs a case that stops without calling itself.",
            "Without one, the calls pile up until the ship runs out of room."],
    "D06": ["Each recursive call must be closer to the base case than the one before.",
            "If the argument stays the same or grows, the base case is never reached."],
    "D07": ["A recursive call returns a value. Use it, or it is lost.",
            "Combine it with this call's own work, then return the result."],
    "D08": ["In C, `==` on text compares where the text lives, not the letters.",
            "Compare one character at a time, with single quotes: `s[i] == 'a'`."],
}
COPY_OTHER = ["This one doesn't match a known pattern.",
              "Compare what your code does, step by step, with what the mission needs."]


def policy(cls, empty_body_kind=None):
    """The ordered modalities for a class. ``empty_body_kind`` (``"if"`` / ``"for"`` / ``"while"``) picks M07's row."""
    if cls == "M07" and empty_body_kind == "if":
        return list(POLICY_M07_IF)
    return list(POLICY.get(cls, POLICY_OTHER))


def choose_modality(cls, tried=(), requested=None, empty_body_kind=None):
    """``(modality, next_modalities, exhausted)``.

    ``requested`` (a valid modality) wins over the policy. Otherwise the first modality of
    the class's list that is not in ``tried``. ``next_modalities`` are the untried ones that
    follow it. See the module docstring for what happens when all are tried.
    """
    tried = list(tried or [])
    order = policy(cls, empty_body_kind)
    if requested in MODALITIES:
        rest = [m for m in order if m not in tried and m != requested]
        return requested, rest, False
    pending = [m for m in order if m not in tried]
    if not pending and "minimal_fix" not in tried:
        pending = ["minimal_fix"]
    if not pending:
        return order[0], order[1:], True
    return pending[0], pending[1:], False


def _empty_body_kind(trace):
    for event in (trace or {}).get("events") or []:
        if event.get("type") == "empty_body":
            return event.get("kind")
    return None


_GEN = c_generator.CGenerator()
_TOKEN = re.compile(
    r"[A-Za-z_]\w*|\d+\.?\d*|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|==|!=|<=|>=|&&|\|\||\+\+|--|[-+*/%]=|\S")


def _reprint(code):
    try:
        return _GEN.visit(c_parser.CParser().parse(preprocess(code)))
    except Exception:
        return None


def _fix_lines(code, fixed):
    """Lines of ``fixed`` that differ from the learner's code printed the same way.

    The fixer returns reprinted code, so a plain line diff against the learner's own text
    marks every line. Printing both with the same generator leaves only the edit.
    A deleted line points at the line that took its place. None if the code does not parse.
    """
    base = _reprint(code)
    if base is None:
        return None
    new = fixed.splitlines()
    lines = set()
    matcher = difflib.SequenceMatcher(a=base.splitlines(), b=new, autojunk=False)
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            lines.add(min(max(j1 + 1, 1), max(len(new), 1)))
        else:
            lines.update(range(j1 + 1, j2 + 1))
    return sorted(lines)


def _tokens(text):
    out = []
    for match in _TOKEN.finditer(text):
        if match.group() in ("{", "}"):
            continue
        out.append((match.group(), text.count("\n", 0, match.start()) + 1))
    return out


def learner_lines(code, fixed):
    """Lines of the *learner's* code that the fix touches (token diff, braces ignored).

    An inserted piece is attributed to the line of the token before it. [] if nothing differs.
    """
    try:
        old = _tokens(preprocess(code))
    except Exception:
        old = _tokens(code)
    new = _tokens(fixed)
    matcher = difflib.SequenceMatcher(a=[t for t, _ in old], b=[t for t, _ in new], autojunk=False)
    lines = set()
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            if old:
                lines.add(old[max(i1 - 1, 0)][1])
        else:
            lines.update(line for _, line in old[i1:i2])
    return sorted(lines)


def _fix_dict(fix, code=None):
    lines = list(fix.get("changed_lines") or [])
    kind = fix.get("kind", "reference")
    if kind == "minimal" and code is not None:
        precise = _fix_lines(code, fix.get("code", ""))
        if precise:
            lines = precise
    return {
        "kind": kind,
        "code": fix.get("code", ""),
        "changed_lines": lines,
        "rule": fix.get("rule"),
        "verified": bool(fix.get("verified", False)),
    }


def _meter_kind(cls, ctx):
    if cls == "D08":
        return "char_vs_string"
    if cls == "M04":
        return "division"
    kinds = {e.get("type") for e in ctx.events("learner")}
    if kinds & {"str_literal_compare", "array_compare"}:
        return "char_vs_string"
    return "division"


def _panel(name, cls, ctx):
    if name == "memory_strip":
        return "memory_strip", _timeline.build_memory_strip(ctx)
    if name == "window_strip":
        return "window_strip", _timeline.build_window_strip(ctx)
    if name == "call_stack":
        return "call_stack", _timeline.build_call_stack(ctx)
    if name == "value_meter":
        return "value_meter", _timeline.build_value_meter(_meter_kind(cls, ctx), ctx)
    return None, None


def _question(modality, cls, ctx, cex, fix, code=""):
    if modality == "counterexample" and cex:
        shown = ", ".join(f"{k} = {v}" for k, v in (cex.get("input") or {}).items())
        return {"prompt": f"What should this return for {shown}?", "answer": cex.get("intended")}
    if modality == "minimal_fix" and fix and fix.get("kind") == "minimal":
        mine = learner_lines(code, fix.get("code", ""))
        if mine:
            return {"prompt": "Which line of your code has to change?", "answer": mine[0]}
    return _timeline.question_for(cls, ctx)


_UNSET = object()


def build_package(problem, code, cls, tried=(), modality=None, *, fix=_UNSET, counterexample=_UNSET):
    """The intervention package of 03 §11.2 for one (attempt, class).

    ``tried`` are the modalities already used for this (learner, class); ``modality``
    forces one (the /intervene request field). ``fix`` and ``counterexample`` may be passed
    in if the caller has them already (the diagnosis pipeline computes both); otherwise
    ``fixer.repair`` and ``counterexample.search`` run here.
    """
    cex = _counterexample.search(problem, code, cls) if counterexample is _UNSET else counterexample
    fix_raw = _fixer.repair(problem, code, cls) if fix is _UNSET else fix
    ctx = _timeline.make_context(problem, code, cex)
    kind = _empty_body_kind(ctx.learner)
    chosen, rest, exhausted = choose_modality(cls, tried, modality, kind)
    fix_out = _fix_dict(fix_raw, code) if fix_raw else None

    package = {
        "class": cls,
        "modality": chosen,
        "next_modalities": rest,
        "copy": list(COPY.get(cls, COPY_OTHER)),
        "question": _question(chosen, cls, ctx, cex, fix_out, code),
        "timeline": _timeline.build_timeline(ctx),
    }
    wanted = []
    for name in (PANEL_OF.get(cls), chosen if chosen in PANELS else None):
        if name and name != "trace_timeline" and name not in wanted:
            wanted.append(name)
    for name in wanted:
        key, value = _panel(name, cls, ctx)
        package[key] = value
    package["counterexample"] = cex
    package["fix"] = fix_out
    if cls == "M10":
        package["output"] = {
            "printed": (ctx.learner or {}).get("printed", ""),
            "returned": (ctx.learner or {}).get("returned"),
            "intended": (ctx.ref or {}).get("returned"),
        }
    package["shown_test"] = {"index": ctx.test, "args": list(ctx.args), "source": ctx.source}
    if exhausted:
        package["exhausted"] = True
    return package
