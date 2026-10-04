"""Timeline, call-stack, window and memory builders (ml_plan/03 §7.4). Package R2.

Every builder takes a ``Context``: the problem, the learner's code, the learner's trace
and the reference trace on one test, plus the entry-function loops found in the
source. ``make_context`` builds one from a problem and a code string by running
``ml.runner``; ``context_from_traces`` builds one from traces you already have
(the fixture traces in tests/fixtures/traces/).

Builders (all pure functions of the context):

    build_timeline(ctx)      -> [{step, line, vars, effect, flag}]       (03 §7.4)
    build_call_stack(ctx)    -> [{depth, fn, args, returned, flag}, ..., "…+N"]
    build_window_strip(ctx)  -> [{step, low, mid, high, cmp, mid_value, flag}, ..., "…+N"]
    build_memory_strip(ctx)  -> {array, values, reads[, writes, final, expected, passes]}
    build_value_meter(kind, ctx) -> {kind, ...}      kind "division" (M04) or "char_vs_string" (D08)
    question_for(cls, ctx)   -> {prompt, answer[, options]}

Timeline flags: extra, missing, void_read, static, reset, overwrite, empty_body (03 §7.4)
plus early_return (03 §7.1, D01). Call-stack flags: no_base, same_arg, growing_arg,
dropped_value. Memory-strip write flags: duplicate, lost, one_pass_end.

Which test is shown: the plan says the display test. When the learner already passes
the display test (a bug that hides on it), the timeline would show nothing, so
``make_context`` takes the display test only if the learner fails it, else the first
failing test, else the first test where the learner's counters differ from the
reference's, else the counterexample input if one is given, else the display test.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from pycparser import c_ast, c_parser

from ml import runner
from ml.c_interp.preprocess import preprocess

MAX_TIMELINE_ROWS = 12      # head 10 + tail 2 when longer
TIMELINE_HEAD = 10
TIMELINE_TAIL = 2
MAX_FRAMES = 12             # 03 §7.4
MAX_WINDOW_ROWS = 12
MAX_WRITES = 24
MAX_READS = 30

# Highest priority first.
FLAG_ORDER = ["void_read", "static", "overwrite", "empty_body", "reset", "early_return", "extra"]
EFFECT_ORDER = ["read_void", "write_cell", "compare", "read_cell", "call", "ret"]


# ---------------------------------------------------------------- loops in the source

@dataclass(frozen=True)
class Loop:
    line: int           # header line
    end: int            # last line of any node inside the loop
    kind: str           # for | while | do
    depth: int          # nesting depth, 0 = outermost
    names: tuple        # identifiers in the loop condition


_loop_cache: dict = {}


def loops_of(code):
    """Loops of the source, outermost first by line. Empty if the code does not parse."""
    key = hashlib.sha256((code or "").encode("utf-8")).hexdigest()
    if key in _loop_cache:
        return _loop_cache[key]
    found = []
    try:
        tree = c_parser.CParser().parse(preprocess(code))
    except Exception:
        _loop_cache[key] = found
        return found

    def last_line(node):
        best = node.coord.line if node.coord else 0
        for _, child in node.children():
            best = max(best, last_line(child))
        return best

    def ids(node):
        out = []
        if node is None:
            return out
        if isinstance(node, c_ast.ID):
            out.append(node.name)
        for _, child in node.children():
            out.extend(ids(child))
        return out

    def walk(node, depth):
        deeper = depth
        if isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)) and node.coord:
            kind = "for" if isinstance(node, c_ast.For) else "while" if isinstance(node, c_ast.While) else "do"
            names = tuple(dict.fromkeys(ids(node.cond)))
            found.append(Loop(node.coord.line, last_line(node), kind, depth, names))
            deeper = depth + 1
        for _, child in node.children():
            walk(child, deeper)

    walk(tree, 0)
    found.sort(key=lambda lp: lp.line)
    if len(_loop_cache) > 256:
        _loop_cache.clear()
    _loop_cache[key] = found
    return found


def main_loop(loops, per_test_iters):
    """The loop with the most iterations on this test; ties go to the outermost (W0 decision 16).

    Returns ``(Loop, iterations)`` or ``(None, 0)``.
    """
    best = None
    for lp in loops:
        count = (per_test_iters or {}).get(f"L{lp.line}", 0)
        if not count:
            continue
        rank = (-count, lp.depth, lp.line)
        if best is None or rank < best[0]:
            best = (rank, lp, count)
    if best is None:
        return None, 0
    return best[1], best[2]


def split_runs(steps, loop):
    """Steps of one loop as ``runs -> passes -> steps``.

    A run is one entry into the loop (a nested loop has one per outer pass). A pass is
    the body steps between two header-line steps. do-while loops are not split.
    """
    if loop is None or loop.kind == "do":
        return []
    runs = []
    current = None
    body = []
    prev_in = False
    for step in steps:
        line = step["line"]
        inside = loop.line <= line <= loop.end
        if not inside:
            if current is not None:
                if body:
                    current.append(body)
                body = []
                runs.append(current)
                current = None
            prev_in = False
            continue
        if not prev_in:
            current = []
            body = []
        prev_in = True
        if line == loop.line:
            if body:
                current.append(body)
                body = []
        else:
            body.append(step)
    if current is not None:
        if body:
            current.append(body)
        runs.append(current)
    return [run for run in runs if run]


def _flat_passes(runs):
    return [p for run in runs for p in run]


# ---------------------------------------------------------------- context

@dataclass
class Context:
    problem: dict
    code: str
    test: int
    args: list
    learner: dict
    ref: dict | None
    ref_code: str | None = None
    # display (learner fails it) | failing | differs | counterexample | none (nothing differs) | given
    source: str = "display"
    _cache: dict = field(default_factory=dict)

    # -- loops and passes
    def learner_loop(self):
        return self._loop("learner")

    def ref_loop(self):
        return self._loop("ref")

    def _loop(self, who):
        key = ("loop", who)
        if key not in self._cache:
            trace, code = (self.learner, self.code) if who == "learner" else (self.ref, self.ref_code)
            if not trace or not code:
                self._cache[key] = (None, 0)
            else:
                self._cache[key] = main_loop(loops_of(code), _test_counters(trace, self.test).get("loop_iters"))
        return self._cache[key]

    def runs(self, who="learner"):
        key = ("runs", who)
        if key not in self._cache:
            trace = self.learner if who == "learner" else self.ref
            loop = self._loop(who)[0]
            self._cache[key] = split_runs((trace or {}).get("steps") or [], loop)
        return self._cache[key]

    def passes(self, who="learner"):
        return _flat_passes(self.runs(who))

    def events(self, who="learner"):
        trace = self.learner if who == "learner" else self.ref
        return events_of(trace, self.test)


def events_of(trace, test):
    """Events of one test (the trace list carries a ``test`` index on each event)."""
    out = []
    for event in (trace or {}).get("events") or []:
        if event.get("test", test) == test:
            out.append(event)
    return out


def _test_counters(trace, test):
    per = (trace or {}).get("per_test") or []
    if 0 <= test < len(per):
        return per[test]
    return trace or {}


# -- running and caching traces

_trace_cache: dict = {}


def _digest(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def trace_of(problem, code, test_index=None):
    """``runner.trace`` with a small in-process cache."""
    key = (problem.get("problem_id"), _digest(code), test_index, _digest(repr(problem.get("tests"))))
    hit = _trace_cache.get(key)
    if hit is None:
        if len(_trace_cache) > 256:
            _trace_cache.clear()
        hit = runner.trace(problem, code, test_index)
        _trace_cache[key] = hit
    return hit


def clear_cache():
    _trace_cache.clear()
    _loop_cache.clear()


def _sig(trace, test):
    """What 'the learner behaves differently from the reference' compares on one test."""
    per = _test_counters(trace, test)
    kinds = Counter(e["type"] for e in events_of(trace, test))
    iters = tuple(sorted((per.get("loop_iters") or {}).items()))
    return (per.get("status"), repr(per.get("returned")), per.get("printed"),
            tuple(sorted((per.get("effects_count") or {}).items())), tuple(sorted(kinds.items())), iters)


def choose_test(problem, code, base, ref_base):
    """``(test_index, source)`` — see the module docstring."""
    display = problem.get("display_test", 0) or 0
    total = len(problem.get("tests") or [])
    failing = []
    try:
        results = runner.run_tests(problem, code)["tests"]["results"]
        failing = [i for i, row in enumerate(results) if not row.get("pass")]
    except Exception:
        failing = []
    if display in failing:
        return display, "display"
    if failing:
        return failing[0], "failing"
    if ref_base is not None:
        for i in range(total):
            if _sig(base, i) != _sig(ref_base, i):
                return i, "differs"
    return display, "none"


def make_context(problem, code, counterexample=None, test=None):
    """Run the learner's code and the first correct variant and pick the test to show.

    ``counterexample`` is the dict from ``counterexample.search`` (used only when every
    test behaves the same). ``test`` forces a test index.
    """
    variants = problem.get("correct_variants") or []
    ref_code = variants[0] if variants else None
    base = trace_of(problem, code, None)
    ref_base = trace_of(problem, ref_code, None) if ref_code else None
    display = problem.get("display_test", 0) or 0
    if test is not None:
        index, source = test, "given"
    else:
        index, source = choose_test(problem, code, base, ref_base)
    if source == "none" and counterexample:
        mini = _mini_problem(problem, counterexample)
        if mini is not None:
            learner = trace_of(mini, code, 0)
            ref = trace_of(mini, ref_code, 0) if ref_code else None
            return Context(mini, code, 0, list(mini["tests"][0]["args"]), learner, ref, ref_code, "counterexample")
    learner = base if index == display else trace_of(problem, code, index)
    ref = ref_base if index == display else (trace_of(problem, ref_code, index) if ref_code else None)
    tests = problem.get("tests") or []
    args = list(tests[index]["args"]) if 0 <= index < len(tests) else []
    return Context(problem, code, index, args, learner, ref, ref_code, source)


def _mini_problem(problem, counterexample):
    inputs = counterexample.get("input")
    if not isinstance(inputs, dict):
        return None
    mini = copy.deepcopy(problem)
    mini["problem_id"] = str(problem.get("problem_id")) + "#cex"
    mini["tests"] = [{"args": list(inputs.values()), "expect": {"returned": 0}}]
    mini["display_test"] = 0
    return mini


def context_from_traces(problem, code, learner, ref=None, test=0, ref_code=None):
    """A context from traces you already have. ``test`` indexes ``per_test`` and the problem's tests."""
    if ref_code is None:
        variants = problem.get("correct_variants") or []
        ref_code = variants[0] if variants else None
    tests = problem.get("tests") or []
    args = list(tests[test]["args"]) if 0 <= test < len(tests) else []
    return Context(problem, code, test, args, learner, ref, ref_code, "given")


# ---------------------------------------------------------------- small helpers

def _num(text):
    if text is None or text == "":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_effect(effect):
    """``"call:f:2:3,4"`` -> ``("call", ["f", "2", "3,4"])``. Fields are kept as strings."""
    name, _, rest = effect.partition(":")
    limit = {"call": 2, "ret": 2}.get(name, None)
    fields = rest.split(":", limit) if limit else (rest.split(":") if rest else [])
    return name, fields


def _effect_pick(effects):
    """The most telling effect of a list, or None."""
    if not effects:
        return None
    ranked = sorted(effects, key=lambda e: EFFECT_ORDER.index(e.split(":")[0]) if e.split(":")[0] in EFFECT_ORDER else len(EFFECT_ORDER))
    return ranked[0]


def _step_flags(step):
    """Set of timeline flags this single step carries on its own."""
    flags = set()
    for effect in step.get("effects") or []:
        if effect.startswith("read_void"):
            flags.add("void_read")
    for event in step.get("events") or []:
        kind = event.get("type")
        if kind == "oob_read":
            flags.add("void_read")
        elif kind == "uninit_read":
            flags.add("static")
        elif kind == "assign_in_cond":
            flags.add("overwrite")
        elif kind == "empty_body":
            flags.add("empty_body")
    return flags


def _pick_flag(flags):
    for name in FLAG_ORDER:
        if name in flags:
            return name
    return None


def _param_names(signature):
    match = re.search(r"\((.*)\)", signature or "")
    if not match or not match.group(1).strip():
        return []
    names = []
    for part in match.group(1).split(","):
        token = part.strip().replace("[", " ").replace("]", " ").split()
        names.append(token[-1] if token else "arg")
    return names


def array_param(problem):
    """``(name, index)`` of the first array parameter of the entry function, else ``(None, None)``."""
    match = re.search(r"\((.*)\)", problem.get("signature") or "")
    if not match:
        return None, None
    for index, part in enumerate(match.group(1).split(",")):
        if "[" in part and "char" not in part:
            token = part.strip().replace("[", " ").replace("]", " ").split()
            return (token[-1] if token else None), index
    return None, None


def _cap_rows(rows):
    if len(rows) <= MAX_TIMELINE_ROWS:
        return rows
    return rows[:TIMELINE_HEAD] + rows[-TIMELINE_TAIL:]


# ---------------------------------------------------------------- timeline

def build_timeline(ctx):
    """``[{step, line, vars, effect, flag}]`` for the shown test (03 §7.4).

    With a loop: one row per pass of the main loop, aligned by pass number with the
    reference. A pass beyond the reference's last is ``extra``; a reference pass the
    learner never reached is a ``missing`` row placed after the learner's rows (its
    ``step`` continues the learner's numbering; ``vars`` are the reference's). Without
    a loop (or without the source): one row per step that has an effect, an event or a
    change of variables. Capped at 12 rows (first 10, last 2).
    """
    rows = _loop_rows(ctx)
    if rows is None:
        rows = _flat_rows(ctx)
    return _cap_rows(rows)


def _row(step, flag, effect=None):
    return {
        "step": step["i"],
        "line": step["line"],
        "vars": dict(step.get("vars") or {}),
        "effect": effect if effect is not None else _effect_pick(step.get("effects") or []),
        "flag": flag,
    }


def _loop_rows(ctx):
    passes = ctx.passes("learner")
    if not passes:
        return None
    ref_passes = ctx.passes("ref") if ctx.ref_loop()[0] is not None else None
    skip = set(ctx.learner_loop()[0].names)
    resets = _reset_hits(passes, skip)
    ref_reset_vars = set()
    ref_vars = set()
    if ref_passes:
        ref_reset_vars = {var for _, var in _reset_hits(ref_passes, set(ctx.ref_loop()[0].names)).values()}
        for seg in ref_passes:
            for st in seg:
                ref_vars.update(st.get("vars") or {})
    rows = []
    for k, seg in enumerate(passes):
        flags = set()
        rep = None
        for st in seg:
            found = _step_flags(st)
            if found and rep is None:
                rep = st
            flags |= found
        if k in resets:
            st, var = resets[k]
            if var not in ref_reset_vars and var in ref_vars:
                flags.add("reset")
                if rep is None or "reset" == _pick_flag(flags):
                    rep = st
        last = seg[-1]
        if last.get("effects") and any(e.startswith("ret:") for e in last["effects"]) and ref_passes is not None \
                and k + 1 == len(passes) and len(passes) < len(ref_passes):
            flags.add("early_return")
        if ref_passes is not None and k >= len(ref_passes):
            flags.add("extra")
        flag = _pick_flag(flags)
        chosen = rep if rep is not None else last
        # The chosen step decides the row; if it has no effect, use the pass's most telling one.
        effect = _effect_pick(chosen.get("effects") or [])
        if effect is None:
            effect = _effect_pick([e for st in seg for e in (st.get("effects") or [])])
        rows.append(_row(chosen, flag, effect))
    if ref_passes and len(passes) < len(ref_passes):
        base = rows[-1]["step"]
        for j, seg in enumerate(ref_passes[len(passes):], start=1):
            last = seg[-1]
            row = _row(last, "missing")
            row["step"] = base + j
            rows.append(row)
            if len(rows) > 60:
                break
    return rows


def _reset_hits(passes, skip):
    """``{pass_index: (step, var)}``: a variable that goes back to the value it first took in pass 0."""
    if len(passes) < 2:
        return {}
    # Only a variable that is set and then changed again inside pass 0 can be "reset":
    # an accumulator or a flag. A temp that is set once per pass is not one.
    first = {}
    seen = {}
    for st in passes[0]:
        for var, value in (st.get("vars") or {}).items():
            if value is None or var in skip:
                continue
            run = seen.setdefault(var, [])
            if not run or run[-1] != value:
                run.append(value)
    for var, run in seen.items():
        if len(run) >= 2:
            first[var] = run[0]
    prev_end = passes[0][-1].get("vars") or {}
    hits = {}
    for k in range(1, len(passes)):
        for st in passes[k]:
            now = st.get("vars") or {}
            for var, init in first.items():
                before = prev_end.get(var)
                if now.get(var) == init and before is not None and before != init and k not in hits:
                    hits[k] = (st, var)
        prev_end = passes[k][-1].get("vars") or {}
    return hits


def _flat_rows(ctx):
    steps = (ctx.learner or {}).get("steps") or []
    rows = []
    prev_vars = None
    for index, st in enumerate(steps):
        flag = _pick_flag(_step_flags(st))
        interesting = bool(flag) or bool(st.get("effects")) or st.get("vars") != prev_vars \
            or index == 0 or index == len(steps) - 1
        prev_vars = st.get("vars")
        if not interesting:
            continue
        rows.append(_row(st, flag))
    return rows


# ---------------------------------------------------------------- call stack

def _frames(trace):
    """Frames from the call / ret effects of the recorded steps, plus the dropped-value marks."""
    frames = []
    stack = []
    dropped = set()
    for st in (trace or {}).get("steps") or []:
        for effect in st.get("effects") or []:
            name, fields = parse_effect(effect)
            if name == "call" and len(fields) >= 2:
                raw = fields[2] if len(fields) > 2 else ""
                args = [_num(a) for a in raw.split(",")] if raw != "" else []
                try:
                    depth = int(fields[1])
                except ValueError:
                    depth = len(stack) + 1
                frames.append({"depth": depth, "fn": fields[0], "args": args, "returned": None, "flag": None})
                stack.append(len(frames) - 1)
            elif name == "ret" and stack:
                value = fields[2] if len(fields) > 2 else ""
                frames[stack.pop()]["returned"] = _num(value)
        for event in st.get("events") or []:
            if event.get("type") == "discarded_call_value" and stack:
                dropped.add(stack[-1])
    return frames, dropped


def _parents(frames):
    """For each frame the index of its caller (the nearest earlier frame one level up), or None."""
    last_at = {}
    out = []
    for index, frame in enumerate(frames):
        out.append(last_at.get(frame["depth"] - 1))
        last_at[frame["depth"]] = index
    return out


def _first_number(args):
    for position, value in enumerate(args):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return position, value
    return None, None


def _direction(frames):
    """Majority sign of (child - parent) on the first numeric argument; None if there are no pairs."""
    signs = Counter()
    for frame, parent in zip(frames, _parents(frames)):
        if parent is None:
            continue
        position, child = _first_number(frame["args"])
        if position is None or position >= len(frames[parent]["args"]):
            continue
        before = frames[parent]["args"][position]
        if isinstance(before, (int, float)):
            signs[(child > before) - (child < before)] += 1
    if not signs:
        return None
    return signs.most_common(1)[0][0]


def build_call_stack(ctx):
    """``[{depth, fn, args, returned, flag}]`` in call order, 12 frames then ``"…+N"`` (03 §7.4).

    Flags: ``same_arg`` / ``growing_arg`` (the first numeric argument does not move the
    way the reference's does), ``no_base`` (the first frame deeper than the reference ever
    goes; with no reference, the frame the depth cap stopped), ``dropped_value`` (the
    frame discarded a recursive call's result). ``same_arg`` and ``growing_arg`` win
    over ``no_base``.
    """
    frames, dropped = _frames(ctx.learner)
    ref_frames, _ = _frames(ctx.ref) if ctx.ref else ([], set())
    ref_dir = _direction(ref_frames) if ref_frames else -1
    if ref_dir is None:
        ref_dir = -1
    ref_depth = max([f["depth"] for f in ref_frames], default=None)
    parents = _parents(frames)
    capped = any(e.get("type") == "depth_cap_hit" for e in (ctx.learner or {}).get("events") or [])
    base_marked = False
    for index, frame in enumerate(frames):
        flag = None
        parent = parents[index]
        if parent is not None and ref_dir != 0:
            position, child = _first_number(frame["args"])
            if position is not None and position < len(frames[parent]["args"]):
                before = frames[parent]["args"][position]
                if isinstance(before, (int, float)):
                    if child == before:
                        flag = "same_arg"
                    elif ref_dir == -1 and child > before:
                        flag = "growing_arg"
        if flag is None and not base_marked:
            beyond = ref_depth is not None and frame["depth"] > ref_depth
            if beyond or (ref_depth is None and capped and index == len(frames) - 1):
                flag = "no_base"
                base_marked = True
        if flag is None and index in dropped:
            flag = "dropped_value"
        frame["flag"] = flag
    shown = frames[:MAX_FRAMES]
    if len(frames) > MAX_FRAMES:
        shown = shown + [f"…+{len(frames) - MAX_FRAMES}"]
    return shown


# ---------------------------------------------------------------- window strip

def _window_names(problem, steps):
    names = list(problem.get("markers") or [])
    if len(names) == 3:
        return names
    seen = set()
    for st in steps[:3]:
        seen.update((st.get("vars") or {}).keys())
    for triple in (("low", "mid", "high"), ("lo", "mid", "hi")):
        if all(name in seen for name in triple):
            return list(triple)
    return None


def _target_of(problem, args):
    """``(array or None, target)`` for a search-style problem."""
    name, index = array_param(problem)
    array = args[index] if index is not None and index < len(args) and isinstance(args[index], list) else None
    names = _param_names(problem.get("signature"))
    scalar = [(n, a) for n, a in zip(names, args) if not isinstance(a, (list, str))]
    if array is None:
        return None, scalar[0][1] if scalar else None
    for n, a in scalar:
        if n in ("x", "target", "key", "val", "value", "k"):
            return array, a
    return array, scalar[-1][1] if scalar else None


def build_window_strip(ctx):
    """One row per pass of the main loop: the window ``low``/``high`` at the start of the
    pass, ``mid``, how the cell at ``mid`` compares with the target, and ``flag: "frozen"``
    when (low, high) is the same as in the previous pass. Capped at 12 rows then ``"…+N"``.
    """
    passes = ctx.passes("learner")
    names = _window_names(ctx.problem, (ctx.learner or {}).get("steps") or [])
    if not passes or not names:
        return []
    low_n, mid_n, high_n = names
    array, target = _target_of(ctx.problem, ctx.args)
    rows = []
    previous = None
    for seg in passes:
        pick = next((st for st in seg if (st.get("vars") or {}).get(mid_n) is not None), None)
        if pick is None:
            continue
        now = pick["vars"]
        low, mid, high = now.get(low_n), now.get(mid_n), now.get(high_n)
        value = None
        cmp = None
        if array is not None and isinstance(mid, int) and 0 <= mid < len(array):
            value = array[mid]
        elif array is None:
            value = mid
        if value is not None and target is not None:
            cmp = "==" if value == target else "<" if value < target else ">"
        flag = "frozen" if previous == (low, high) else None
        previous = (low, high)
        rows.append({"step": pick["i"], "low": low, "mid": mid, "high": high,
                     "cmp": cmp, "mid_value": value, "flag": flag})
    if len(rows) > MAX_WINDOW_ROWS:
        extra = len(rows) - MAX_WINDOW_ROWS
        return rows[:MAX_WINDOW_ROWS] + [f"…+{extra}"]
    return rows


# ---------------------------------------------------------------- memory strip

def _writes_and_reads(trace):
    reads, writes = [], []
    for st in (trace or {}).get("steps") or []:
        for effect in st.get("effects") or []:
            name, fields = parse_effect(effect)
            if name in ("read_cell", "read_void") and fields:
                value = _num(fields[0])
                if isinstance(value, int):
                    reads.append(value)
            elif name == "write_cell" and len(fields) >= 2:
                i, v = _num(fields[0]), _num(fields[1])
                if isinstance(i, int):
                    writes.append((st["i"], i, v))
    return reads, writes


def _replay(values, writes):
    arr = list(values)
    frames = []
    for step, i, v in writes:
        old = arr[i] if 0 <= i < len(arr) else None
        if 0 <= i < len(arr):
            arr[i] = v
        frames.append({"step": step, "i": i, "old": old, "new": v, "flag": None})
    return frames, arr


def build_memory_strip(ctx):
    """The array and how the code touched it.

    Always ``{array, values, reads}`` (``values`` is the test's input array, ``reads`` the
    cell indexes read in order, out-of-range ones included). When the code writes cells it
    also has ``writes`` (``{step, i, old, new, flag}``, flags ``duplicate`` / ``lost`` /
    ``one_pass_end``), ``final`` (the array after the learner's writes) and, with a
    reference, ``expected``. ``passes`` ``{yours, needed}`` is added when the main loop was
    entered fewer times than the reference's.
    """
    name, index = array_param(ctx.problem)
    values = []
    if index is not None and index < len(ctx.args) and isinstance(ctx.args[index], list):
        values = list(ctx.args[index])
    reads, writes = _writes_and_reads(ctx.learner)
    strip = {"array": name or "a", "values": values, "reads": reads[:MAX_READS]}
    if not writes:
        return strip
    frames, final = _replay(values, writes)
    init, end = Counter(values), Counter(final)
    lost = {v for v in init if end[v] < init[v]}
    dup = {v for v in end if end[v] > init.get(v, 0)}
    for value in lost:
        for frame in reversed(frames):
            if frame["old"] == value:
                frame["flag"] = "lost"
                break
    for value in dup:
        for frame in reversed(frames):
            if frame["new"] == value and frame["flag"] is None:
                frame["flag"] = "duplicate"
                break
    expected = None
    if ctx.ref:
        _, ref_writes = _writes_and_reads(ctx.ref)
        if ref_writes:
            expected = _replay(values, ref_writes)[1]
    yours_passes = len(ctx.runs("learner"))
    needed_passes = len(ctx.runs("ref")) if ctx.ref_loop()[0] is not None else 0
    if needed_passes and yours_passes < needed_passes:
        strip["passes"] = {"yours": yours_passes, "needed": needed_passes}
        if frames and frames[-1]["flag"] is None:
            frames[-1]["flag"] = "one_pass_end"
    strip["writes"] = frames[:MAX_WRITES]
    strip["final"] = final
    if expected is not None:
        strip["expected"] = expected
    return strip


# ---------------------------------------------------------------- value meter

def _source_line(code, line):
    lines = (code or "").splitlines()
    return lines[line - 1] if 1 <= line <= len(lines) else ""


def build_value_meter(kind, ctx):
    """``division`` (M04): the exact and the integer-division result. ``char_vs_string`` (D08):
    the character's code against a string literal."""
    if kind == "division":
        line = None
        for event in ctx.events("learner"):
            if event.get("type") == "intdiv" and event.get("remainder_nonzero"):
                line = event.get("line")
                break
        exact = (ctx.ref or {}).get("returned")
        yours = (ctx.learner or {}).get("returned")
        lost = None
        if isinstance(exact, (int, float)) and isinstance(yours, (int, float)):
            lost = round(exact - yours, 6)
        return {"kind": "division", "line": line, "exact": exact, "yours": yours, "lost": lost}
    if kind == "char_vs_string":
        line = None
        for event in ctx.events("learner"):
            if event.get("type") in ("str_literal_compare", "array_compare"):
                line = event.get("line")
                break
        char = "a"
        if line:
            found = re.search(r'"([^"]*)"', _source_line(ctx.code, line))
            if found and found.group(1):
                char = found.group(1)[0]
        return {
            "kind": "char_vs_string",
            "line": line,
            "char": {"text": f"'{char}'", "code": ord(char)},
            "string": {"text": f'"{char}"', "note": "an array of characters; == compares where it lives, not what it holds"},
        }
    raise ValueError(f"unknown value meter kind: {kind}")


# ---------------------------------------------------------------- questions (03 §7.4)

def _loop_var(ctx, who="ref"):
    loop = ctx.ref_loop()[0] if who == "ref" else ctx.learner_loop()[0]
    trace = ctx.ref if who == "ref" else ctx.learner
    if loop is None or not trace:
        return None
    steps = trace.get("steps") or []
    best = None
    for name in loop.names:
        values = [s["vars"].get(name) for s in steps if s.get("vars") and name in s["vars"]]
        values = [v for v in values if v is not None]
        if len(set(values)) > 1:
            return name
        if best is None and values:
            best = name
    return best


def _exit_value(ctx, name):
    loop = ctx.ref_loop()[0]
    if loop is None or name is None:
        return None
    last = None
    for st in (ctx.ref or {}).get("steps") or []:
        if st["line"] == loop.line:
            last = st
    if last is None:
        return None
    return (last.get("vars") or {}).get(name)


def _ref_pass_count(ctx):
    loop, count = ctx._loop("ref")
    return count if loop is not None else None


def _accumulator(ctx):
    """The variable that holds the reference's answer at the end, else the last one to change."""
    trace = ctx.ref or {}
    steps = trace.get("steps") or []
    if not steps:
        return None, None
    skip = set((ctx.ref_loop()[0].names) if ctx.ref_loop()[0] else ())
    final = steps[-1].get("vars") or {}
    returned = trace.get("returned")
    for name, value in final.items():
        if name not in skip and value is not None and value == returned:
            return name, value
    changed = None
    for name, value in final.items():
        if name not in skip and value is not None:
            changed = (name, value)
    return changed if changed else (None, None)


def _as_reference(ctx):
    """The same test with the reference playing the learner."""
    return Context(ctx.problem, ctx.ref_code, ctx.test, ctx.args, ctx.ref, ctx.ref, ctx.ref_code, ctx.source)


def question_for(cls, ctx):
    """``{prompt, answer}`` for the class's primary panel (03 §7.4, one template per class).

    ``answer`` is ``None`` when the reference gives no value to read it from.
    """
    ref = ctx.ref or {}
    returned = ref.get("returned")
    generic = {"prompt": "What should the function return for this input?", "answer": returned}
    if cls == "M01":
        name = _loop_var(ctx, "ref")
        return {"prompt": f"At which value of {name or 'the loop variable'} should the loop stop?",
                "answer": _exit_value(ctx, name)}
    if cls == "M02":
        return {"prompt": "How many times should the loop run?", "answer": _ref_pass_count(ctx)}
    if cls in ("M03", "M05"):
        name, value = _accumulator(ctx)
        if name is None:
            return generic
        return {"prompt": f"What should {name} be when the loop is done?", "answer": value}
    if cls == "M06":
        var = next((e.get("var") for e in ctx.events("learner") if e.get("type") == "assign_in_cond"), None)
        line = next((e.get("line") for e in ctx.events("learner") if e.get("type") == "assign_in_cond"), None)
        if var and line:
            for st in ref.get("steps") or []:
                if st["line"] == line:
                    return {"prompt": f"What is {var} right after the check?", "answer": (st.get("vars") or {}).get(var)}
        return generic
    if cls == "M07":
        empty = next((e for e in ctx.events("learner") if e.get("type") == "empty_body"), None)
        if empty and empty.get("kind") == "if":
            steps = (ctx.learner or {}).get("steps") or []
            after = None
            for i, st in enumerate(steps):
                if st["line"] == empty["line"] and any(e.get("type") == "empty_body" for e in st.get("events") or []):
                    after = steps[i + 1]["line"] if i + 1 < len(steps) else None
                    break
            return {"prompt": "Which line should run only when the condition is true?", "answer": after}
        return {"prompt": "How many times should the loop body run?", "answer": _ref_pass_count(ctx)}
    if cls == "M08":
        good = [i for i in _writes_and_reads(ref)[0] if i >= 0]
        cells = [i for st in ref.get("steps") or [] for e in st.get("effects") or []
                 if e.startswith("read_cell:") for i in [_num(e.split(":")[1])] if isinstance(i, int)]
        answer = max(cells) if cells else (max(good) if good else None)
        if answer is None:
            _, index = array_param(ctx.problem)
            if index is not None and index < len(ctx.args) and isinstance(ctx.args[index], list):
                answer = len(ctx.args[index]) - 1
        return {"prompt": "Which is the last cell the loop should read?", "answer": answer}
    if cls == "M04":
        return {"prompt": "What should the function return for this input?", "answer": returned}
    if cls == "M10":
        return {"prompt": "What should the call hand back to its caller?", "answer": returned}
    if cls == "D01":
        return {"prompt": "How many cells should be checked before the search gives its answer?",
                "answer": _ref_pass_count(ctx)}
    if cls == "D02":
        rows = [r for r in build_window_strip(_as_reference(ctx)) if isinstance(r, dict)]
        answer = [rows[1]["low"], rows[1]["high"]] if len(rows) > 1 else None
        return {"prompt": "After the first check, which window (low, high) should come next?", "answer": answer}
    if cls == "D03":
        strip = build_memory_strip(_as_reference(ctx))
        return {"prompt": "What should the array hold when the function is done?", "answer": strip.get("final", strip["values"])}
    if cls == "D04":
        count = len(ctx.runs("ref")) if ctx.ref_loop()[0] is not None else 0
        if not count:
            _, index = array_param(ctx.problem)
            if index is not None and index < len(ctx.args) and isinstance(ctx.args[index], list):
                count = max(len(ctx.args[index]) - 1, 1)
        return {"prompt": "How many passes does this array need?", "answer": count or None}
    if cls in ("D05", "D06"):
        frames, _ = _frames(ctx.ref) if ctx.ref else ([], set())
        answer = None
        if frames:
            deepest = max(frames, key=lambda f: f["depth"])
            _, answer = _first_number(deepest["args"])
        return {"prompt": "At which argument should the calls stop?", "answer": answer}
    if cls == "D07":
        return {"prompt": "What should the first call return?", "answer": returned}
    if cls == "D08":
        return {"prompt": 'Does "abc" == "abc" compare the letters in C?',
                "options": ["Yes", "No, it compares where the text lives"],
                "answer": "No, it compares where the text lives"}
    return generic
