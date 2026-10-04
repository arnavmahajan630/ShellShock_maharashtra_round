"""Runs the short C fragment shown in a probe, trap, exam or quiz item and says what it does.

An item's `code` is a fragment: zero or more function definitions, then loose statements.
The fragment is wrapped into a program without moving any of its lines:

    <function definitions> int rl_item() { <loose statements>
    printf("<<tail>>"); <tail>
    return 0; }

`tail` is extra C the learner does not see, used to make the answer visible, for example
`printf("%d", x);` for "what is x afterwards?". The marker keeps what the tail prints apart
from what the shown code prints.

    seen = observe(code, tail, backend)          # backend: "interp" or "gcc"
    raw = raw_answer(seen, kind)                 # the answer as the machine sees it
    text = say(raw, says)                        # the same answer as the option text

Answer kinds:
    printed          what the shown code printed, stripped
    tail             what the hidden tail printed, stripped
    count:<effect>   how many times a world effect happened, e.g. count:fire
    effect           which world effects happened, e.g. "fire", "door_open", "nothing"
    state            the value asked for by a `state_query` (05 §3.1)

Before the kind is looked at: a run that does not end gives its status ("timeout",
"runtime_error"); a run that read a variable with no value gives "uninit_read"; a run that
touched a cell outside its array gives "oob". A `state` answer is about a moment during the
run, so it is given even when the run never ends. C leaves the last two undefined, so only the
interpreter can see them and gcc is not asked (`defined_in_c`).
"""
import re
from concurrent.futures import ThreadPoolExecutor

from ml.c_interp import harness
from ml.oracle import gcc

ENTRY = "rl_item"
MARK = "<<tail>>"
SIGNATURE = f"int {ENTRY}()"
WORLD_EFFECTS = ["fire", "launch", "door_open", "door_closed", "scan"]
UNDEFINED_EVENTS = {"uninit_read": "uninit_read", "oob_read": "oob", "oob_write": "oob"}
DEFAULT_SAYS = {
    "timeout": "never stops",
    "uninit_read": "unpredictable",
    "oob": "outside the array",
    "fire": "fires",
    "door_open": "door opens",
    "nothing": "nothing happens",
}

_FUNCTION_START = re.compile(r"\s*(?:(?:int|float|double|char|void)\s+)+[A-Za-z_]\w*\s*\(")
_DIRECTIVE = re.compile(r"\s*#[^\n]*(?:\n|$)")


class FragmentError(ValueError):
    pass


def _skip_literal(text, i):
    quote = text[i]
    i += 1
    while i < len(text) and text[i] != quote:
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _matching(text, i, opener, closer):
    """Index just past the bracket that closes the one at text[i]."""
    depth = 0
    while i < len(text):
        char = text[i]
        if char in "\"'":
            i = _skip_literal(text, i)
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise FragmentError("unbalanced brackets")


def split(fragment):
    """(definitions, statements): the leading #define lines and function definitions, and the rest."""
    text = fragment.replace("\r\n", "\n").replace("\r", "\n")
    at = 0
    while True:
        directive = _DIRECTIVE.match(text, at)
        if directive and directive.end() > at:
            at = directive.end()
            continue
        start = _FUNCTION_START.match(text, at)
        if not start:
            break
        after_params = _matching(text, start.end() - 1, "(", ")")
        brace = re.compile(r"\s*\{").match(text, after_params)
        if not brace:
            break
        at = _matching(text, brace.end() - 1, "{", "}")
    return text[:at], text[at:]


def build(fragment, tail=""):
    """The full program. Lines of the fragment keep their numbers."""
    definitions, statements = split(fragment)
    return f'{definitions} int {ENTRY}() {{ {statements}\nprintf("{MARK}"); {tail}\nreturn 0; }}\n'


def _state(steps, query):
    """Value of query.var after query.after_line has run for the query.hit-th time, or None."""
    hits = [s for s in steps if s["line"] == query["after_line"]]
    if len(hits) < query["hit"]:
        return None
    return hits[query["hit"] - 1]["vars"].get(query["var"])


def _instrument(fragment, query, as_float):
    """For gcc: the fragment with a print added at the end of the queried line. The run ends there,
    so a state query also works on code that never stops."""
    lines = fragment.replace("\r\n", "\n").split("\n")
    index = query["after_line"] - 1
    if not 0 <= index < len(lines):
        raise FragmentError(f"state_query.after_line {query['after_line']} is outside the code")
    spec = "%f" if as_float else "%d"
    lines[index] += (f' {{ rl_hits = rl_hits + 1; if (rl_hits == {query["hit"]}) {{ '
                     f'printf("[[{spec}]]", {query["var"]}); __rl_stop(); __rl_ret_int(0); exit(__rl_finish()); }} }}')
    return "\n".join(lines)


def observe(fragment, tail="", backend="interp", state_query=None, state_is_float=False):
    """What happened: {"status", "printed", "effects", "events", "state", "error"}."""
    if backend == "interp":
        program = build(fragment, tail)
        problem = {"signature": SIGNATURE, "tests": [{"args": [], "expect": {}}], "display_test": 0}
        trace = harness.trace(problem, program)
        error = None
        if trace["status"] in ("parse_error", "unsupported"):
            why = harness.check(program, problem)
            error = f"line {why.get('line')}: {why.get('reason')}"
        return {
            "status": trace["status"], "printed": trace["printed"],
            "effects": {name: trace["effects_count"].get(name, 0) for name in WORLD_EFFECTS},
            "events": [e["type"] for e in trace["events"]],
            "state": _state(trace["steps"], state_query) if state_query else None,
            "error": error,
        }
    if backend == "gcc":
        program = build(fragment, tail)
        if state_query:
            program = "int rl_hits = 0; void exit(int); " + build(_instrument(fragment, state_query, state_is_float), tail)
        seen = gcc.observe({"signature": SIGNATURE}, program, [[]])
        run = seen["runs"][0]
        printed, state = run["printed"], None
        if state_query:
            marks = re.findall(r"\[\[(.*?)\]\]", printed)
            printed = re.sub(r"\[\[.*?\]\]", "", printed)
            if marks:
                state = float(marks[0]) if state_is_float else int(marks[0])
        return {"status": run["status"], "printed": printed,
                "effects": {name: run["effects"].get(name, 0) for name in WORLD_EFFECTS},
                "events": [], "state": state, "error": run["error"]}
    raise ValueError(f"unknown backend {backend!r}")


def defined_in_c(seen):
    """False when the interpreter saw a read or write that C leaves undefined."""
    return not any(event in UNDEFINED_EVENTS for event in seen["events"])


def _number(value):
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else f"{value:g}"
    return str(value)


def raw_answer(seen, kind):
    """The machine's answer as a string, or None when there is none."""
    if seen["status"] in ("parse_error", "unsupported"):
        return None
    undefined = next((UNDEFINED_EVENTS[e] for e in seen["events"] if e in UNDEFINED_EVENTS), None)
    if kind == "state":                 # asked about a moment during the run, so the run need not end
        if undefined:
            return undefined
        if seen["state"] is not None:
            return _number(seen["state"])
        return seen["status"] if seen["status"] != "ok" else None
    if seen["status"] != "ok":
        return seen["status"]
    if undefined:
        return undefined
    if kind in ("printed", "tail"):
        shown, _, hidden = seen["printed"].partition(MARK)
        return (shown if kind == "printed" else hidden).strip()
    if kind.startswith("count:"):
        return str(seen["effects"].get(kind[len("count:"):], 0))
    if kind == "effect":
        return "+".join(name for name in WORLD_EFFECTS if seen["effects"].get(name)) or "nothing"
    raise ValueError(f"unknown answer kind {kind!r}")


def say(raw, says=None):
    """The option text for a raw answer: the item's own `says`, then the shared wording, else as is."""
    if raw is None:
        return None
    if says and raw in says:
        return says[raw]
    return DEFAULT_SAYS.get(raw, raw)


def interp_answer(fragment, tail="", kind="tail", says=None, state_query=None):
    """{"text", "defined", "is_float", "status", "events", "error"} from the interpreter.
    `text` is None when the fragment cannot run."""
    seen = observe(fragment, tail, "interp", state_query)
    text = say(raw_answer(seen, kind), says)
    return {"text": text, "defined": defined_in_c(seen), "is_float": isinstance(seen["state"], float),
            "status": seen["status"], "events": seen["events"],
            "error": None if text is not None else (seen["error"] or "no answer")}


def gcc_answer(fragment, tail="", kind="tail", says=None, state_query=None, is_float=False):
    """{"text", "error"} from gcc. Only meaningful when the interpreter said the run is defined in C."""
    seen = observe(fragment, tail, "gcc", state_query, is_float)
    return {"text": say(raw_answer(seen, kind), says), "error": seen["error"]}


def answer(fragment, tail="", kind="tail", says=None, state_query=None, backends=("interp", "gcc")):
    """Run on each backend and give {"interp": text, "gcc": text or None, "agree": bool, "error": str or None}.

    "gcc" is None when C leaves the fragment's behaviour undefined (then only the interpreter decides).
    """
    first = interp_answer(fragment, tail, kind, says, state_query)
    out = {"interp": first["text"], "gcc": None, "agree": first["text"] is not None, "error": first["error"]}
    if out["agree"] and "gcc" in backends and first["defined"]:
        second = gcc_answer(fragment, tail, kind, says, state_query, first["is_float"])
        out["gcc"] = second["text"]
        if out["gcc"] != out["interp"]:
            out["agree"] = False
            out["error"] = f"interpreter says {out['interp']!r}, gcc says {out['gcc']!r} ({second['error']})"
    return out


def check_many(jobs, backends=("interp", "gcc"), workers=6):
    """Run many fragments. Each job is {"fragment", "tail", "kind", "says", "state_query"} (all but
    the fragment optional). One result per job: {"text", "gcc", "defined", "status", "events", "error"}.

    "text" is the interpreter's answer (None when the fragment cannot run). "gcc" is gcc's answer,
    or None when gcc was not asked: not in `backends`, no interpreter answer, or C leaves the run
    undefined. "error" is set when there is no answer or the two disagree.
    """
    def args(job):
        return (job["fragment"], job.get("tail") or "", job.get("kind") or "tail", job.get("says"),
                job.get("state_query"))

    results = []
    for job in jobs:
        try:
            first = interp_answer(*args(job))
        except FragmentError as error:
            first = {"text": None, "defined": False, "is_float": False, "status": "parse_error",
                     "events": [], "error": str(error)}
        results.append({"text": first["text"], "gcc": None, "defined": first["defined"],
                        "is_float": first["is_float"], "status": first["status"],
                        "events": first["events"], "error": first["error"]})

    todo = [i for i, r in enumerate(results) if r["text"] is not None and r["defined"]]
    if "gcc" in backends and todo:
        def ask(i):
            try:
                return gcc_answer(*args(jobs[i]), results[i]["is_float"])
            except FragmentError as error:
                return {"text": None, "error": str(error)}
        firsts = [ask(todo[0])]            # builds the gcc support file once before the threads start
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rest = list(pool.map(ask, todo[1:]))
        for i, got in zip(todo, firsts + rest):
            results[i]["gcc"] = got["text"]
            if got["text"] != results[i]["text"]:
                results[i]["error"] = (f"the interpreter says {results[i]['text']!r}, "
                                       f"gcc says {got['text']!r} ({got['error']})")
    for result in results:
        del result["is_float"]
    return results
