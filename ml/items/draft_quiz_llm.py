"""Drafts quiz-item candidates with DeepSeek (plans/05 §3.1, §7), and writes their explanations.

    python -m ml.items.draft_quiz_llm drafts                 all classes and types
    python -m ml.items.draft_quiz_llm drafts --classes M01 D05
    python -m ml.items.draft_quiz_llm explain                one sentence per built item

Every call is saved under ml/data/llm_raw/quiz_drafts/ or quiz_explain/, so a rerun costs nothing.

The model is never asked for an answer. For each exercise it writes two fragments:
    code          what the learner sees; it contains one mistake
    belief_code   the program a learner who holds that mistake thinks they are reading
ml/items/build_quiz.py runs both and takes the correct answer and the belief answer from the
runs. A draft whose two fragments do not run, or give the same result, is thrown away there.

`explain` is the second pass: for each item that build_quiz.py kept, the model gets the code and
the answers the machine found, and writes one sentence saying why. Output: ml/items/quiz_explain.json.
"""
import argparse
import json
import sys
from pathlib import Path

from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS
from ml.text import llm_client

HERE = Path(__file__).resolve().parent
DRAFTS = HERE / "quiz_drafts.json"
EXPLAIN = HERE / "quiz_explain.json"
ITEMS = HERE.parent / "data" / "quiz_items.json"
CHECKS = HERE.parent / "data" / "quiz_checks.json"

DRAFT_MODEL = "deepseek-v4-pro"       # writes code; hidden reasoning on
EXPLAIN_MODEL = "deepseek-v4-pro"     # explains an answer it is given; hidden reasoning on (see notes/B4.md)
PER_CALL = {"predict_output": 6, "mcq": 5, "next_state": 4}

# concept: one of classes.PLANETS or classes.SECTORS (the filter of GET /quiz/next)
# shows:   how the mistake looks in code
# thinks:  how to write the program the learner believes they are reading
GUIDE = {
    "M01": {"concept": "loops",
            "shows": "a counting loop whose number of passes is easy to misjudge by one: `i <= n` counting from 0, "
                     "`i < n - 1`, a loop that starts at 1 with `<`, or a countdown with `>= 0`. "
                     "Do not index an array with the loop variable.",
            "thinks": "the same code with the comparison the student imagines (`<=` read as `<`, or `<` read as "
                      "`<=`), so the loop makes one pass fewer or one pass more"},
    "M02": {"concept": "loops",
            "shows": "a `while` loop (or a `for` loop with an empty update part) whose loop variable never changes, "
                     "or changes only inside an `if` that is not taken, so the loop never ends",
            "thinks": "the same code with the missing update (for example `i++;`) added as the last line of the "
                      "loop body, so the loop ends normally"},
    "M03": {"concept": "loops",
            "shows": "a loop that adds to a running total (or a count, or a product), where the total is set back "
                     "to its start value inside the loop body before the add",
            "thinks": "the same code with the total set once before the loop and the line inside the loop removed"},
    "M04": {"concept": "variables",
            "shows": "two `int` values divided with `/` where the true quotient has a fractional part, and the "
                     "result stored in a `float` variable",
            "thinks": "the same code with `(float)` written before the first operand of the division, so the exact "
                      "decimal is computed"},
    "M05": {"concept": "variables",
            "shows": "a local variable declared without a value (for example `int total;`) and then read "
                     "(`total += ...`, `total++`, or used in a condition)",
            "thinks": "the same code with the declaration giving 0 (`int total = 0;`)"},
    "M06": {"concept": "conditions",
            "shows": "a single `=` inside an `if` or `while` condition (for example `if (mode = 2)`), where `==` "
                     "would give a different result. The code must still end.",
            "thinks": "the same code with `==` in that condition"},
    "M07": {"concept": "conditions",
            "shows": "a `;` directly after an `if (...)` or `for (...)` header, so the statement or block on the "
                     "next lines is not controlled by it. The code must still end.",
            "thinks": "the same code without that `;`"},
    "M08": {"concept": "arrays",
            "shows": "an `int` array with an initialiser that is read as if its first cell were index 1: `a[1]` "
                     "used as the first value, or a loop `for (i = 1; i < n; i++)` that skips `a[0]`. Every index "
                     "must stay inside the array.",
            "thinks": "the same code with every array index shifted down by one (`a[i]` becomes `a[i - 1]`, `a[1]` "
                      "becomes `a[0]`): the cells a student who counts from 1 thinks are being read"},
    "M10": {"concept": "functions",
            "shows": "a function that prints its result with `printf` and then returns 0; the caller stores the "
                     "returned value in a variable and uses it. When the function prints, it prints the number "
                     "followed by one space (`printf(\"%d \", x);`), never a newline.",
            "thinks": "the same code where the function still prints its result and ALSO returns that result "
                      "(the student believes that printing a value hands it back to the caller)"},
    "D01": {"concept": "searching",
            "shows": "a search over an array with `if (a[i] == x) return i; else return -1;` inside the loop, or "
                     "the flag form `if (a[i] == x) found = 1; else found = 0;`. The wanted value must not be in "
                     "the first cell, and for the flag form not in the last cell either.",
            "thinks": "the same code without the `else` part (so the loop keeps looking and nothing is overwritten)"},
    "D02": {"concept": "searching",
            "shows": "a binary search over a sorted array (or a number-guessing loop) that narrows the window with "
                     "`low = mid;` instead of `low = mid + 1;`, with values chosen so that the window stops "
                     "shrinking and the loop never ends",
            "thinks": "the same code with `low = mid + 1;` (and `high = mid - 1;`)"},
    "D03": {"concept": "array_tech",
            "shows": "two array cells 'swapped' with two plain assignments and no temporary variable "
                     "(`a[i] = a[j]; a[j] = a[i];`), or a shift that overwrites a cell before its value was saved",
            "thinks": "the same code with a correct swap through a temporary variable"},
    "D04": {"concept": "sorting",
            "shows": "exactly one pass of bubble sort (one loop over neighbouring pairs, no outer loop) or one "
                     "selection step, on an array that needs more than one pass to become sorted",
            "thinks": "the same code with the outer loop added, so the array ends up fully sorted"},
    "D05": {"concept": "recursion",
            "shows": "a recursive function with no base case, or with a base case the calls jump over (for example "
                     "`if (n == 0)` with a call `f(n - 2)` started on an odd number)",
            "thinks": "the same code with a base case that is reached (for example `if (n <= 0) return 0;`)"},
    "D06": {"concept": "recursion",
            "shows": "a recursive function with a correct base case whose recursive call passes the same or a "
                     "larger argument (`f(n)` or `f(n + 1)`), so it never reaches the base case",
            "thinks": "the same code with the argument made smaller in the recursive call (`f(n - 1)`)"},
    "D07": {"concept": "recursion",
            "shows": "a recursive function that calls itself as a bare statement (`f(n - 1);`) and then returns "
                     "only its own part (`return n;`), so the result of the call is thrown away",
            "thinks": "the same code where the result of the recursive call is combined into the return value "
                      "(`return n + f(n - 1);`)"},
    "D08": {"concept": "strings",
            "shows": "a `char` compared with a string literal using `==` (for example `s[i] == \"a\"`), on a char "
                     "array declared as `char s[] = \"...\";`",
            "thinks": "the same code with a char literal (`'a'`) in that comparison"},
}

SYSTEM = ("You write short C exercises for first-year programming students. "
          "You reply with one JSON object and nothing else.")

RULES = """Rules for every fragment:
- A fragment is C without main(): zero or more function definitions first, then plain statements. It is run exactly as written.
- Allowed: int, float, char, 1D arrays with initialisers, if/else, for, while, functions, recursion, printf with %d %c %s %.1f, strlen.
- Not allowed: pointers, &, struct, scanf, switch, global variables, #include, #define, comments, do-while, the ?: operator, ++ or -- inside a larger expression.
- The game functions fire(); and open_door(); exist (no arguments, no result). Use them rarely.
- At most 12 lines, at most 60 characters per line, 4-space indentation. One statement per line. Loop and if bodies in braces.
- Keep every value that matters between -99 and 999.
- Except where the mistake itself needs it, never read a variable that has no value and never index outside an array.
- Space-ship names are welcome (fuel, shield, crew, cargo, hull, probe); plain names are fine too."""

SHAPES = {
    "predict_output": """Type: predict the output.
- `code` prints its result with printf as one short line (at most 20 characters).
- `belief_code` prints what the student expects, in the same format.
- `distractors`: two other short outputs a student might guess, in the same format.
JSON shape: {"items": [{"code": "...", "belief_code": "...", "distractors": ["...", "..."], "difficulty": 1}]}""",
    "mcq": """Type: multiple choice about a value.
- `code` prints nothing.
- `question` asks for one value once the code has run, for example "What is total after the loop?" or "What does depth(3) give?".
- `tail` is one printf statement that prints exactly that value. I append it to both fragments, so both must use the same names.
- `distractors`: two other values a student might guess, written the way the tail prints them.
JSON shape: {"items": [{"code": "...", "question": "...", "tail": "printf(...);", "belief_code": "...", "distractors": ["...", "..."], "difficulty": 1}]}""",
    "next_state": """Type: predict the next state.
- `code` has a loop or a recursive function and prints nothing.
- Choose one line of `code` that is a plain assignment statement alone on its line inside braces (never a for, while or if header). `after_line` is its 1-based line number in `code`.
- `var` is an int variable that is in scope on that line. `hit` is 2, 3 or 4.
- The learner is asked: what is `var` right after that line has run for the `hit`-th time?
- `belief_after_line` is the line number of the same statement in `belief_code`.
- At that moment the real value and the value the student expects must differ.
- `distractors`: two other numbers a student might guess.
JSON shape: {"items": [{"code": "...", "var": "...", "after_line": 3, "hit": 2, "belief_code": "...", "belief_after_line": 3, "distractors": ["...", "..."], "difficulty": 1}]}""",
}


def draft_messages(cls, kind):
    info, guide = CLASS_INFO[cls], GUIDE[cls]
    task = f"""Write {PER_CALL[kind]} exercises about one beginner mistake.

The mistake: {info['subtitle']}.
A student who makes it believes: "{info['belief']}".
How it shows in code: {guide['shows']}.

For each exercise write two fragments:
- `code`: what the student is shown. It contains exactly this one mistake and is otherwise ordinary, correct C.
- `belief_code`: the program this student THINKS they are reading: {guide['thinks']}. Apart from that change it is the same as `code`, line for line.
When run, the two fragments must give different results. I run both myself. Do not tell me any result.

{SHAPES[kind]}
`difficulty`: 1 = one idea, 2 = several steps to trace, 3 = tricky.

Make the {PER_CALL[kind]} exercises clearly different from each other: different code shapes, data and names.

{RULES}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def run_drafts(classes):
    jobs = [(cls, kind) for cls in classes for kind in PER_CALL]
    answers = llm_client.chat_json_many("quiz_drafts", [draft_messages(c, k) for c, k in jobs],
                                        model=DRAFT_MODEL, thinking=True, workers=8)
    drafts = json.loads(DRAFTS.read_text(encoding="utf-8")) if DRAFTS.exists() else []
    kept = [d for d in drafts if d["class"] not in classes]
    failed = 0
    for (cls, kind), (parsed, _cached, error) in zip(jobs, answers):
        if error or not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
            failed += 1
            print(f"{cls} {kind}: no usable reply ({error or 'wrong shape'})")
            continue
        for n, item in enumerate(parsed["items"], 1):
            if isinstance(item, dict) and isinstance(item.get("code"), str) and isinstance(item.get("belief_code"), str):
                kept.append({"draft_id": f"{cls}_{kind}_{n}", "class": cls, "type": kind, **item})
    kept.sort(key=lambda d: (MISCONCEPTIONS.index(d["class"]), d["type"], d["draft_id"]))
    DRAFTS.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(kept)} drafts in {DRAFTS.name}; {failed} of {len(jobs)} calls gave nothing")
    print("tokens so far:", llm_client.usage_summary("quiz_drafts"))
    return 1 if failed else 0


def _event_fact(event):
    """One measured event as a plain sentence, or None for events that explain nothing."""
    line, kind = event["line"], event["type"]
    if kind == "uninit_read":
        return f"Line {line} reads {event.get('var')} before it was given a value; C does not define what it holds."
    if kind in ("oob_read", "oob_write"):
        size = event.get("size")
        return (f"Line {line} uses {event.get('arr')}[{event.get('idx')}], but {event.get('arr')} has only {size} "
                f"cells (indexes 0 to {size - 1 if isinstance(size, int) else '?'}); C does not define the result.")
    if kind == "empty_body":
        return (f"Line {line}: the `;` right after the {event.get('kind')} header ends that statement, "
                f"so it controls nothing after it.")
    if kind == "assign_in_cond":
        return (f"Line {line}: the condition stores {event.get('value')} in {event.get('var')} (it is an "
                f"assignment), and that stored value decides the branch.")
    if kind == "discarded_call_value":
        return f"Line {line} calls {event.get('fn')} and throws its result away."
    if kind in ("str_literal_compare", "array_compare"):
        return f"Line {line} compares a char with a string literal; that comparison is false."
    if kind == "intdiv" and event.get("into_float") and event.get("remainder_nonzero"):
        return f"Line {line} divides two ints, so the fraction is dropped before the result is stored in a float."
    return None


def run_facts(item, check):
    """What the interpreter saw when it ran the item's code, as plain sentences for the explainer."""
    from ml.c_interp import harness
    from ml.items import snippet
    problem = {"signature": snippet.SIGNATURE, "tests": [{"args": [], "expect": {}}], "display_test": 0}
    trace = harness.trace(problem, snippet.build(item["code"], check.get("tail") or ""))
    lines = item["code"].split("\n")
    ended = trace["status"] == "ok"
    undefined = any(e["type"] in snippet.UNDEFINED_EVENTS for e in trace["events"])
    facts = ["The run ends normally." if ended else "The run never ends."]
    if ended:
        for key, count in sorted(trace["loop_iters"].items(), key=lambda pair: int(pair[0][1:])):
            if int(key[1:]) <= len(lines):
                facts.append(f"The loop on line {key[1:]} ran its body {count} times.")
    seen = set()
    for event in trace["events"]:
        fact = _event_fact(event) if event["line"] <= len(lines) else None
        if fact and (event["type"], event["line"]) not in seen:
            seen.add((event["type"], event["line"]))
            facts.append(fact)
    own = [step for step in trace["steps"] if step["line"] <= len(lines)]
    if own and not undefined:
        last = {name: value for name, value in own[-1]["vars"].items() if value is not None}
        if ended and last:
            facts.append("Values when the shown code finished: " + ", ".join(f"{k} = {v}" for k, v in last.items()) + ".")
        elif not ended:                 # an endless run: only what has stopped changing is worth telling
            earlier = own[len(own) // 2]["vars"]
            fixed = ", ".join(f"{k} = {v}" for k, v in last.items() if earlier.get(k) == v)
            if fixed:
                facts.append(f"Values that no longer change while it runs: {fixed}.")
    printed = trace["printed"].partition(snippet.MARK)[0]
    if printed and ended and not undefined:
        facts.append(f"The shown code printed: {printed!r}.")
    return facts


def explain_messages(item, check):
    wrong = "; ".join(f"a student who believes \"{CLASS_INFO[c]['belief']}\" expects {answer}"
                      for c, answer in (check.get("belief_answers") or item.get("belief") or {}).items())
    correct = check.get("expect") or item.get("correct")
    facts = "\n".join(f"- {fact}" for fact in run_facts(item, check))
    numbered = "\n".join(f"{n:2}  {line}" for n, line in enumerate(item["code"].split("\n"), 1))
    task = f"""A student was shown this C code. It is the body of a function, so its variables are local, and it runs exactly as written:

{numbered}

Question: {item['question']}
The right answer, found by running the code: {correct}
Common wrong answer: {wrong or 'none recorded'}

Facts measured while the code ran (trust these over your own tracing):
{facts}

Write `explain`: one or two plain sentences (at most 35 words) that say why the right answer is right, by pointing at what this code actually does. Every number and every claim in it must agree with the facts above. Do not mention any other answer as if it were true, do not name the mistake, do not start with "Because".
JSON shape: {{"explain": "..."}}"""
    return [{"role": "system", "content": "You explain C to beginners in plain, exact words. "
                                           "You reply with one JSON object and nothing else."},
            {"role": "user", "content": task}]


def run_explain():
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    checks = json.loads(CHECKS.read_text(encoding="utf-8"))
    answers = llm_client.chat_json_many("quiz_explain",
                                        [explain_messages(item, checks.get(item["item_id"], {})) for item in items],
                                        model=EXPLAIN_MODEL, temperature=0.3, workers=8, thinking=True)
    # keyed by the code and question, so an explanation survives items being renumbered
    out = {}
    missing = 0
    for item, (parsed, _cached, error) in zip(items, answers):
        text = (parsed or {}).get("explain") if not error else None
        if isinstance(text, str) and 3 <= len(text.split()) <= 60:
            out[explain_key(item)] = " ".join(text.split())
        else:
            missing += 1
    EXPLAIN.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"{len(out)} explanations in {EXPLAIN.name}; {missing} calls gave nothing")
    print("tokens so far:", llm_client.usage_summary("quiz_explain"))
    return 1 if missing else 0


def explain_key(item):
    return item["type"] + "|" + " ".join(item["code"].split()) + "|" + item["question"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", choices=["drafts", "explain"])
    parser.add_argument("--classes", nargs="*", default=MISCONCEPTIONS)
    args = parser.parse_args(argv)
    if args.job == "drafts":
        return run_drafts(args.classes)
    return run_explain()


if __name__ == "__main__":
    sys.exit(main())
