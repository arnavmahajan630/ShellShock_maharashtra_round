"""Student-style test programs written by an LLM: a stand-in for the hand-written R sets.

ml_plan/03 §3.4 asks for realistic programs written by hand. The team could not do that, so
DeepSeek role-plays a first-semester student instead. It is told the student's wrong belief
(the "Wrong belief" text only, never our mutation rules) and a messy personal style.
Rows are marked source "R-llm" and must never be reported as hand-written or as real
student code. A second model then labels every program from the code alone ("rater 2").

Every program is run through the interpreter and compared with a reference solution on a
few inputs (REFERENCE below). A program meant to be wrong is kept only if it differs from the
reference on at least one input; one meant to be correct only if it never differs. A draft
that fails this is rewritten up to two more times, then dropped. This shows a program is
wrong, not that it is wrong for the intended reason; rater 2 and, later, the real problem
bank's tests are the further checks.

Run from the repo root:
    python -m ml.text.gen_realistic_llm --dry-run
    python -m ml.text.gen_realistic_llm
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pycparser import c_parser

from ml.contracts import schemas as S
from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "realistic_llm.jsonl"
META = ROOT / "ml" / "data" / "realistic_llm_meta.json"
AUTHOR_MODEL = "deepseek-v4-pro"
RATER_MODEL = "deepseek-flash"
HOLDOUT = {"P04", "P09", "P13", "Q04", "Q09", "Q16"}
MAX_DRAFTS = 3

# id -> (signature, task, family). Restated from the problem tables of 03 §3.3.
PROBLEMS = {
    "P01": ("void fire_shots(int n)", "Call fire() exactly n times. fire() already exists; do not define it.", "count_loop"),
    "P03": ("int total_energy(int cells[], int n)", "Return the sum of the n values in cells.", "array_accumulate"),
    "P05": ("int charge_steps(int level, int target)", "Each step adds 7 to level. Return how many steps it takes until level is at least target.", "while_progress"),
    "P06": ("int power_up(int base, int k)", "Return base raised to the power k (k >= 0), using a loop.", "accumulate_product"),
    "P07": ("float avg_fuel(int tanks[], int n)", "Return the average of the n values in tanks, with decimals.", "average"),
    "P08": ("int last_beacon(int ids[], int n)", "Return the last of the n values in ids.", "index_access"),
    "P10": ("int sum_first_k(int a[], int n, int k)", "Return the sum of the first k values of a (k <= n).", "prefix_loop"),
    "P11": ("int door_open(int code)", "Return 1 if code is 42, otherwise 0.", "equality_check"),
    "P12": ("int shield_mode(int energy)", "Return 0 if energy is below 30, 1 if it is below 70, otherwise 2.", "branch_bands"),
    "P13": ("float fuel_percent(int fuel, int cap)", "Return fuel as a percentage of cap, with decimals.", "ratio"),
    "P14": ("int distance(int a, int b)", "Return the absolute difference between a and b.", "return_value"),
    "Q01": ("int linear_search(int a[], int n, int x)", "Return the index of x in a, or -1 if it is not there.", "linear_search"),
    "Q02": ("int count_occurrences(int a[], int n, int x)", "Return how many times x appears in a.", "count_match"),
    "Q03": ("int binary_search(int a[], int n, int x)", "a is sorted. Return the index of x using binary search, or -1.", "binary_search"),
    "Q06": ("void bubble_sort(int a[], int n)", "Sort a in place, smallest first, with bubble sort.", "bubble_sort"),
    "Q07": ("void selection_sort(int a[], int n)", "Sort a in place, smallest first, with selection sort.", "selection_sort"),
    "Q08": ("int is_sorted(int a[], int n)", "Return 1 if a is in non-decreasing order, otherwise 0.", "pairwise_check"),
    "Q09": ("int second_largest(int a[], int n)", "Return the second largest value in a (n >= 2, all values different).", "running_max2"),
    "Q10": ("void reverse(int a[], int n)", "Reverse a in place.", "two_pointer_swap"),
    "Q11": ("void rotate_left(int a[], int n)", "Rotate a left by one position in place: the first value moves to the end.", "shift"),
    "Q12": ("int pair_sum_exists(int a[], int n, int target)", "a is sorted. Return 1 if two different cells add up to target, using two indexes that move inward; otherwise 0.", "two_pointer_scan"),
    "Q14": ("int count_vowels(char s[])", "Return how many lowercase vowels (a, e, i, o, u) are in the string s.", "string_count"),
    "Q15": ("int is_palindrome(char s[])", "Return 1 if the string s reads the same backwards, otherwise 0. strlen may be used.", "string_two_pointer"),
    "Q16": ("int sum_digits(int n)", "Return the sum of the digits of n (n >= 0). It must use recursion.", "rec_digits"),
    "Q17": ("int factorial(int n)", "Return n! (factorial(0) is 1). It must use recursion.", "rec_product"),
    "Q18": ("int array_sum_rec(int a[], int n)", "Return the sum of the first n values of a. It must use recursion.", "rec_array"),
}

# Two problems per class, taken from the "Targets" columns of 03 §3.3. D02 uses binary search
# twice: the guessing problem (Q05) has too loose a counting rule to check by running.
CLASS_PROBLEMS = {
    "M01": ["P01", "Q10"], "M02": ["P05", "Q12"], "M03": ["P03", "Q02"], "M04": ["P07", "P13"],
    "M05": ["P06", "Q09"], "M06": ["P11", "Q01"], "M07": ["P12", "Q06"], "M08": ["P08", "Q11"],
    "M10": ["P14", "Q08"], "D01": ["Q01", "Q08"], "D02": ["Q03", "Q03"], "D03": ["Q06", "Q10"],
    "D04": ["Q06", "Q07"], "D05": ["Q17", "Q18"], "D06": ["Q17", "Q16"], "D07": ["Q18", "Q16"],
    "D08": ["Q14", "Q15"],
}

PERSONAS = {
    "hostel_2am": "typing at 2 am in the hostel: names like ans, temp1, x2, a leftover debug printf, one Hinglish comment such as // yaha add karo",
    "youtube_learner": "learned from YouTube: uses while loops where a for would do, declares every variable at the top, keeps an unused variable",
    "over_careful": "over-careful: a comment on almost every line restating it, extra parentheses, one small helper function",
    "cramped": "cramped style: almost no spaces, inconsistent braces, single-letter names, a commented-out earlier attempt",
    "long_way": "does everything the long way: i = i + 1, separate flag variables, redundant else branches, long names like totalValueSoFar",
    "notes_copier": "copied the shape from class notes but renamed things in snake_case, with a couple of Hindi or Marathi comments",
}

RULES = """Rules of the online judge (the student knows them):
- Submit only the function with the given signature (helper functions are allowed). No main().
- No scanf, no pointers (* or &), no struct, no malloc, no #include lines. Arrays use [] only.
- Plain C."""

SYSTEM = ("You role-play first-semester engineering students in India writing C for an online judge. "
          "You stay in character and write exactly the code that student would submit. Reply with a JSON object only.")

CORRECT_STYLES = [
    ("P03", "loops over the array backwards"), ("P06", "handles k == 0 as a special case first, then loops"),
    ("P08", "walks through the whole array keeping the latest value instead of indexing the last cell"),
    ("P12", "nested ifs instead of else-if"), ("Q01", "uses a found-index variable and break instead of returning inside the loop"),
    ("Q03", "computes mid as low + (high - low) / 2 and loops while low <= high"),
    ("Q06", "uses a swapped flag to stop early"), ("Q10", "uses two indexes i and j that move toward each other"),
    ("Q15", "builds the length with its own loop instead of strlen"), ("Q17", "uses n <= 1 as the base case and a local variable for the result"),
]
OTHER_SLIPS = [
    ("P03", "uses -= where += was meant"), ("P14", "returns a - b without making it positive"),
    ("P12", "mistypes one of the two limits (for example 60 instead of 70)"),
    ("Q02", "compares each cell with n instead of x"), ("Q08", "flips the comparison so it accepts descending order"),
    ("Q17", "returns 0 in the base case instead of 1"),
]
TWIN_PROBLEMS = ["P03", "P10", "Q01", "Q06"]
JUNK = [
    ("P03", "G3b", "writes the whole solution in Python out of habit"),
    ("P11", "G1", "submits only comments describing the plan, with no code at all"),
    ("P08", "G6", "hard-codes the answer from the sample test (returns a fixed number and ignores the parameters)"),
    ("P14", "G3c", "writes C++ with cout and #include <iostream>"),
    ("Q17", "G5b", "ignores the recursion requirement and solves it with a loop only"),
    ("P03", "G4", "uses pointer arithmetic such as *(cells + i)"),
]

# id -> (correct solution, what to compare, inputs). Written for the run check only; the real
# problem bank (packages B1, B2) has its own variants and tests.
_BUBBLE = ("void {name}(int a[], int n) {{ for (int i = 0; i < n - 1; i++) for (int j = 0; j < n - 1 - i; j++) "
           "if (a[j] > a[j + 1]) {{ int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }} }}")
_SORT_INPUTS = [[[3, 1, 2], 3], [[2, 1], 2], [[1, 2, 3], 3], [[4, 3, 2, 1], 4], [[5, 1, 4, 2, 8], 5], [[7], 1]]
REFERENCE = {
    "P01": ("void fire_shots(int n) { for (int i = 0; i < n; i++) fire(); }", "fire", [[0], [1], [3], [5]]),
    "P03": ("int total_energy(int cells[], int n) { int t = 0; for (int i = 0; i < n; i++) t += cells[i]; return t; }",
            "returned", [[[2, 4, 6], 3], [[5], 1], [[1, 2, 3, 4], 4], [[0, 0, 9], 3], [[7, 1], 2]]),
    "P05": ("int charge_steps(int level, int target) { int s = 0; while (level < target) { level += 7; s++; } return s; }",
            "returned", [[0, 7], [0, 20], [10, 10], [3, 50], [5, 6]]),
    "P06": ("int power_up(int base, int k) { int r = 1; for (int i = 0; i < k; i++) r *= base; return r; }",
            "returned", [[2, 3], [5, 0], [3, 1], [2, 10], [7, 2]]),
    "P07": ("float avg_fuel(int tanks[], int n) { int s = 0; for (int i = 0; i < n; i++) s += tanks[i]; return (float)s / n; }",
            "returned", [[[1, 2], 2], [[3, 4, 4], 3], [[10], 1], [[1, 2, 3, 4], 4], [[5, 5], 2]]),
    "P08": ("int last_beacon(int ids[], int n) { return ids[n - 1]; }",
            "returned", [[[7, 8, 9], 3], [[4], 1], [[1, 2, 3, 4, 5], 5], [[9, 3], 2]]),
    "P10": ("int sum_first_k(int a[], int n, int k) { int s = 0; for (int i = 0; i < k; i++) s += a[i]; return s; }",
            "returned", [[[1, 2, 3, 4], 4, 2], [[5, 6, 7], 3, 3], [[9, 1, 1], 3, 1], [[2, 2, 2, 2], 4, 0]]),
    "P11": ("int door_open(int code) { if (code == 42) return 1; return 0; }", "returned", [[42], [7], [0], [41], [-42]]),
    "P12": ("int shield_mode(int energy) { if (energy < 30) return 0; if (energy < 70) return 1; return 2; }",
            "returned", [[10], [29], [30], [65], [69], [70], [100], [0]]),
    "P13": ("float fuel_percent(int fuel, int cap) { return (float)fuel / cap * 100; }",
            "returned", [[30, 40], [1, 3], [50, 50], [0, 10], [7, 8]]),
    "P14": ("int distance(int a, int b) { if (a > b) return a - b; return b - a; }",
            "returned", [[3, 8], [8, 3], [5, 5], [-2, 4], [0, 9]]),
    "Q01": ("int linear_search(int a[], int n, int x) { for (int i = 0; i < n; i++) { if (a[i] == x) return i; } return -1; }",
            "returned", [[[4, 7, 1, 9], 4, 7], [[4, 7, 1, 9], 4, 4], [[4, 7, 1, 9], 4, 9], [[4, 7, 1, 9], 4, 5],
                         [[3], 1, 3], [[1, 2, 3], 3, 3]]),
    "Q02": ("int count_occurrences(int a[], int n, int x) { int c = 0; for (int i = 0; i < n; i++) { if (a[i] == x) c++; } return c; }",
            "returned", [[[1, 2, 1, 3, 1], 5, 1], [[1, 2, 1, 3, 1], 5, 2], [[1, 2, 1, 3, 1], 5, 9], [[4, 4], 2, 4],
                         [[5], 1, 5], [[3, 5, 7], 3, 3]]),
    "Q03": ("int binary_search(int a[], int n, int x) { int low = 0, high = n - 1; while (low <= high) { int mid = (low + high) / 2; "
            "if (a[mid] == x) return mid; if (a[mid] < x) low = mid + 1; else high = mid - 1; } return -1; }",
            "returned", [[[1, 3, 5, 7, 9], 5, 7], [[1, 3, 5, 7, 9], 5, 1], [[1, 3, 5, 7, 9], 5, 9], [[1, 3, 5, 7, 9], 5, 4],
                         [[2, 4], 2, 4], [[2], 1, 2], [[1, 3, 5, 7], 4, 3], [[1, 3, 5, 7], 4, 8], [[1, 3, 5, 7], 4, 0]]),
    "Q06": (_BUBBLE.format(name="bubble_sort"), "array0", _SORT_INPUTS),
    "Q07": (_BUBBLE.format(name="selection_sort"), "array0", _SORT_INPUTS),
    "Q08": ("int is_sorted(int a[], int n) { for (int i = 0; i < n - 1; i++) { if (a[i] > a[i + 1]) return 0; } return 1; }",
            "returned", [[[1, 2, 3], 3], [[2, 1, 3], 3], [[1, 3, 2], 3], [[5], 1], [[1, 1, 2], 3], [[1, 2, 3, 0], 4], [[3, 2, 1], 3]]),
    "Q09": ("int second_largest(int a[], int n) { int big, second; if (a[0] > a[1]) { big = a[0]; second = a[1]; } "
            "else { big = a[1]; second = a[0]; } for (int i = 2; i < n; i++) { if (a[i] > big) { second = big; big = a[i]; } "
            "else if (a[i] > second) second = a[i]; } return second; }",
            "returned", [[[4, 9, 2], 3], [[1, 2], 2], [[9, 8, 7, 6], 4], [[3, 5, 4, 1], 4], [[10, 20, 15], 3], [[-5, -2, -9], 3]]),
    "Q10": ("void reverse(int a[], int n) { for (int i = 0; i < n / 2; i++) { int t = a[i]; a[i] = a[n - 1 - i]; a[n - 1 - i] = t; } }",
            "array0", [[[1, 2, 3], 3], [[1, 2, 3, 4], 4], [[5], 1], [[1, 2], 2], [[9, 8, 7, 6, 5], 5]]),
    "Q11": ("void rotate_left(int a[], int n) { int first = a[0]; for (int i = 0; i < n - 1; i++) a[i] = a[i + 1]; a[n - 1] = first; }",
            "array0", [[[1, 2, 3], 3], [[5, 6, 7, 8], 4], [[4], 1], [[1, 2], 2]]),
    "Q12": ("int pair_sum_exists(int a[], int n, int target) { int i = 0, j = n - 1; while (i < j) { int s = a[i] + a[j]; "
            "if (s == target) return 1; if (s < target) i++; else j--; } return 0; }",
            "returned", [[[1, 2, 4, 7], 4, 9], [[1, 2, 4, 7], 4, 8], [[1, 2, 4, 7], 4, 3], [[1, 2, 4, 7], 4, 10],
                         [[1, 2, 4, 7], 4, 2], [[1, 3], 2, 4], [[2, 5, 9], 3, 7], [[2, 5, 9], 3, 6]]),
    "Q14": ("int count_vowels(char s[]) { int c = 0; for (int i = 0; s[i] != '\\0'; i++) { "
            "if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u') c++; } return c; }",
            "returned", [["hello"], ["sky"], ["aeiou"], ["banana"], [""]]),
    "Q15": ("int is_palindrome(char s[]) { int n = 0; while (s[n] != '\\0') n++; for (int i = 0; i < n / 2; i++) { "
            "if (s[i] != s[n - 1 - i]) return 0; } return 1; }",
            "returned", [["level"], ["abca"], ["a"], ["ab"], ["abba"], ["abcba"], ["abcda"]]),
    "Q16": ("int sum_digits(int n) { if (n < 10) return n; return n % 10 + sum_digits(n / 10); }",
            "returned", [[0], [5], [25], [123], [909], [10]]),
    "Q17": ("int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }", "returned", [[0], [1], [3], [5], [10]]),
    "Q18": ("int array_sum_rec(int a[], int n) { if (n == 0) return 0; return a[n - 1] + array_sum_rec(a, n - 1); }",
            "returned", [[[1, 2, 3], 3], [[5], 1], [[4, 4, 4, 4], 4], [[2, 7], 2], [[9, 1, 1], 3], [[1, 2], 0]]),
}

# Final labels where the author's intended label and rater 2 disagreed and a third reading
# (by the person assembling the set) sided against the intended label. Keyed by ast_hash.
ADJUDICATED = {
    "t_ab3e250b64dc": ("OTHER", "meant as D01, but the code never sets the flag to 0 at all: no early exit and no "
                     "overwrite of an earlier match, so it is not that belief"),
}

REDO_NOTE = {
    "wrong": "\n\n(Researcher's note, draft {n}: an earlier draft behaved correctly on every input, so it did not show the "
             "student's thinking. Trace the code by hand on a small input and make sure it really goes wrong.)",
    "correct": "\n\n(Researcher's note, draft {n}: an earlier draft gave a wrong result on some input. "
               "This student's code has to be correct on every input.)",
    "broken": "\n\n(Researcher's note, draft {n}: an earlier draft did not compile under the judge's rules: {why}. "
              "Keep to the rules.)",
}


def entry_name(problem_id):
    return PROBLEMS[problem_id][0].split("(")[0].split()[-1]


def task_text(problem_id):
    signature, task, _ = PROBLEMS[problem_id]
    return f"Task: {task}\nSignature: {signature}"


def student_prompt(problem_id, persona, mind, extra=""):
    user = f"""{task_text(problem_id)}

{RULES}

The student: {PERSONAS[persona]}.
{mind}
{extra}
Write the code this student submits, with their habits visible. Do not write comments that admit or point at a mistake.

Return JSON: {{"code": "<the full submission>", "note": "<one line for the researcher: what in the code comes from the student's thinking>"}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def plan():
    """Every program to write: dicts with kind, label, problem_id, persona, messages."""
    names = list(PERSONAS)
    items = []

    def add(kind, label, problem_id, mind, extra="", **more):
        persona = names[len(items) % len(names)]
        items.append({"kind": kind, "label": label, "problem_id": problem_id, "persona": persona,
                      "messages": student_prompt(problem_id, persona, mind, extra), **more})

    for cls in MISCONCEPTIONS:
        for problem_id in CLASS_PROBLEMS[cls]:
            add("misconception", cls, problem_id,
                f"The student truly believes this about C, and nobody has corrected them: {CLASS_INFO[cls]['belief']}.",
                "Because of that belief the submission is wrong: it gives a wrong result (or never finishes) on at "
                "least some inputs. It has no other bug.")
    for problem_id, style in CORRECT_STYLES:
        add("correct_unusual", "CORRECT", problem_id,
            f"The student understands C well. Their solution is correct on every input but unusual: it {style}.")
    for problem_id, slip in OTHER_SLIPS:
        add("other", "OTHER", problem_id,
            f"The student understands C but made one careless slip: the code {slip}.",
            "Apart from that slip the code is fine.")
    for problem_id in TWIN_PROBLEMS:
        add("hard_twin", "M01", problem_id,
            "The student's loop over the array runs one step too far, so it touches one cell past the end.",
            "That is the only bug.", soft_label={"M01": 0.5, "M08": 0.5})
    for problem_id, gate, behaviour in JUNK:
        add("junk", "GATE", problem_id, f"This student {behaviour}.", "(The judge rules are ignored here.)", gate=gate)
    return items


def run_outputs(problem_id, code):
    """(overall status, one (status, output) per input) from the interpreter."""
    from ml import runner
    _, key, inputs = REFERENCE[problem_id]
    problem = {"problem_id": problem_id, "name": entry_name(problem_id), "signature": PROBLEMS[problem_id][0],
               "display_test": 0, "forbid": [], "tests": [{"args": args, "expect": {key: None}} for args in inputs]}
    result = runner.run_tests(problem, code, backend="interp")
    return result["status"], [(r["got"].get("status", "ok"), r["got"].get(key)) for r in result["tests"]["results"]]


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= 1e-3
    return a == b


def run_check(kind, problem_id, code):
    """(redo key, reason) if the program does not behave as its kind requires, else None. Junk is not run."""
    if kind == "junk":
        return None
    if not re.search(rf"\b{entry_name(problem_id)}\s*\(", code):
        return "broken", f"no function named {entry_name(problem_id)}"
    status, outputs = run_outputs(problem_id, code)
    if status in ("parse_error", "unsupported"):
        return "broken", f"the judge reports {status}"
    _, expected = run_outputs(problem_id, REFERENCE[problem_id][0])
    differs = any(s != es or not same(v, ev) for (s, v), (es, ev) in zip(outputs, expected))
    if kind == "correct_unusual":
        return ("correct", "gives a wrong result on some input") if differs else None
    return None if differs else ("wrong", "behaves correctly on every input, so it shows no bug")


RATER_SYSTEM = "You label student C submissions for a programming course. Reply with a JSON object only."


def rater_prompt(code, problem_id):
    classes = "\n".join(f"- {k}: {CLASS_INFO[k]['subtitle']}. Wrong belief: {CLASS_INFO[k]['belief']}" for k in MISCONCEPTIONS)
    user = f"""{task_text(problem_id)}

Submission:
{code}

Choose exactly one label for this submission from the code alone:
{classes}
- CORRECT: it solves the task on every input.
- OTHER: it is wrong, but not because of any belief above.
- GATE: it is not a valid answer in the allowed C (wrong language, empty, hard-coded, uses pointers or scanf, no recursion where required).

Return JSON: {{"label": "<one id from the list>"}}"""
    return [{"role": "system", "content": RATER_SYSTEM}, {"role": "user", "content": user}]


def kappa(a, b):
    """Cohen's kappa between two label lists."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in set(a) | set(b)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def write_programs(items, llm_client):
    """Draft, run, and redraft. Returns (rows, meta, dropped)."""
    rows, meta, dropped = [], {}, []
    pending = [dict(item, base=item["messages"][1]["content"]) for item in items]
    for draft in range(1, MAX_DRAFTS + 1):
        written = llm_client.chat_json_many("realistic", [i["messages"] for i in pending], temperature=1.0,
                                            workers=6, model=AUTHOR_MODEL, thinking=True)
        again = []
        for item, (parsed, _, error) in zip(pending, written):
            has_code = not error and isinstance(parsed, dict) and isinstance(parsed.get("code"), str) and parsed["code"].strip()
            code = parsed["code"].strip("\n") if has_code else ""
            problem = ("broken", "no code came back") if not has_code else run_check(item["kind"], item["problem_id"], code)
            if problem:
                if draft < MAX_DRAFTS:
                    note = REDO_NOTE[problem[0]].format(n=draft + 1, why=problem[1])
                    again.append(dict(item, messages=[item["messages"][0], {"role": "user", "content": item["base"] + note}]))
                else:
                    dropped.append([item["label"], item["problem_id"], item["kind"], problem[1]])
                continue
            row_id = f"RL-{len(rows) + 1:03d}"
            problem_id = item["problem_id"]
            rows.append({
                "id": row_id, "source": "R-llm", "problem_id": problem_id, "family": PROBLEMS[problem_id][2],
                "split": "holdout_problem" if problem_id in HOLDOUT else "train", "code": code,
                "label": item["label"], "soft_label": item.get("soft_label"),
                "labels_all": sorted(item["soft_label"]) if "soft_label" in item else [item.get("gate") or item["label"]],
                "ast_hash": "t_" + hashlib.sha1("".join(code.split()).encode()).hexdigest()[:12],
                "author": f"{AUTHOR_MODEL} as {item['persona']}", "rater2_label": None,
            })
            meta[row_id] = {"kind": item["kind"], "persona": item["persona"], "note": parsed.get("note", ""),
                            "drafts": draft, "run_check": "not run (junk item)" if item["kind"] == "junk" else "passed"}
        print(f"draft {draft}: {len(pending)} written, {len(again)} to rewrite")
        pending = again
        if not pending:
            break
    return rows, meta, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    items = plan()
    print(f"{len(items)} programs planned:", dict(Counter(i["kind"] for i in items)))
    if args.dry_run:
        print(items[0]["messages"][1]["content"])
        return
    if args.limit:
        items = items[:args.limit]

    from ml.text import llm_client
    rows, meta, dropped = write_programs(items, llm_client)

    rated = llm_client.chat_json_many("realistic_rater", [rater_prompt(r["code"], r["problem_id"]) for r in rows],
                                      temperature=0.0, workers=6, model=RATER_MODEL)
    for row, (parsed, _, error) in zip(rows, rated):
        row["rater2_label"] = parsed.get("label") if isinstance(parsed, dict) and not error else None
    intended = [r["label"] for r in rows]       # what the author was asked for, before any relabelling
    second = ["M01" if r["soft_label"] and r["rater2_label"] in r["soft_label"] else (r["rater2_label"] or "none") for r in rows]
    agree = sum(x == y for x, y in zip(intended, second))
    print(f"kept {len(rows)} of {len(items)}; dropped after {MAX_DRAFTS} drafts: {len(dropped)}")
    for entry in dropped:
        print("   dropped:", entry)
    print("kept by kind:", dict(Counter(v["kind"] for v in meta.values())),
          "| kept on a later draft:", sum(v["drafts"] > 1 for v in meta.values()))
    print(f"rater 2 agrees with the intended label on {agree}/{len(rows)} ({agree / len(rows):.0%}); kappa {kappa(intended, second):.2f}")
    for row, got in zip(rows, second):
        if row["label"] != got:
            print(f"   {row['id']} {row['problem_id']} ({meta[row['id']]['kind']}): intended {row['label']}, rater 2 said {got}")
    for row in rows:
        if row["ast_hash"] in ADJUDICATED:
            label, why = ADJUDICATED[row["ast_hash"]]
            meta[row["id"]]["relabelled"] = {"from": row["label"], "to": label, "why": why}
            row["label"], row["labels_all"] = label, [label]
            print(f"   {row['id']} relabelled to {label}: {why}")
        S.DatasetRow.model_validate(row)
    print("final labels:", dict(Counter(r["label"] for r in rows)))
    print("tokens:", {job: llm_client.usage_summary(job) for job in ("realistic", "realistic_rater")})
    if args.limit:
        return

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    META.write_text(json.dumps({
        "about": "LLM-written stand-in for the hand-written realistic set. Not hand-written, not real student code. "
                 "Rater 2 is also an LLM. Each kept program was run against a reference solution (see run_check); "
                 "that shows it is wrong or right, not that it is wrong for the intended reason.",
        "author_model": AUTHOR_MODEL, "rater_model": RATER_MODEL, "planned": len(items), "kept": len(rows),
        "agreement": agree / len(rows), "kappa": kappa(intended, second), "dropped": dropped, "items": meta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
