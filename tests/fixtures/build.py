"""Builds the sample data in tests/fixtures/ and the fake API responses in server/fixtures/.

Run from the repo root:  python -m tests.fixtures.build

Everything written here is checked against ml/contracts/schemas.py first. The traces are
produced by small hand-written simulations of four fixed programs, not by the interpreter,
so they show the agreed trace shape before the interpreter exists.
"""
import json
from pathlib import Path

import numpy as np

from ml.contracts import schemas as S
from ml.contracts.classes import CLASS_INFO, LABELS, MISCONCEPTIONS, band
from ml.contracts.feature_names import FEATURES
from ml.contracts.params import EXAM_TIME_LIMIT_S, POPULATION_PRIOR
from ml.contracts.subset import EFFECTS_COUNT_KEYS, GARBAGE, GATE_MESSAGES, STEP_CAP

HERE = Path(__file__).parent
SERVER = HERE.parent.parent / "server" / "fixtures"
MODEL_VERSION = "fixtures"
ENVELOPE = {"model_version": MODEL_VERSION, "latency_ms": 1.0}
G0 = {"code": "G0", "message": GATE_MESSAGES["G0"]}


def load_problem(name):
    return json.loads((HERE / "problems" / name).read_text(encoding="utf-8"))


P03 = load_problem("P03_total_energy.json")
P11 = load_problem("P11_door_open.json")
Q17 = load_problem("Q17_factorial.json")


# ---------------------------------------------------------------- trace recorder

class Recorder:
    """Collects what the interpreter would: steps for one test, counters and events for all."""

    def __init__(self, max_steps=None):
        self.steps, self.events = [], []
        self.loop_iters, self.branch = {}, {}
        self.effects_count = {k: 0 for k in EFFECTS_COUNT_KEYS}
        self.max_depth = 0
        self.executed = 0
        self.max_steps = max_steps
        self.truncated = False
        self.recording = False
        self.test = 0

    def step(self, line, variables, events=(), effects=()):
        self.executed += 1
        events = [dict(e, line=line) for e in events]
        for effect in effects:
            name = effect.split(":")[0]
            self.effects_count[name] = self.effects_count.get(name, 0) + 1
        self.events += [dict(e, test=self.test) for e in events]
        if not self.recording:
            return
        if self.max_steps is not None and len(self.steps) >= self.max_steps:
            self.truncated = True
            return
        self.steps.append({"i": len(self.steps), "line": line, "vars": dict(variables),
                           "events": events, "effects": list(effects)})

    def loop(self, line):
        self.loop_iters[f"L{line}"] = self.loop_iters.get(f"L{line}", 0) + 1

    def take(self, line, outcome):
        counts = self.branch.setdefault(f"B{line}", {"true": 0, "false": 0})
        counts["true" if outcome else "false"] += 1

    def trace(self, status, returned, printed=""):
        return {"status": status, "returned": returned, "printed": printed, "steps": self.steps,
                "loop_iters": self.loop_iters, "branch": self.branch, "events": self.events,
                "effects_count": self.effects_count, "max_depth": self.max_depth,
                "truncated": self.truncated}


def run_all(problem, program, max_steps=None):
    """Run `program(rec, *args)` on every test; record steps on the display test only."""
    rec = Recorder(max_steps)
    outcomes = []
    for index, test in enumerate(problem["tests"]):
        rec.test, rec.recording = index, index == problem["display_test"]
        args = json.loads(json.dumps(test["args"]))     # arrays may be changed in place
        status, returned = program(rec, *args)
        outcomes.append((status, returned, args))
    shown = outcomes[problem["display_test"]]
    worst = next((s for s, _, _ in outcomes if s != "ok"), "ok")
    return rec.trace(worst, shown[1]), outcomes


def tests_object(problem, outcomes):
    results = []
    for test, (status, returned, args) in zip(problem["tests"], outcomes):
        got = {"array0": args[0]} if "array0" in test["expect"] else {"returned": returned}
        expected = {k: v for k, v in test["expect"].items() if k in got}
        results.append({"args": test["args"], "expected": expected, "got": got,
                        "pass": status == "ok" and got == expected})
    return {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}


# ---------------------------------------------------------------- the four programs

P03_LE = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}"""


def p03_for(rec, cells, n, rel_le):
    """total_energy with `i < n` (correct) or `i <= n` (the hard twin)."""
    rec.max_depth = max(rec.max_depth, 1)
    v = {"n": n, "total": 0, "i": None}
    rec.step(2, v, effects=[f"call:total_energy:1:cells,{n}"])
    v["i"] = 0
    rec.step(3, v)
    while True:
        go = v["i"] <= n if rel_le else v["i"] < n
        rec.step(3, v)
        if not go:
            break
        rec.loop(3)
        i = v["i"]
        if i < len(cells):
            v["total"] += cells[i]
            rec.step(4, v, effects=[f"read_cell:{i}"])
        else:
            v["total"] += GARBAGE
            rec.step(4, v, events=[{"type": "oob_read", "arr": "cells", "idx": i, "size": len(cells)}],
                     effects=[f"read_void:{i}"])
        v["i"] += 1
        rec.step(3, v)
    rec.step(6, v, effects=[f"ret:total_energy:1:{v['total']}"])
    return "ok", v["total"]


P03_NO_UPDATE = """int total_energy(int cells[], int n) {
    int total = 0;
    int i = 0;
    while (i < n) {
        total += cells[i];
    }
    return total;
}"""


def p03_no_update(rec, cells, n):
    """total_energy as a while loop whose `i++` is missing: runs into the step cap."""
    rec.max_depth = max(rec.max_depth, 1)
    start = rec.executed
    v = {"n": n, "total": 0, "i": None}
    rec.step(2, v, effects=[f"call:total_energy:1:cells,{n}"])
    v["i"] = 0
    rec.step(3, v)
    while True:
        if rec.executed - start >= STEP_CAP:
            rec.step(4, v, events=[{"type": "step_cap_hit"}])
            return "timeout", None
        rec.step(4, v)
        rec.loop(4)
        v["total"] += cells[0]
        rec.step(5, v, effects=["read_cell:0"])


P11_SEMI = """int door_open(int code) {
    if (code == 42); {
        return 1;
    }
    return 0;
}"""


def p11_semi(rec, code):
    """door_open with a stray `;` after the if: the block below always runs."""
    rec.max_depth = max(rec.max_depth, 1)
    v = {"code": code}
    rec.take(2, code == 42)
    rec.step(2, v, events=[{"type": "empty_body", "kind": "if"}], effects=[f"call:door_open:1:{code}"])
    rec.step(3, v, effects=["ret:door_open:1:1"])
    return "ok", 1


Q17_OK = Q17["correct_variants"][0]


def q17_ok(rec, n, depth=1):
    """Correct recursive factorial (base case n <= 1)."""
    rec.max_depth = max(rec.max_depth, depth)
    v = {"n": n}
    rec.take(2, n <= 1)
    rec.step(2, v, effects=[f"call:factorial:{depth}:{n}"])
    if n <= 1:
        rec.step(3, v, effects=[f"ret:factorial:{depth}:1"])
        return "ok", 1
    _, rest = q17_ok(rec, n - 1, depth + 1)
    rec.step(5, v, effects=[f"ret:factorial:{depth}:{n * rest}"])
    return "ok", n * rest


Q06 = {
    "problem_id": "Q06", "name": "bubble_sort", "sector": "sorting", "difficulty": 2, "world": "cargo_hold",
    "signature": "void bubble_sort(int a[], int n)",
    "starter": "void bubble_sort(int a[], int n) {\n    // TODO\n}",
    "prompt": "Sort the crates by weight, lightest first, in place.",
    "markers": ["i", "j"],
    "display_test": 0,
    "tests": [
        {"args": [[3, 1, 2], 3], "expect": {"array0": [1, 2, 3]}, "sample": True},
        {"args": [[2, 1], 2], "expect": {"array0": [1, 2]}, "sample": True},
        {"args": [[1, 2, 3], 3], "expect": {"array0": [1, 2, 3]}},
        {"args": [[4, 3, 2, 1], 4], "expect": {"array0": [1, 2, 3, 4]}},
    ],
}

Q06_NO_TEMP = """void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                a[j] = a[j + 1];
                a[j + 1] = a[j];
            }
        }
    }
}"""


def q06_no_temp(rec, a, n):
    """Bubble sort that swaps without a temp: a value gets duplicated."""
    rec.max_depth = max(rec.max_depth, 1)
    v = {"n": n, "i": 0, "j": None}
    rec.step(2, v, effects=[f"call:bubble_sort:1:a,{n}"])
    while True:
        rec.step(2, v)
        if not v["i"] < n - 1:
            break
        rec.loop(2)
        v["j"] = 0
        rec.step(3, v)
        while True:
            rec.step(3, v)
            if not v["j"] < n - 1 - v["i"]:
                break
            rec.loop(3)
            j = v["j"]
            swap = a[j] > a[j + 1]
            rec.take(4, swap)
            rec.step(4, v, effects=[f"read_cell:{j}", f"read_cell:{j + 1}", f"compare:{j}:{j + 1}"])
            if swap:
                a[j] = a[j + 1]
                rec.step(5, v, effects=[f"read_cell:{j + 1}", f"write_cell:{j}:{a[j]}"])
                a[j + 1] = a[j]
                rec.step(6, v, effects=[f"read_cell:{j}", f"write_cell:{j + 1}:{a[j + 1]}"])
            v["j"] += 1
            rec.step(3, v)
        v["i"] += 1
        rec.step(2, v)
    return "ok", None


# ---------------------------------------------------------------- diagnosis helpers

def posterior(**given):
    """A full 19-class posterior: the named values, the rest spread evenly."""
    rest = (1.0 - sum(given.values())) / (len(LABELS) - len(given))
    return {k: round(given.get(k, rest), 4) for k in LABELS}


def top(*pairs):
    return [{"id": k, "p": p, "name": CLASS_INFO[k]["name"], "subtitle": CLASS_INFO[k]["subtitle"],
             "band": band(p)} for k, p in pairs]


NOVELTY = {"knn_dist": 1.8, "tau_d": 3.2, "p_max": 0.46, "tau_p": 0.55, "abstain": False}
PROBE_T1_A = {"probe_id": "P_T1_a", "prompt": "Index of the last valid cell?", "code": "int a[5];",
              "options": ["4", "5", "depends on values"], "eig_bits": 0.71}

DIAG_T1 = {
    "status": "ambiguous",
    "posterior": posterior(M01=0.46, M08=0.44, M07=0.03, CORRECT=0.01, OTHER=0.02),
    "top": top(("M01", 0.46), ("M08", 0.44)),
    "twin_set": "T1", "two_bug": False, "novelty": NOVELTY,
    "evidence": [
        {"type": "CODE", "text": "Loop condition uses `<=` (line 3).", "line": 3,
         "feature": "a_main_cond_op_le", "weight": 0.9},
        {"type": "RUN", "text": "Reads `cells[3]`, one cell past the end (line 4).", "line": 4,
         "feature": "b_oob_read_idx_eq_n", "weight": 0.7},
        {"type": "RUN", "text": "This code is identical for both explanations. Asking one question."},
    ],
    "next_probe": PROBE_T1_A, "probes_asked": [], "model_version": MODEL_VERSION,
}

DIAG_T1_AFTER_PROBE = {
    "status": "confident",
    "posterior": posterior(M08=0.90, M01=0.07, CORRECT=0.005, OTHER=0.005),
    "top": top(("M08", 0.90), ("M01", 0.07)),
    "twin_set": "T1", "two_bug": False, "novelty": dict(NOVELTY, p_max=0.90),
    "evidence": DIAG_T1["evidence"][:2] + [
        {"type": "PROBE", "text": "You said the last valid cell of `int a[5]` is 5."}],
    "next_probe": None, "probes_asked": ["P_T1_a"], "model_version": MODEL_VERSION,
}

DIAG_M07 = {
    "status": "confident",
    "posterior": posterior(M07=0.88, M06=0.05, OTHER=0.03),
    "top": top(("M07", 0.88), ("M06", 0.05)),
    "twin_set": None, "two_bug": False, "novelty": dict(NOVELTY, p_max=0.88),
    "evidence": [
        {"type": "CODE", "text": "A `;` right after `if (...)` gives it an empty body (line 2).", "line": 2,
         "feature": "a_empty_body_if", "weight": 0.9},
        {"type": "RUN", "text": "The door opened for every code, not only 42.", "line": 3,
         "feature": "b_branch_always", "weight": 0.6},
    ],
    "next_probe": None, "probes_asked": [], "model_version": MODEL_VERSION,
}

DIAG_D03 = {
    "status": "confident",
    "posterior": posterior(D03=0.88, D04=0.04, OTHER=0.03),
    "top": top(("D03", 0.88), ("D04", 0.04)),
    "twin_set": None, "two_bug": False, "novelty": dict(NOVELTY, p_max=0.88),
    "evidence": [
        {"type": "CODE", "text": "Two cells are assigned to each other with no temp (lines 5-6).", "line": 5,
         "feature": "a_swap_no_temp", "weight": 0.9},
        {"type": "RUN", "text": "After your swap, crate 1 appears twice and 3 is gone.", "line": 6,
         "feature": "b_multiset_changed", "weight": 0.8},
    ],
    "next_probe": None, "probes_asked": [], "model_version": MODEL_VERSION,
}

DIAG_CORRECT = {
    "status": "correct", "posterior": posterior(CORRECT=0.97), "top": top(("CORRECT", 0.97)),
    "twin_set": None, "two_bug": False, "novelty": dict(NOVELTY, p_max=0.97), "evidence": [],
    "next_probe": None, "probes_asked": [], "model_version": MODEL_VERSION,
}


def by(field, cases, default):
    """A fixture whose response depends on one request field (see server/app/main.py)."""
    return {"_by": field, "cases": cases, "default": default}


def public_problem(p):
    item = p.get("predict_item")
    return {
        "problem_id": p["problem_id"], "name": p["name"], "planet": p.get("planet"), "sector": p.get("sector"),
        "difficulty": p["difficulty"], "world": p["world"], "signature": p["signature"],
        "starter": p["starter"], "prompt": p["prompt"],
        "predict_item": {k: item[k] for k in ("item_id", "code", "question", "options", "correct")} if item else None,
        "markers": p["markers"],
        "sample_tests": [{"args": t["args"], "expect": t["expect"]} for t in p["tests"] if t.get("sample")],
    }


def exam_coding_item(p):
    pub = public_problem(p)
    return {"item_id": p["problem_id"], "kind": "coding", "sector": p["sector"], "difficulty": p["difficulty"],
            "prompt": p["prompt"], "signature": p["signature"], "starter": p["starter"],
            "markers": p["markers"], "sample_tests": pub["sample_tests"]}


# ---------------------------------------------------------------- build

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check(model, obj):
    """Validate a response, or every case of a `by` fixture, against its schema."""
    for case in (obj["cases"].values() if "_by" in obj else [obj]):
        model.model_validate(case)
    return obj


def build_traces():
    t1, t1_out = run_all(P03, lambda rec, c, n: p03_for(rec, c, n, rel_le=True))
    ok, ok_out = run_all(P03, lambda rec, c, n: p03_for(rec, c, n, rel_le=False))
    cap, cap_out = run_all(P03, p03_no_update, max_steps=60)
    semi, semi_out = run_all(P11, p11_semi)
    rec, rec_out = run_all(Q17, q17_ok)
    swap, swap_out = run_all(Q06, q06_no_temp)
    traces = {"p03_le_oob_read": t1, "p03_no_update_step_cap": cap, "q17_recursion_ok": rec,
              "p03_ok": ok, "p11_if_semi": semi, "q06_swap_no_temp": swap}
    for name, trace in traces.items():
        S.Trace.model_validate(trace)
    for name in ("p03_le_oob_read", "p03_no_update_step_cap", "q17_recursion_ok"):
        write(HERE / "traces" / f"{name}.json", traces[name])
    tests = {"p03_le_oob_read": tests_object(P03, t1_out), "p03_ok": tests_object(P03, ok_out),
             "p11_if_semi": tests_object(P11, semi_out), "q17_recursion_ok": tests_object(Q17, rec_out),
             "q06_swap_no_temp": tests_object(Q06, swap_out)}
    return traces, tests


REASON_SAMPLES = {
    "M01": ("ctx_trap_m01", "for (i = 2; i <= 6; i++) fire();", "4", [
        "from 2 to 6 is 4 steps so it fires 4 times",
        "6 minus 2 is 4 so four shots",
        "the loop stops before 6 so 2 3 4 5 only",
        "i think <= 6 means it ends at 5",
        "6 - 2 = 4",
        "it goes up to 6 but 6 is not counted",
        "four because the last value is excluded always",
        "loop runs end minus start times na, so 4",
        "2 se 6 tak 4 baar chalega",
        "the bound is where it stops, it does not run for the bound itself",
    ]),
    "M08": ("ctx_trap_m08", "int a[4] = {3,5,7,9};", "9", [
        "a[4] is the fourth element which is 9",
        "the array has 4 cells so cell 4 is the last one",
        "4th value is 9",
        "arrays go 1 2 3 4 so a[4] = 9",
        "last index equals the size",
        "a[1] is 3 so a[4] must be 9",
        "size is 4, last one is a[4]",
        "array 1 se start hota hai to a[4] last hai",
        "counting from one the fourth is nine",
        "because it has four elements and i want the fourth",
    ]),
    "M05": ("ctx_trap_m05", "int c; c++;", "1", [
        "c starts at 0 so after ++ it is 1",
        "new variables are zero",
        "int c gives 0 and then plus one",
        "0 + 1 = 1",
        "it is empty at first which means 0",
        "the default value of an int is 0",
        "c khali hai matlab 0, phir 1",
        "nothing was stored so it counts from zero",
        "declared variables are always initialised to 0 in C",
        "c++ on a fresh variable makes it 1",
    ]),
    "D03": ("ctx_trap_d03", "a = {4, 7}; a[0] = a[1]; a[1] = a[0];", "{7, 4}", [
        "the two lines swap the values so 7 and 4",
        "first takes second and second takes first",
        "they exchange with each other",
        "a[0] becomes 7 and a[1] becomes the old 4",
        "swap ho gaya dono ka",
        "each one gets the other's value",
        "it is a swap so the order flips",
        "a[1] = a[0] gives it the 4 from before",
        "both assignments together mean they trade places",
        "the old value of a[0] goes into a[1]",
    ]),
    "D07": ("ctx_trap_d07", "int fact(int n) { if (n == 1) return 1; fact(n - 1); return n; }", "24", [
        "it calls fact for 3 2 1 and multiplies them all",
        "recursion goes down to 1 and comes back with 24",
        "4 * 3 * 2 * 1",
        "the calls inside add up to the answer on the way back",
        "fact(n-1) is used for the result automatically",
        "factorial of 4 is 24",
        "each call returns into the one above so it builds 24",
        "andar wala call ka answer upar aa jata hai",
        "the function is named fact so it gives the factorial",
        "the recursive call's value gets combined with n",
    ]),
    "CORRECT_REASON": ("ctx_trap_m08", "int a[4] = {3,5,7,9};", "outside the array", [
        "indexes go 0 to 3 so a[4] is past the end",
        "the last valid one is a[3]",
        "size 4 means cells 0 1 2 3 only",
        "a[4] does not exist, it is out of bounds",
        "array 0 se start hota hai, a[4] bahar hai",
        "there are four cells and the fourth is index 3",
        "it reads memory after the array",
        "9 is at a[3] not a[4]",
        "index n is always one past the last cell",
        "valid indexes stop at size minus one",
    ]),
}


HINGLISH_WORDS = {"hai", "hai,", "baar", "dono", "khali", "andar", "tak", "gaya"}


def build_reasons():
    rows = []
    for label, (context, code, chosen, sentences) in REASON_SAMPLES.items():
        for n, text in enumerate(sentences):
            rows.append({"id": f"s-{label}-{n:02d}", "label": label, "context_id": context, "code": code,
                         "chosen": chosen, "text": text,
                         "voice": "hinglish" if set(text.split()) & HINGLISH_WORDS else "plain",
                         "source": "sample", "split": "test" if n >= 8 else "train"})
    for row in rows:
        S.ReasonRow.model_validate(row)
    path = HERE / "reasons_sample.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def build_feature_matrix():
    """A made-up matrix in the real column order, so the model package can run before real data exists."""
    rng = np.random.default_rng(0)
    n = 1900
    y = np.arange(n) % len(LABELS)
    x = (rng.random((n, len(FEATURES))) < 0.08).astype("float32")
    x[np.arange(n), y * 3 % len(FEATURES)] = (rng.random(n) < 0.9)       # one telltale column per class
    x[rng.random(n) < 0.15, FEATURES.index("b_pass_frac"):FEATURES.index("t_has_array_param")] = np.nan
    np.savez_compressed(HERE / "features_synth.npz", X=x, y=y, groups=np.arange(n) % 28,
                        features=np.array(FEATURES), labels=np.array(LABELS))


def build_server(traces, tests):
    problems = [public_problem(P03), public_problem(P11), public_problem(Q17)]
    for p in problems:
        S.PublicProblem.model_validate(p)
    write(SERVER / "problems.json", problems)
    write(SERVER / "problem.json", by("problem_id", {p["problem_id"]: p for p in problems}, "P03"))

    write(SERVER / "learner_create.json", check(S.LearnerCreated, {"learner_id": "demo-learner", **ENVELOPE}))

    knowledge = {k: {"state": "UNSEEN", "p_active": POPULATION_PRIOR} for k in MISCONCEPTIONS}
    knowledge["M01"] = {"state": "STABLE", "p_active": 0.08, "times_seen": 2,
                        "evidence": [{"type": "RUN", "text": "Loop ran 6 times; the mission needed 5."}],
                        "interventions": ["trace_timeline"], "last_tested": "loops:P01", "recheck_queued": True}
    knowledge["M08"] = {"state": "ACTIVE", "p_active": 0.90, "times_seen": 1,
                        "evidence": [{"type": "PROBE", "text": "You said the last valid cell of `int a[5]` is 5."}],
                        "interventions": [], "last_tested": "loops:P03", "recheck_queued": False}
    knowledge["M06"] = {"state": "MASTERED", "p_active": 0.04, "times_seen": 2,
                        "evidence": [{"type": "RUN", "text": "Ghost return P17 passed."}],
                        "interventions": ["trace_timeline"], "last_tested": "conditions:P17", "recheck_queued": False}
    learner = {
        "learner_id": "demo-learner", "callsign": "NOVA", "misconceptions": knowledge,
        "nodes": {"conditions:P11": {"status": "done", "stars": 3}, "conditions:P12": {"status": "done", "stars": 2},
                  "loops:P01": {"status": "done", "stars": 2}, "loops:P03": {"status": "unstable", "stars": 0},
                  "arrays:P08": {"status": "done", "stars": 3}},
        "attempts": [
            {"attempt_id": "at_001", "problem_id": "P11", "ts": "2026-10-04T10:00:00Z", "passed": 5, "total": 5,
             "status": "correct", "top": "CORRECT"},
            {"attempt_id": "at_002", "problem_id": "P01", "ts": "2026-10-04T10:06:00Z", "passed": 0, "total": 4,
             "status": "confident", "top": "M01"},
            {"attempt_id": "at_003", "problem_id": "P03", "ts": "2026-10-04T10:15:00Z", "passed": 0, "total": 5,
             "status": "ambiguous", "top": "M01"},
        ],
        **ENVELOPE,
    }
    write(SERVER / "learner.json", check(S.LearnerResponse, learner))

    write(SERVER / "run.json", check(S.RunResponse, by("problem_id", {
        "P03": {"gate": G0, "trace": traces["p03_ok"], "tests": tests["p03_ok"], **ENVELOPE},
        "Q17": {"gate": G0, "trace": traces["q17_recursion_ok"], "tests": tests["q17_recursion_ok"], **ENVELOPE},
    }, "P03")))

    attempts = {
        "P03": {"attempt_id": "at_fix_t1", "gate": G0, "trace": traces["p03_le_oob_read"],
                "tests": tests["p03_le_oob_read"], "diagnosis": DIAG_T1, **ENVELOPE},
        "P11": {"attempt_id": "at_fix_m07", "gate": G0, "trace": traces["p11_if_semi"],
                "tests": tests["p11_if_semi"], "diagnosis": DIAG_M07, **ENVELOPE},
        "Q06": {"attempt_id": "at_fix_d03", "gate": G0, "trace": traces["q06_swap_no_temp"],
                "tests": tests["q06_swap_no_temp"], "diagnosis": DIAG_D03, **ENVELOPE},
        "Q17": {"attempt_id": "at_fix_ok", "gate": G0, "trace": traces["q17_recursion_ok"],
                "tests": tests["q17_recursion_ok"], "diagnosis": DIAG_CORRECT, **ENVELOPE},
        "GATE": {"attempt_id": "at_fix_gate", "gate": {"code": "G3b", "message": GATE_MESSAGES["G3b"]},
                 "trace": None, "tests": None, "diagnosis": None, **ENVELOPE},
    }
    write(SERVER / "attempt.json", check(S.AttemptResponse, by("problem_id", attempts, "P03")))
    lab = {k: {f: v for f, v in a.items() if f != "attempt_id"} for k, a in attempts.items()}
    write(SERVER / "lab_diagnose.json", check(S.LabDiagnoseResponse, by("problem_id", lab, "P03")))
    write(SERVER / "probe_answer.json", check(S.ProbeAnswerResponse, {"diagnosis": DIAG_T1_AFTER_PROBE, **ENVELOPE}))

    intervention = {
        "class": "M08", "modality": "memory_strip", "next_modalities": ["counterexample", "minimal_fix"],
        "copy": ["Arrays start at cell 0.", "An array of n cells ends at cell n−1; cell n is the void."],
        "question": {"prompt": "Which is the last cell the loop should read?", "answer": 2},
        "timeline": [{"step": s["i"], "line": s["line"], "vars": s["vars"],
                      "effect": s["effects"][0] if s["effects"] else None,
                      "flag": "void_read" if any(e.startswith("read_void") for e in s["effects"]) else None}
                     for s in traces["p03_le_oob_read"]["steps"] if s["line"] == 4],
        "memory_strip": {"array": "cells", "values": [2, 4, 6], "reads": [0, 1, 2, 3]},
        "counterexample": {"input": {"cells": [3], "n": 1}, "intended": 3, "yours": 3 + GARBAGE,
                           "effect_diff": "1 extra void read"},
        "fix": {"kind": "minimal", "code": P03_LE.replace("i <= n", "i < n"), "changed_lines": [3],
                "rule": "<= to <", "verified": True},
        **ENVELOPE,
    }
    write(SERVER / "intervene.json", check(S.InterveneResponse, intervention))

    def conditions(p_ok, trap_ok, transfer_ok, p):
        return [
            {"id": "p_active", "label": "Misconception probability < 0.15", "met": p_ok, "detail": f"{p:.2f}"},
            {"id": "trap", "label": "Trap item passed", "met": trap_ok,
             "detail": "answered: outside the array" if trap_ok else "predicted 9, actual: outside the array"},
            {"id": "transfer", "label": "Different-family transfer passed", "met": transfer_ok,
             "detail": "P09 max_shield" if transfer_ok else "not attempted yet"},
        ]
    write(SERVER / "reassess.json", check(S.ReassessResponse, by("item_type", {
        "trap": {"state": "PROBATION", "p_active": 0.31, "conditions": conditions(False, False, False, 0.31),
                 "resolved_level": None, "next_item": {"item_id": "P09", "item_type": "transfer_code"}, **ENVELOPE},
        "transfer_code": {"state": "STABLE", "p_active": 0.09, "conditions": conditions(True, True, True, 0.09),
                          "resolved_level": "STABLE", "next_item": None, **ENVELOPE},
    }, "transfer_code")))

    # ---- exam: Q06 (coding) -> Q08 (coding) -> xt_rec_1 (trace) -> report
    q06 = exam_coding_item(Q06)
    q08 = {"item_id": "Q08", "kind": "coding", "sector": "sorting", "difficulty": 1,
           "prompt": "Return 1 if the crates are in order, lightest first, otherwise 0.",
           "signature": "int is_sorted(int a[], int n)",
           "starter": "int is_sorted(int a[], int n) {\n    // TODO\n}", "markers": ["i"],
           "sample_tests": [{"args": [[1, 2, 3], 3], "expect": {"returned": 1}},
                            {"args": [[2, 1, 3], 3], "expect": {"returned": 0}}]}
    xt_rec = {"item_id": "xt_rec_1", "kind": "trace", "sector": "recursion", "difficulty": 1,
              "code": "int fact(int n) {\n    if (n == 0) return 1;\n    return n * fact(n - 1);\n}",
              "question": "How many times is fact called for fact(3)?", "options": ["3", "4", "never stops"]}
    write(SERVER / "exam_start.json", check(S.ExamStartResponse, {
        "exam_id": "ex_fixture", "item": q06, "progress": {"k": 1, "n": 3},
        "time_limit_s": EXAM_TIME_LIMIT_S, **ENVELOPE}))
    write(SERVER / "exam_answer.json", check(S.ExamAnswerResponse, by("item_id", {
        "Q06": {"logged": True, "next_item": q08, "progress": {"k": 2, "n": 3}, **ENVELOPE},
        "Q08": {"logged": True, "next_item": xt_rec, "progress": {"k": 3, "n": 3}, **ENVELOPE},
        "xt_rec_1": {"logged": True, "next_item": None, "progress": {"k": 3, "n": 3}, **ENVELOPE},
    }, "Q06")))

    def report(d05):
        return {
            "exam_id": "ex_fixture", "learner_id": "demo-learner", "items_answered": 3, "time_used_s": 412,
            "sectors": [
                {"sector": "sorting", "items": ["Q06", "Q08"], "passed": 1, "rating_before": 1400, "rating_after": 1389},
                {"sector": "recursion", "items": ["xt_rec_1"], "passed": 0, "rating_before": 1400, "rating_after": 1376},
            ],
            "findings": [
                {"class": "D03", "status": "NEW", "p_active": 0.82, "name": CLASS_INFO["D03"]["name"],
                 "subtitle": CLASS_INFO["D03"]["subtitle"],
                 "evidence": [{"item_id": "Q06", "diagnosis": DIAG_D03}]},
                {"class": "M01", "status": "HELD", "p_active": 0.06, "name": CLASS_INFO["M01"]["name"],
                 "subtitle": CLASS_INFO["M01"]["subtitle"], "state_after": "MASTERED",
                 "evidence": [{"item_id": "Q08"}]},
                d05,
            ],
            "deferred_probes": [] if d05["status"] != "UNCERTAIN" else [
                {"probe_id": "P_T7_a", "prompt": "What does h(2) do?", "code": "int h(int n) { return h(n - 1); }",
                 "options": ["0", "never stops", "2"], "for_class": "D05"}],
            "recommendations": [{"class": "D03", "problems": ["Q10", "Q11"]}],
            "adaptivity_log": [
                {"order": 1, "item_id": "Q06", "sector": "sorting", "eig_bits": 0.58,
                 "reason": "Q06 bubble_sort: most informative about Cargo Overwrite and Half-Sorted Hold; covers Sorting.",
                 "outcome": "fail_D03"},
                {"order": 2, "item_id": "Q08", "sector": "sorting", "eig_bits": 0.62,
                 "reason": "Rechecks Boundary Drift (stable since Loops) on a pair loop; covers Sorting.",
                 "outcome": "pass"},
                {"order": 3, "item_id": "xt_rec_1", "sector": "recursion", "eig_bits": 0.41,
                 "reason": "xt_rec_1: checks Endless Warp and Static Warp; covers Recursion.", "outcome": "wrong"},
            ],
        }
    uncertain = {"class": "D05", "status": "UNCERTAIN", "p_active": 0.48, "name": CLASS_INFO["D05"]["name"],
                 "subtitle": CLASS_INFO["D05"]["subtitle"], "twin_set": "T7"}
    sharpened = dict(uncertain, status="NEW", p_active=0.86)
    write(SERVER / "exam_report.json", check(S.ExamReportResponse, {"report": report(uncertain), **ENVELOPE}))
    write(SERVER / "exam_probe.json", check(S.ExamReportResponse, {"report": report(sharpened), **ENVELOPE}))

    # ---- 05: quiz, code items, reasons
    quiz_item = {"item_id": "qz_m01_m08_mcq_01", "type": "mcq", "concept": "loops",
                 "code": P03["predict_item"]["code"], "question": "How many cells does the loop read?",
                 "options": ["3", "4", "2"], "difficulty": 1}
    write(SERVER / "quiz_next.json", check(S.QuizNextResponse, {
        "item": quiz_item, "reason": "Checks Boundary Drift and Index Origin Fault, which your Loops record leaves open.",
        **ENVELOPE}))
    explain = "i takes 0, 1, 2 and 3, so the loop reads four cells; the fourth is past the end of the array."
    write(SERVER / "quiz_answer.json", check(S.QuizAnswerResponse, by("answer", {
        "3": {"correct": False, "correct_answer": "4", "explain": explain, "ask_reason": True,
              "updates": [{"class": "M01", "p_before": 0.10, "p_after": 0.31},
                          {"class": "M08", "p_before": 0.10, "p_after": 0.31}], **ENVELOPE},
        "4": {"correct": True, "correct_answer": "4", "explain": explain, "ask_reason": False,
              "updates": [{"class": "M01", "p_before": 0.10, "p_after": 0.03},
                          {"class": "M08", "p_before": 0.10, "p_after": 0.03}], **ENVELOPE},
    }, "3")))

    holes_starter = ("int total_energy(int cells[], int n) {\n    int total = ____;\n"
                     "    for (int i = ____; i ____ n; i++) total += cells[i];\n    return total;\n}")
    buggy_m05 = P03["correct_variants"][0].replace("int total = 0;", "int total;")
    buggy_m03 = ("int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n"
                 "        total = 0;\n        total += cells[i];\n    }\n    return total;\n}")
    code_items = [
        {"item_id": "cs_P03_01", "type": "complete_snippet", "problem_id": "P03", "starter": holes_starter,
         "holes": [{"id": "h1", "line": 2, "kind": "acc_init", "choices": ["0", "1", "cells[0]"]},
                   {"id": "h2", "line": 3, "kind": "loop_init", "choices": ["0", "1"]},
                   {"id": "h3", "line": 3, "kind": "loop_rel", "choices": ["<", "<=", "!="]}]},
        {"item_id": "fb_P03_m05_01", "type": "fix_bug", "problem_id": "P03", "starter": buggy_m05},
        {"item_id": "dl_P03_m03_01", "type": "debug_line", "problem_id": "P03", "code": buggy_m03,
         "allow_no_bug": True},
    ]
    for item in code_items:
        S.PublicCodeItem.model_validate(item)
    write(SERVER / "code_items.json", code_items)
    fixed_m03 = P03["correct_variants"][0]
    write(SERVER / "code_items_debug.json", check(S.DebugAnswerResponse, by("line", {
        "4": {"correct": True, "bug_lines": [4], "explain": "total is set back to 0 on every pass.",
              "fix": {"code": fixed_m03, "changed_lines": [4]},
              "updates": [{"class": "M03", "p_before": 0.10, "p_after": 0.04}], **ENVELOPE},
        "other": {"correct": False, "bug_lines": [4], "explain": "total is set back to 0 on every pass.",
                  "fix": {"code": fixed_m03, "changed_lines": [4]},
                  "updates": [{"class": "M03", "p_before": 0.10, "p_after": 0.24}], **ENVELOPE},
    }, "other")))

    reason = {"status": "matched", "top": top(("M08", 0.81), ("M01", 0.12)), "reader": "tfidf",
              "updates": [{"class": "M08", "p_before": 0.31, "p_after": 0.62},
                          {"class": "M01", "p_before": 0.31, "p_after": 0.14}], **ENVELOPE}
    write(SERVER / "reason.json", check(S.ReasonResponse, reason))
    write(SERVER / "lab_reason.json", check(S.ReasonResponse, dict(reason, updates=[])))

    write(SERVER / "metrics.json", {
        "stub": True, "model_version": MODEL_VERSION, "generated_at": "2026-10-04T00:00:00Z",
        "effective_n": {"rows": 0, "unique_hash": 0, "unique_prog_op": 0},
        "cards": [{"id": "E1", "title": "Grouped CV by problem", "slice": "TRAIN (stub)", "n": 0,
                   "metrics": {"macro_f1": 0.0, "macro_f1_std": 0.0, "acc": 0.0}, "per_class": {},
                   "plots": [], "caveat": "Stub: no model has been trained yet."}],
        "baselines": [], "domains": {"main": {}, "dsa": {}},
        "cross_domain": {"m_recall_on_dsa_trained_main_only": 0.0, "shortcut_auc": 0.0},
        "exam_sim": {"curves": {"adaptive": [], "fixed": [], "random": []}, "final_f1": {}, "stable_recheck_rate": {}},
        "resolution": {"rows": []}, "failure_audit": [],
    })


def main():
    for name in ("P03_total_energy.json", "P11_door_open.json", "Q17_factorial.json"):
        S.Problem.model_validate(load_problem(name))
    traces, tests = build_traces()
    rows = build_reasons()
    build_feature_matrix()
    build_server(traces, tests)
    print(f"traces: 3 in tests/fixtures/traces, reasons: {len(rows)}, features: {len(FEATURES)} columns, "
          f"server fixtures: {len(list(SERVER.glob('*.json')))}")


if __name__ == "__main__":
    main()
