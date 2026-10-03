"""Self-check of the gcc stand-in (package A2).

    python -m ml.oracle.selfcheck

Runs, on the three sample problems in tests/fixtures/problems/:
  - every correct variant              -> must pass every test
  - deliberately wrong programs        -> must fail exactly the listed tests
  - an endless loop                    -> must come back as "timeout", not hang
  - a compile error, a crash in one test, endless recursion, a sample-only run
and prints the time each run took. Exit code 0 only when every check holds.
"""
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from ml.contracts import schemas
from ml.oracle import gcc

PROBLEMS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "problems"

# (name, problem, code, expected status, indices of the tests that must fail)
CASES = [
    ("wrong: overwrite instead of accumulate (M03 shape; passes tests 1 and 3 by luck)", "P03",
     "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i < n; i++) {\n"
     "        total = cells[i];\n    }\n    return total;\n}", "ok", [0, 2, 4]),
    ("wrong: = instead of == (M06 shape)", "P11",
     "int door_open(int code) {\n    if (code = 42) {\n        return 1;\n    }\n    return 0;\n}",
     "ok", [1, 2, 3, 4]),
    ("wrong: i <= n reads one cell past the end", "P03",
     "int total_energy(int cells[], int n) {\n    int total = 0;\n    for (int i = 0; i <= n; i++) {\n"
     "        total += cells[i];\n    }\n    return total;\n}", "ok", [0, 1, 2, 3, 4]),
    ("endless loop: counter never updated", "P03",
     "int total_energy(int cells[], int n) {\n    int total = 0;\n    int i = 0;\n    while (i < n) {\n"
     "        total += cells[i];\n    }\n    return total;\n}", "timeout", [0, 1, 2, 3, 4]),
    ("endless recursion: no base case", "Q17",
     "int factorial(int n) {\n    return n * factorial(n - 1);\n}", "timeout", [0, 1, 2, 3, 4]),
    ("compile error", "P03",
     "int total_energy(int cells[], int n) {\n    return cells[0] +;\n}", "parse_error", [0, 1, 2, 3, 4]),
    ("crash in one test only (division by zero when code == 7)", "P11",
     "int door_open(int code) {\n    int z = 10 / (code - 7);\n    return code == 42;\n}",
     "runtime_error", [1]),
]


def load_problems():
    return {p["problem_id"]: p for p in (json.loads(f.read_text(encoding="utf-8"))
                                         for f in sorted(PROBLEMS.glob("*.json")))}


def timed(problem, code, **kwargs):
    started = time.perf_counter()
    result = gcc.run_tests(problem, code, **kwargs)
    schemas.RunResult.model_validate(result)
    return result, (time.perf_counter() - started) * 1000


def main():
    problems, failures, times = load_problems(), [], []
    print(f"gcc: {gcc.gcc_path()}")

    def report(ok, label, result, ms, extra=""):
        tests = result["tests"]
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<78} {result['status']:<13} "
              f"{tests['passed']}/{tests['total']}  {ms:7.0f} ms {extra}")
        times.append(ms)
        if not ok:
            failures.append(label)

    print("correct variants (all must pass):")
    for pid, problem in problems.items():
        for k, code in enumerate(problem["correct_variants"]):
            result, ms = timed(problem, code)
            ok = result["status"] == "ok" and result["tests"]["passed"] == result["tests"]["total"]
            report(ok, f"{pid} variant {k}", result, ms)

    print("programs that must not pass:")
    for label, pid, code, status, must_fail in CASES:
        result, ms = timed(problems[pid], code)
        failed = [i for i, r in enumerate(result["tests"]["results"]) if not r["pass"]]
        ok = result["status"] == status and failed == must_fail
        report(ok, label, result, ms, f"failed tests {failed}")
        if status == "timeout" and ms > (gcc.TEST_TIMEOUT_S + 3) * 1000:
            failures.append(f"{label}: took {ms:.0f} ms, limit is {gcc.TEST_TIMEOUT_S} s per test")

    print("sample_only:")
    result, ms = timed(problems["P03"], problems["P03"]["correct_variants"][0], sample_only=True)
    report(result["tests"]["total"] == 2 and result["tests"]["passed"] == 2, "P03 variant 0, sample tests only", result, ms)

    print("cache:")
    result, ms = timed(problems["P03"], problems["P03"]["correct_variants"][0])
    report(result["tests"]["passed"] == 5 and ms < 50, "P03 variant 0 again (answered from the cache)", result, ms)

    leftovers = [d for d in glob.glob(os.path.join(tempfile.gettempdir(), "relearn_gcc_*", "run_*"))]
    print(f"temp folders left by finished runs: {len(leftovers)}")
    if leftovers:
        failures.append("temp folders not cleaned up")

    uncached = sorted(times[:-1])
    print(f"timings over {len(uncached)} uncached runs: median {uncached[len(uncached) // 2]:.0f} ms, "
          f"max {uncached[-1]:.0f} ms (the first run also compiles rl_support.c)")
    if failures:
        print(f"SELFCHECK FAILED: {len(failures)} check(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
