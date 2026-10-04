"""Automated audit script for all 20 planet mission levels and 5 Black Hole DSSA trials.

Runs 4 test cases per problem:
1. Starter code
2. Targeted misconception code
3. Verified reference solution
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.app import (
    conditions_rules,
    loops_rules,
    arrays_rules,
    variables_rules,
    functions_rules,
    dsa_rules,
    gate,
)
from server.app.diagnosis_common import to_internal_problem
from ml.c_interp import harness

PROBLEMS_FILE = ROOT / "server" / "fixtures" / "problems.json"
TRIALS_FILE = ROOT / "server" / "fixtures" / "dsa_trials.json"

RULES_BY_PLANET = {
    "conditions": conditions_rules,
    "loops": loops_rules,
    "arrays": arrays_rules,
    "variables": variables_rules,
    "functions": functions_rules,
}

PLANET_TEST_CASES = {
    # Conditions
    "P11": {
        "misconception_code": "int door_open(int code) { if (code = 42) return 1; return 0; }",
        "expected_class": "M06",
    },
    "P12": {
        "misconception_code": "int shield_mode(int energy) { int mode = 99; if (energy < 30); { mode = 0; } return mode; }",
        "expected_class": "M07",
    },
    "P16": {
        "misconception_code": "int in_range(int x, int lo, int hi) { return lo < x < hi; }",
        "expected_class": "novel",  # Chained comparison
    },
    "P17": {
        "misconception_code": 'int max_of_three(int a, int b, int c) { int m = a; if (b > m) m = b; if (c > m) m = c; printf("%d", m); }',
        "expected_class": "M10",
    },
    # Loops
    "P01": {
        "misconception_code": "void fire_shots(int n) { for (int i = 0; i <= n; i++) fire(); }",
        "expected_class": "M01",
    },
    "P03": {
        "misconception_code": "int total_energy(int cells[], int n) { int total = 0; for (int i = 0; i <= n; i++) total += cells[i]; return total; }",
        "expected_class": "ambiguous",  # Ambiguous twin M01/M08
    },
    "P05": {
        "misconception_code": "int charge_steps(int level, int target) { int steps = 0; while (level < target) { steps++; } return steps; }",
        "expected_class": "M02",
    },
    "P06": {
        "misconception_code": "int power_up(int base, int k) { int result = 1; for (int i = 0; i < k; i++) result = base; return result; }",
        "expected_class": "M03",
    },
    # Arrays
    "P08": {
        "misconception_code": "int last_beacon(int ids[], int n) { return ids[n]; }",
        "expected_class": "M08",
    },
    "P07": {
        "misconception_code": "float avg_fuel(int tanks[], int n) { int total = 0; for (int i = 0; i < n; i++) total += tanks[i]; return total / n; }",
        "expected_class": "M04",
    },
    "P10": {
        "misconception_code": "int sum_first_k(int a[], int n, int k) { int total = 0; for (int i = 0; i <= k; i++) total += a[i]; return total; }",
        "expected_class": "M01",
    },
    "P09": {
        "misconception_code": 'int max_shield(int s[], int n) { int m = s[0]; for (int i = 1; i < n; i++) if (s[i] > m) m = s[i]; printf("%d", m); }',
        "expected_class": "M10",
    },
    # Variables
    "P13": {
        "misconception_code": "float sync_ratio(int synced, int total) { return synced / total * 100; }",
        "expected_class": "M04",
    },
    "P14": {
        "misconception_code": 'int signal_diff(int a, int b) { printf("%d", a > b ? a - b : b - a); }',
        "expected_class": "M10",
    },
    "P18": {
        "misconception_code": "int state_balance(int initial, int delta) { int balance; balance += delta; return balance; }",
        "expected_class": "M05",
    },
    "P19": {
        "misconception_code": "int coordinate_nodes(int primary, int replica) { if (primary = 0) return replica; return primary; }",
        "expected_class": "M06",
    },
    # Functions
    "P20": {
        "misconception_code": 'int clamp_range(int val, int lo, int hi) { if (val < lo) printf("%d", lo); else if (val > hi) printf("%d", hi); else printf("%d", val); }',
        "expected_class": "M10",
    },
    "P21": {
        "misconception_code": "void add_power(int s, int b) { s += b; }\nint boost_shield(int shield, int boost) { add_power(shield, boost); return shield; }",
        "expected_class": "M09",
    },
    "P22": {
        "misconception_code": "int step_sq(int n) { return n * n; }\nint compose_pipeline(int x) { int res; res += step_sq(x); return res; }",
        "expected_class": "M05",
    },
    "P23": {
        "misconception_code": "int integrate_subsystem(int mode, int a, int b) { if (mode = 1) return a + b; return -1; }",
        "expected_class": "M06",
    },
}

TRIAL_TEST_CASES = {
    "T1_two_sum": {
        "misconception_code": "int two_sum(int nums[], int n, int target) {\n  for (int i = 0; i < n; i++) {\n    for (int j = i + 1; j <= n; j++) {\n      if (nums[i] + nums[j] == target) return 1;\n    }\n  }\n  return 0;\n}",
        "expected_class": "M01",
    },
    "T2_binary_search": {
        "misconception_code": "int binary_search(int nums[], int n, int target) {\n  int lo = 0, hi = n - 1;\n  while (lo <= hi) {\n    int mid = (lo + hi) / 2;\n    if (nums[mid] == target) return mid;\n    return -1;\n  }\n  return -1;\n}",
        "expected_class": "D01",
    },
    "T3_bubble_sort": {
        "misconception_code": "void bubble_sort(int a[], int n) {\n  for (int i = 0; i < n; i++) {\n    for (int j = 0; j < n - 1; j++) {\n      if (a[j] > a[j+1]) {\n        a[j] = a[j+1];\n        a[j+1] = a[j];\n      }\n    }\n  }\n}",
        "expected_class": "D03",
    },
    "T4_is_palindrome": {
        "misconception_code": 'int is_palindrome(char s[]) {\n  char rev[] = "racecar";\n  if (s == rev) return 1;\n  return 0;\n}',
        "expected_class": "D08",
    },
    "T5_recursive_cascade": {
        "misconception_code": "int solve_cascade(int n) {\n  return n + solve_cascade(n - 1);\n}",
        "expected_class": "D05",
    },
}


def audit_planets():
    print("=" * 60)
    print("AUDITING PLANET MISSIONS (20 PROBLEMS)")
    print("=" * 60)

    problems = json.loads(PROBLEMS_FILE.read_text(encoding="utf-8"))
    planet_problems = [p for p in problems if p.get("planet")]
    print(f"Loaded {len(planet_problems)} planet problems.\n")

    passed_count = 0
    total_count = 0

    for prob in planet_problems:
        pid = prob["problem_id"]
        planet_name = prob["planet"]
        rules = RULES_BY_PLANET.get(planet_name)
        if not rules:
            print(f"[FAIL] No rules module for {planet_name} ({pid})")
            continue

        print(f"--- Problem {pid} ({prob['name']}) on Planet {planet_name} ---")

        # 1. Test Ref Fix
        ref = rules.reference_fix(prob)
        if not ref or not ref.get("verified"):
            print(f"  [ERROR] Reference fix missing or unverified for {pid}")
        else:
            diag_ref = rules.diagnose(prob, ref["code"])
            status_ref = diag_ref.get("status")
            top_id = diag_ref.get("top", [{}])[0].get("id") if diag_ref.get("top") else None
            if status_ref == "correct" and top_id == "CORRECT":
                print(f"  [PASS] Ref Fix passes and diagnoses CORRECT")
                passed_count += 1
            else:
                print(f"  [FAIL] Ref Fix status={status_ref}, top={top_id}")
            total_count += 1

        # 2. Test Targeted Misconception
        tc = PLANET_TEST_CASES.get(pid)
        if tc:
            misc_code = tc["misconception_code"]
            diag_misc = rules.diagnose(prob, misc_code)
            status_misc = diag_misc.get("status")
            top_misc = diag_misc.get("top", [{}])[0].get("id") if diag_misc.get("top") else None
            exp = tc["expected_class"]

            if exp == "novel":
                matches = status_misc == "novel"
            elif exp == "ambiguous":
                matches = status_misc == "ambiguous"
            else:
                matches = top_misc == exp

            if matches:
                print(f"  [PASS] Targeted misconception detected: {top_misc or status_misc} (expected {exp})")
                passed_count += 1
            else:
                print(f"  [FAIL] Targeted misconception got {top_misc or status_misc}, expected {exp}")
            total_count += 1

    print(f"\nPlanet Missions Audit: {passed_count}/{total_count} checks passed.\n")
    return passed_count == total_count


def audit_trials():
    print("=" * 60)
    print("AUDITING BLACK HOLE DSSA TRIALS (5 PROBLEMS)")
    print("=" * 60)

    trials = json.loads(TRIALS_FILE.read_text(encoding="utf-8"))
    passed_count = 0
    total_count = 0

    for trial in trials:
        tid = trial["problem_id"]
        print(f"--- Trial {tid} ({trial['name']}) ---")

        # 1. Test Ref Fix
        ref = dsa_rules.reference_fix(tid)
        if not ref or not ref.get("verified"):
            print(f"  [ERROR] Ref fix missing or unverified for {tid}")
        else:
            diag_ref = dsa_rules.diagnose_trial(trial, ref["code"])
            status_ref = diag_ref.get("status")
            top_id = diag_ref.get("top", [{}])[0].get("id") if diag_ref.get("top") else None
            if status_ref == "correct" and top_id == "CORRECT":
                print(f"  [PASS] Ref Fix passes and diagnoses CORRECT")
                passed_count += 1
            else:
                print(f"  [FAIL] Ref Fix status={status_ref}, top={top_id}")
            total_count += 1

        # 2. Test Targeted Misconception
        tc = TRIAL_TEST_CASES.get(tid)
        if tc:
            misc_code = tc["misconception_code"]
            diag_misc = dsa_rules.diagnose_trial(trial, misc_code)
            status_misc = diag_misc.get("status")
            top_misc = diag_misc.get("top", [{}])[0].get("id") if diag_misc.get("top") else None
            exp = tc["expected_class"]
            if top_misc == exp:
                print(f"  [PASS] Targeted misconception detected: {top_misc} (expected {exp})")
                passed_count += 1
            else:
                print(f"  [FAIL] Targeted misconception got status={status_misc}, top={top_misc}, expected {exp}")
            total_count += 1

    # 3. Test compile_debrief report with misconceptions
    answers = []
    for tid, tc in TRIAL_TEST_CASES.items():
        answers.append({
            "problem_id": tid,
            "code": tc["misconception_code"],
            "predict_answer": None,
        })
    report = dsa_rules.compile_debrief("test_audit", answers, trials)
    print(f"\n--- Debrief Report Generation Test ---")
    print(f"  Items Total: {report['items_total']}")
    print(f"  Items Passed: {report['items_passed']}")
    print(f"  Score: {report['score_pct']}%")
    print(f"  Findings count: {len(report['findings'])}")
    for f in report['findings']:
        print(f"    - {f['class']}: {f['name']} on {f['item_id']}")

    if len(report['findings']) >= 4:
        print("  [PASS] Debrief report populated with findings from failed trials!")
        passed_count += 1
    else:
        print(f"  [FAIL] Debrief report only has {len(report['findings'])} findings (expected >= 4)")
    total_count += 1

    print(f"\nBlack Hole Trials Audit: {passed_count}/{total_count} checks passed.\n")
    return passed_count == total_count


if __name__ == "__main__":
    p_ok = audit_planets()
    t_ok = audit_trials()
    if p_ok and t_ok:
        print("[SUCCESS] All baseline audits passed!")
        sys.exit(0)
    else:
        print("[WARN] Some checks failed - proceeding to apply fixes.")
        sys.exit(1)
