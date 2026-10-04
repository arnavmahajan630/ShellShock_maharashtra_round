"""Rule-based diagnoser and report generator for the Black Hole Deep Space Trials.

Diagnoses both foundational C misconceptions (M01, M08) and DSA-specific misconceptions
(D01 Premature Abort, D02 Frozen Window, D03 Cargo Overwrite, D04 Single-Pass Sort,
D05 Endless Warp, D07 Lost Echo, D08 Signal Mismatch).
"""
import re
from typing import Any, Dict, List

from ml.c_interp import harness
from ml.contracts.classes import CLASS_INFO, band
from server.app.diagnosis_common import FLOOR, card, posterior

_PREDICT_CLASS = {
    "T1_two_sum": "M01",
    "T2_binary_search": "D01",
    "T3_bubble_sort": "D03",
    "T4_is_palindrome": "D08",
    "T5_recursive_cascade": "D05",
}

_DSA_EVIDENCE = {
    "D01": "The function returns `-1` on the very first comparison failure inside the loop — the search aborts before examining the remaining elements (Premature Abort).",
    "D02": "The search window does not shrink (`lo = mid` instead of `lo = mid + 1` or `hi = mid`), causing an infinite loop (Frozen Window).",
    "D03": "`a[j] = a[j+1]; a[j+1] = a[j];` overwrites `a[j]` before copying it, losing the original value and duplicating the adjacent crate (Cargo Overwrite).",
    "D04": "The sort uses only a single pass through the array. One pass moves only the maximum element to the end, leaving the rest unsorted (Half-Sorted Hold).",
    "D05": "The recursive call has no reachable base case, calling itself endlessly until hitting the recursion depth cap of 100 (Endless Warp).",
    "D07": "The recursive call's return value is discarded instead of being added to `n` (Lost Echo).",
    "D08": "`==` compares character array memory addresses in C, not string contents (Signal Mismatch). Compare element by element.",
    "M01": "The loop bound extends one element too far (`<= n` instead of `< n`), reading past the end of the array (Boundary Drift).",
    "M08": "Array indexing starts at 0 and ends at n−1. Accessing index n is out of bounds (Index Origin Fault).",
    "M10": "`printf` displays output to the screen but the function returns without handing a value back to the caller (Silent Messenger).",
}

_REFERENCE_FIXES = {
    "T1_two_sum": """int two_sum(int nums[], int n, int target) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (nums[i] + nums[j] == target) {
                return 1;
            }
        }
    }
    return 0;
}""",
    "T2_binary_search": """int binary_search(int nums[], int n, int target) {
    int lo = 0;
    int hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}""",
    "T3_bubble_sort": """void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}""",
    "T4_is_palindrome": """int is_palindrome(char s[]) {
    int n = 0;
    while (s[n] != '\\0') {
        n++;
    }
    int i = 0;
    int j = n - 1;
    while (i < j) {
        if (s[i] != s[j]) {
            return 0;
        }
        i++;
        j--;
    }
    return 1;
}""",
    "T5_recursive_cascade": """int solve_cascade(int n) {
    if (n <= 0) {
        return 0;
    }
    return n + solve_cascade(n - 1);
}""",
}


def reference_fix(problem_id: str) -> Dict[str, Any] | None:
    code = _REFERENCE_FIXES.get(problem_id)
    if not code:
        return None
    return {"code": code, "verified": True}


def diagnose_trial(trial: Dict[str, Any], code: str, predict_answer: Any = None) -> Dict[str, Any]:
    """`predict_answer` is the option index chosen for the trial's predict_item MCQ (D11-style
    comprehension check), if any. Each trial's predict_item names the exact misconception its
    correct option tests for (`_PREDICT_CLASS`) — getting it wrong is itself evidence of that
    misconception, independent of whether the submitted code happens to pass. Previously this
    answer was only logged, never diagnosed, so a wrong MCQ pick never showed up in the report.
    """
    problem_id = trial["problem_id"]
    tests = trial.get("sample_tests", []) + trial.get("hidden_tests", [])
    internal_problem = {
        "problem_id": problem_id,
        "signature": trial["signature"],
        "tests": tests,
        "display_test": 0,
        "forbid": [],
    }

    run = harness.run_tests(internal_problem, code)
    trace = harness.trace(internal_problem, code)
    events = trace.get("events") or []
    passed = run["tests"]["passed"]
    total = run["tests"]["total"]

    def result(status: str, top_id: str | None, p: float, evidence_text: str | None, extra_evidence: list | None = None):
        top = [card(top_id, p)] if top_id and top_id in CLASS_INFO else []
        evidence = [{"type": "RUN", "text": evidence_text}] if evidence_text else []
        evidence += extra_evidence or []
        return {
            "status": status,
            "top": top,
            "evidence": evidence,
            "posterior": posterior(primary=top_id, p_primary=p if top_id else 0.0),
            "passed": passed,
            "total": total,
            "is_correct": passed == total,
        }

    predict_item = trial.get("predict_item") or {}
    predict_wrong = (
        predict_answer is not None and predict_item
        and str(predict_answer) != str(predict_item.get("correct"))
    )
    predict_evidence = (
        [{"type": "YOU PREDICTED", "text": f"You predicted option {predict_answer} for \"{predict_item.get('question')}\"; "
                                            f"the correct option was {predict_item.get('correct')}."}]
        if predict_wrong else []
    )

    if total and passed == total:
        if predict_wrong and problem_id in _PREDICT_CLASS:
            cls = _PREDICT_CLASS[problem_id]
            return result(
                "confident", cls, 0.6,
                f"Your code passes every test, but you predicted the wrong behavior for the classic "
                f"{CLASS_INFO[cls]['name']} pattern this trial is built around — worth reviewing why "
                f"your version avoids it.",
                extra_evidence=predict_evidence,
            )
        return result("correct", "CORRECT", 0.95, None)

    # 1. Trial 1: Two Sum (Arrays)
    if problem_id == "T1_two_sum":
        if any(e["type"] in ("oob_read", "oob_write") for e in events) or re.search(r"<=\s*n\b", code):
            return result("confident", "M01", 0.90, _DSA_EVIDENCE["M01"])
        if re.search(r"for\s*\(\s*(?:int\s+)?\w+\s*=\s*1\s*;", code):
            return result("confident", "M08", 0.88, _DSA_EVIDENCE["M08"])
        # Premature Abort (D01's pattern, applied here): bailing out of the whole search on
        # the first pair that doesn't sum to target, instead of continuing to check the rest.
        if re.search(r"!=\s*target\s*\)\s*\{?\s*return\b", code):
            return result(
                "confident", "D01", 0.90,
                "The function returns as soon as one pair's sum doesn't equal `target` — it gives "
                "up on the very first mismatch instead of continuing to check the remaining pairs "
                "(Premature Abort).",
            )

    # 2. Trial 2: Binary Search (Searching)
    if problem_id == "T2_binary_search":
        # Check premature abort D01
        m_abort = re.search(r"while\s*\([^)]*\)\s*\{[^}]*return\s+-1\s*;", code)
        if m_abort:
            return result("confident", "D01", 0.92, _DSA_EVIDENCE["D01"])
        # Check frozen window D02
        if "lo = mid;" in code or "hi = mid;" in code or any(r.get("got", {}).get("status") == "timeout" for r in run["tests"]["results"]):
            return result("confident", "D02", 0.90, _DSA_EVIDENCE["D02"])

    # 3. Trial 3: Bubble Sort (Sorting)
    if problem_id == "T3_bubble_sort":
        # Check swap overwrite D03
        if re.search(r"a\[\s*\w+\s*\]\s*=\s*a\[\s*\w+\s*\+\s*1\s*\]\s*;\s*a\[\s*\w+\s*\+\s*1\s*\]\s*=\s*a\[\s*\w+\s*\]\s*;", code):
            return result("confident", "D03", 0.94, _DSA_EVIDENCE["D03"])
        for r in run["tests"]["results"]:
            got_arr = r.get("got", {}).get("array0")
            if isinstance(got_arr, list) and len(got_arr) >= 2 and got_arr[0] == got_arr[1]:
                return result("confident", "D03", 0.92, _DSA_EVIDENCE["D03"])
        # Check single pass D04
        for_count = len(re.findall(r"\bfor\b", code))
        if for_count == 1:
            return result("confident", "D04", 0.88, _DSA_EVIDENCE["D04"])

    # 4. Trial 4: Valid Palindrome (Strings)
    if problem_id == "T4_is_palindrome":
        # Check string equality comparison D08
        if re.search(r"==\s*s\b|\bs\s*==|s\s*==\s*rev", code):
            return result("confident", "D08", 0.92, _DSA_EVIDENCE["D08"])
        if any(e["type"] in ("oob_read", "oob_write") for e in events):
            return result("confident", "M08", 0.88, _DSA_EVIDENCE["M08"])

    # 5. Trial 5: Recursive Cascade (Recursion)
    if problem_id == "T5_recursive_cascade":
        # Check depth cap timeout / missing base case D05
        has_timeout = any(r.get("got", {}).get("status") == "timeout" or r.get("got", {}).get("max_depth_le") == 100 for r in run["tests"]["results"])
        if has_timeout or not re.search(r"if\s*\([^)]*<=\s*0|if\s*\([^)]*==\s*0", code):
            return result("confident", "D05", 0.94, _DSA_EVIDENCE["D05"])
        # Check lost echo D07
        if re.search(r"solve_cascade\s*\([^)]*\)\s*;\s*return\s+n\s*;", code):
            return result("confident", "D07", 0.89, _DSA_EVIDENCE["D07"])

    # General checks
    if any(e["type"] == "missing_return" for e in events) and trace.get("printed"):
        return result("confident", "M10", 0.85, _DSA_EVIDENCE["M10"])

    # None of the code-pattern checks above matched (or the bug isn't one of the specific
    # regexes they look for), but a wrong predict answer still names the exact misconception
    # this trial targets — surface that instead of falling through to an unhelpful "novel".
    if predict_wrong and problem_id in _PREDICT_CLASS:
        cls = _PREDICT_CLASS[problem_id]
        return result(
            "confident", cls, 0.70,
            f"Your predicted behavior for this pattern doesn't match {CLASS_INFO[cls]['name']} — "
            f"that's the misconception this trial is built around.",
            extra_evidence=predict_evidence,
        )

    return result("novel", "OTHER", 0.50, "Solution failed one or more test cases.", extra_evidence=predict_evidence)


def compile_debrief(exam_id: str, answers: List[Dict[str, Any]], trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    trial_map = {t["problem_id"]: t for t in trials}
    sectors_map = {
        "arrays": {"sector": "arrays", "name": "Arrays & Indexing", "items": [], "passed": 0, "rating_before": 1400, "rating_after": 1400},
        "searching": {"sector": "searching", "name": "Binary Search", "items": [], "passed": 0, "rating_before": 1400, "rating_after": 1400},
        "sorting": {"sector": "sorting", "name": "Sorting Algorithms", "items": [], "passed": 0, "rating_before": 1400, "rating_after": 1400},
        "strings": {"sector": "strings", "name": "String Traversal", "items": [], "passed": 0, "rating_before": 1400, "rating_after": 1400},
        "recursion": {"sector": "recursion", "name": "Recursion & Trees", "items": [], "passed": 0, "rating_before": 1400, "rating_after": 1400},
    }

    findings = []
    seen_misconceptions = set()
    total_passed = 0
    attempted_sectors = set()

    for ans in answers:
        pid = ans.get("problem_id")
        trial = trial_map.get(pid)
        if not trial:
            continue

        sector_key = trial.get("sector", "arrays")
        attempted_sectors.add(sector_key)
        sec = sectors_map.get(sector_key)
        if sec:
            sec["items"].append(pid)

        code = ans.get("code") or ""
        diag = diagnose_trial(trial, code, ans.get("predict_answer"))

        if diag["is_correct"]:
            total_passed += 1
            if sec:
                sec["passed"] += 1
                sec["rating_after"] += 40
        elif sec:
            sec["rating_after"] = max(1100, sec["rating_after"] - 30)

        # Findings come from the diagnosed class regardless of pass/fail — a predict-item miss
        # on otherwise-passing code is still real evidence of a misconception (diagnose_trial
        # already keeps `is_correct` accurate for the score; this only gates what's reported).
        top = diag.get("top") or []
        if top:
            top_id = top[0]["id"]
            if top_id not in ("CORRECT", "OTHER") and top_id not in seen_misconceptions:
                seen_misconceptions.add(top_id)
                info = CLASS_INFO.get(top_id, {})
                findings.append({
                    "class": top_id,
                    "status": "ACTIVE",
                    "p_active": top[0]["p"],
                    "name": info.get("name", top_id),
                    "subtitle": info.get("subtitle", ""),
                    "belief": info.get("belief", ""),
                    "item_id": pid,
                    "trial_title": trial.get("title", pid),
                    "evidence": diag.get("evidence", []),
                })

    # Only report on sectors the pilot actually attempted — a 1-trial run shouldn't
    # show the other four sectors as "REVIEW" when nothing was tried there.
    attempted_sector_data = [sectors_map[k] for k in sectors_map if k in attempted_sectors]

    # Build recommendations based on weak sectors, scoped to what was attempted. A sector with
    # an active finding needs review even if its code happened to pass (a wrong predict answer
    # on passing code is still a misconception worth sending the pilot back to practice).
    finding_sectors = {trial_map[f["item_id"]].get("sector", "arrays") for f in findings if f["item_id"] in trial_map}
    recommendations = []
    for s_data in attempted_sector_data:
        if s_data["passed"] == 0 or s_data["sector"] in finding_sectors:
            recommendations.append({
                "sector": s_data["sector"],
                "title": f"Review {s_data['name']}",
                "description": f"Focus on core invariants and boundary contracts in {s_data['name']}.",
                "route": "/map",
            })

    if not recommendations:
        recommendations.append({
            "sector": "mastery",
            "title": "All Sectors Mastered!" if len(attempted_sector_data) == len(sectors_map) else "Trial Cleared!",
            "description": "Outstanding performance across all 5 Deep Space Trials."
            if len(attempted_sector_data) == len(sectors_map)
            else "No active misconceptions detected in the trial(s) you attempted.",
            "route": "/map",
        })

    items_total = len(answers) or 1
    return {
        "exam_id": exam_id,
        "items_total": items_total,
        "items_passed": total_passed,
        "score_pct": round((total_passed / items_total) * 100),
        "sectors": attempted_sector_data,
        "findings": findings,
        "recommendations": recommendations,
    }
