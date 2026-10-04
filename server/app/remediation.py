"""Remediation and Misconception Clearance Guide Engine.

Provides concrete, actionable guidance to permanently clear identified programming misconceptions.
Uses LLM (Gemini) when available to personalize guidance to student code, with deterministic,
verified expert templates as fallback.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ml.contracts.classes import CLASS_INFO
from server.app.multimodal.llm import call_llm, is_live_available

log = logging.getLogger("relearn.remediation")

EXPERT_REMEDIATIONS: Dict[str, Dict[str, Any]] = {
    "M01": {
        "root_cause": "Off-by-one in loop condition: confusing <= with < when indexing from 0.",
        "rule_to_remember": "An array of size n has indices 0 through n-1. Loop condition must be i < n, NOT i <= n.",
        "code_fix": {
            "wrong": "for (int i = 0; i <= n; i++) {\n    sum += arr[i]; // Accesses arr[n] (out of bounds!)\n}",
            "right": "for (int i = 0; i < n; i++) {\n    sum += arr[i]; // Safely stops at arr[n-1]\n}",
        },
        "self_check": "Test with n = 1: does the loop body execute exactly 1 time?",
    },
    "M02": {
        "root_cause": "The loop control variable never moves toward the exit condition, locking execution in an infinite loop.",
        "rule_to_remember": "Every loop must modify at least one variable involved in its stopping condition inside the body.",
        "code_fix": {
            "wrong": "while (i < n) {\n    process(arr[i]); // i never increases!\n}",
            "right": "while (i < n) {\n    process(arr[i]);\n    i++; // Advances toward exit\n}",
        },
        "self_check": "Trace the first two passes: does the stopping condition evaluate closer to false on step 2?",
    },
    "M03": {
        "root_cause": "Using plain assignment = instead of += when accumulating a running total, resetting progress each pass.",
        "rule_to_remember": "To accumulate, add to the previous total: sum += val, never sum = val.",
        "code_fix": {
            "wrong": "for (int i = 0; i < n; i++) {\n    total = arr[i]; // Overwrites total each step\n}",
            "right": "for (int i = 0; i < n; i++) {\n    total += arr[i]; // Accumulates into running sum\n}",
        },
        "self_check": "Trace steps 0 and 1: does total equal arr[0] + arr[1]?",
    },
    "M04": {
        "root_cause": "Dividing two integers (a / b) in C drops any remainder, truncating toward zero.",
        "rule_to_remember": "In C, integer / integer = integer. Cast at least one operand to float or double before dividing.",
        "code_fix": {
            "wrong": "float avg = total / n; // 7 / 2 evaluates to 3.0",
            "right": "float avg = (float)total / n; // Evaluates to 3.5",
        },
        "self_check": "Mental test: 5 / 2 in C evaluates to 2; (double)5 / 2 evaluates to 2.5.",
    },
    "M05": {
        "root_cause": "Reading a local variable before assigning it an initial value; local variables contain random memory garbage.",
        "rule_to_remember": "Local variables in C do NOT default to 0. Always initialize accumulators: int sum = 0;.",
        "code_fix": {
            "wrong": "int sum;\nfor (int i = 0; i < n; i++) sum += arr[i]; // Adds to garbage",
            "right": "int sum = 0;\nfor (int i = 0; i < n; i++) sum += arr[i]; // Clean start",
        },
        "self_check": "Ask: what is this variable's exact value on line 1 before any loop or function executes?",
    },
    "M06": {
        "root_cause": "Using single = (assignment) instead of double == (equality comparison) inside if (...).",
        "rule_to_remember": "= stores a value and evaluates truthy whenever non-zero; == tests if two values are equal.",
        "code_fix": {
            "wrong": "if (code = 42) {\n    shield = 1; // Assigns 42 to code; always evaluates true!\n}",
            "right": "if (code == 42) {\n    shield = 1; // Compares code to 42\n}",
        },
        "self_check": "Turn on compiler warnings (-Wall): gcc flags 'suggest parentheses around assignment used as truth value'.",
    },
    "M07": {
        "root_cause": "A stray semicolon right after an if/for/while header creates an empty body, leaving the block to run unconditionally.",
        "rule_to_remember": "Never place a semicolon directly after the closing parenthesis of if, for, or while.",
        "code_fix": {
            "wrong": "if (shield_power > 50);\n{\n    cycle_hatch(); // Runs unconditionally!\n}",
            "right": "if (shield_power > 50) {\n    cycle_hatch(); // Gated by condition\n}",
        },
        "self_check": "Inspect each if/for line: does it end with { or an immediate statement, never ; ?",
    },
    "M08": {
        "root_cause": "Assuming C arrays are 1-indexed (1 to n) rather than 0-indexed (0 to n-1).",
        "rule_to_remember": "C arrays always start at index 0. The last valid element of array arr[n] is arr[n-1].",
        "code_fix": {
            "wrong": "for (int i = 1; i <= n; i++) {\n    process(arr[i]); // Misses arr[0], crashes at arr[n]\n}",
            "right": "for (int i = 0; i < n; i++) {\n    process(arr[i]); // Covers arr[0] through arr[n-1]\n}",
        },
        "self_check": "An array of 5 elements has slots [0, 1, 2, 3, 4]. Slot [5] is invalid.",
    },
    "M09": {
        "root_cause": "Expecting scalar function parameter changes to modify the argument in the caller.",
        "rule_to_remember": "C passes scalar parameters by value (copy). To mutate caller variables, pass a pointer (int* ptr).",
        "code_fix": {
            "wrong": "void boost(int energy) { energy += 10; } // Caller variable unchanged",
            "right": "void boost(int* energy) { *energy += 10; } // Mutates caller memory",
        },
        "self_check": "If the function signature does not take a pointer (*), the caller's variable cannot change.",
    },
    "M10": {
        "root_cause": "Calling printf to display the result on screen instead of returning it to the caller.",
        "rule_to_remember": "printf outputs text to stdout for human eyes; return hands structured data back to the calling function or harness.",
        "code_fix": {
            "wrong": "int compute(int x) {\n    printf(\"%d\", x * 2); // Caller gets undefined return value!\n}",
            "right": "int compute(int x) {\n    return x * 2; // Returns computed value\n}",
        },
        "self_check": "Check the function signature: if it starts with int or float, there must be a matching return <value>; statement.",
    },
    "D01": {
        "root_cause": "Returning not-found (-1 or false) from an else branch inside the loop on the very first miss.",
        "rule_to_remember": "Only return found inside the loop. The not-found exit must come AFTER the entire loop has completed.",
        "code_fix": {
            "wrong": "for (int i = 0; i < n; i++) {\n    if (arr[i] == target) return i;\n    else return -1; // Aborts on first non-match!\n}",
            "right": "for (int i = 0; i < n; i++) {\n    if (arr[i] == target) return i;\n}\nreturn -1; // Checked every element first",
        },
        "self_check": "If target is at index 2, does your code exit at index 0? If yes, you have an early exit bug.",
    },
    "D02": {
        "root_cause": "Binary search window never shrinks because low = mid or high = mid, freezing the search window.",
        "rule_to_remember": "mid was already evaluated. Exclude it from the remaining window: low = mid + 1 and high = mid - 1.",
        "code_fix": {
            "wrong": "if (arr[mid] < target) low = mid; // Freezes when low + 1 == high\nelse high = mid;",
            "right": "if (arr[mid] < target) low = mid + 1; // Strictly shrinks window\nelse high = mid - 1;",
        },
        "self_check": "Trace a 2-element array [3, 7] searching for 5: does the window shrink to 0 elements and terminate cleanly?",
    },
    "D03": {
        "root_cause": "Attempting to swap elements without a temporary buffer, overwriting the first value.",
        "rule_to_remember": "Two assignments cannot execute simultaneously. A swap requires a temporary buffer: temp = a; a = b; b = temp;.",
        "code_fix": {
            "wrong": "arr[j] = arr[j + 1]; // Overwrites arr[j]!\narr[j + 1] = arr[j]; // Copies back the overwritten value",
            "right": "int temp = arr[j];\narr[j] = arr[j + 1];\narr[j + 1] = temp; // Safely swaps",
        },
        "self_check": "Trace with a = 5, b = 9. After step 1 (a = b), both a and b are 9! A temporary storage variable is mandatory.",
    },
    "D04": {
        "root_cause": "Using a single pass for bubble sort; one pass only moves the single largest item to the end.",
        "rule_to_remember": "Bubble sort requires two nested loops: outer loop runs n-1 passes; inner loop performs adjacent comparisons.",
        "code_fix": {
            "wrong": "for (int j = 0; j < n - 1; j++) {\n    if (arr[j] > arr[j + 1]) swap(&arr[j], &arr[j + 1]);\n} // Only bubbles 1 element!",
            "right": "for (int i = 0; i < n - 1; i++) {\n    for (int j = 0; j < n - i - 1; j++) {\n        if (arr[j] > arr[j + 1]) swap(&arr[j], &arr[j + 1]);\n    }\n}",
        },
        "self_check": "Trace [5, 4, 3, 2, 1]: after one pass, array is [4, 3, 2, 1, 5], which is not yet sorted.",
    },
    "D05": {
        "root_cause": "Omitting the base case in a recursive function, causing infinite recursion and stack overflow.",
        "rule_to_remember": "Always write and verify the base case at the top of the function before any recursive calls.",
        "code_fix": {
            "wrong": "int solve(int n) {\n    return n + solve(n - 1); // No base case -> stack overflow!\n}",
            "right": "int solve(int n) {\n    if (n <= 0) return 0; // Base case halts recursion\n    return n + solve(n - 1);\n}",
        },
        "self_check": "Ask: what is the smallest input (e.g. n = 0 or n = 1) where this function returns without recurring?",
    },
    "D06": {
        "root_cause": "Calling recursion with an unchanged argument (e.g. solve(n) instead of solve(n - 1)).",
        "rule_to_remember": "Every recursive step must pass an argument strictly closer to the base case.",
        "code_fix": {
            "wrong": "int cascade(int n) {\n    if (n <= 0) return 0;\n    return n + cascade(n); // Unchanged argument!\n}",
            "right": "int cascade(int n) {\n    if (n <= 0) return 0;\n    return n + cascade(n - 1); // Strictly shrinks\n}",
        },
        "self_check": "Verify the recursive argument: is it strictly smaller or partitioned compared to the incoming argument?",
    },
    "D07": {
        "root_cause": "Calling the recursive subproblem without capturing or combining its return value.",
        "rule_to_remember": "The result of a recursive call must be captured or directly returned: return n + solve(n - 1);.",
        "code_fix": {
            "wrong": "int solve(int n) {\n    if (n <= 0) return 0;\n    solve(n - 1); // Return value discarded!\n    return n;\n}",
            "right": "int solve(int n) {\n    if (n <= 0) return 0;\n    return n + solve(n - 1); // Combines recursive result\n}",
        },
        "self_check": "If a function returns a non-void value, verify that any recursive call's return value is stored or used.",
    },
    "D08": {
        "root_cause": "Using == to compare C string contents (char*), which only compares memory pointer addresses.",
        "rule_to_remember": "In C, string literals and char* must be compared using strcmp(s1, s2) == 0 from <string.h>.",
        "code_fix": {
            "wrong": "if (str == \"admin\") { // Compares pointer addresses!\n    grant_access();\n}",
            "right": "if (strcmp(str, \"admin\") == 0) { // Compares actual characters\n    grant_access();\n}",
        },
        "self_check": "Mental test: two different string variables with the same text have different memory addresses, so == fails.",
    },
}


def get_remediation_for_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a structured remediation guide for a diagnosed misconception finding."""
    cls = finding.get("class", "OTHER")
    name = finding.get("name") or CLASS_INFO.get(cls, {}).get("name", cls)
    subtitle = finding.get("subtitle") or CLASS_INFO.get(cls, {}).get("subtitle", "")
    trial_title = finding.get("trial_title") or finding.get("item_id", "")
    evidence = finding.get("evidence") or []
    evidence_text = "\n".join(e.get("text", "") for e in evidence if isinstance(e, dict))

    base = EXPERT_REMEDIATIONS.get(cls)
    if base is None:
        base = {
            "root_cause": subtitle or "Unexpected logic path or invariant violation.",
            "rule_to_remember": "Review function contracts, boundary inputs (0, 1, max), and step invariants.",
            "code_fix": {
                "wrong": "// Faulty assumption or unverified boundary condition",
                "right": "// Validate preconditions and maintain loop/recursion invariants",
            },
            "self_check": "Trace edge cases: empty input, single element, boundary limits.",
        }

    # If live Gemini is enabled, ask LLM for customized code-level remediation
    if is_live_available() and evidence_text:
        prompt = (
            f"You are an expert C programming tutor for Re:Learn.\n"
            f"Student made the misconception {cls}: {name} ({subtitle}) on trial '{trial_title}'.\n"
            f"Observed code evidence:\n{evidence_text}\n\n"
            f"Provide a JSON response with:\n"
            f'{{"root_cause": "concise 1-2 sentence explanation of why this error occurred", '
            f'"rule_to_remember": "1 punchy invariant rule to remember", '
            f'"self_check": "1 concrete test check to perform before compiling"}}\n'
            f"Do not include markdown or extra keys."
        )
        try:
            raw = call_llm(prompt, system_instruction="You are a C programming pedagogy expert. Output valid JSON only.")
            if raw and "{" in raw:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(cleaned)
                return {
                    "class": cls,
                    "name": name,
                    "subtitle": subtitle,
                    "trial_title": trial_title,
                    "root_cause": parsed.get("root_cause") or base["root_cause"],
                    "rule_to_remember": parsed.get("rule_to_remember") or base["rule_to_remember"],
                    "code_fix": base["code_fix"],
                    "self_check": parsed.get("self_check") or base["self_check"],
                }
        except Exception as exc:
            log.warning("LLM remediation generation failed (%s); using expert template", exc)

    return {
        "class": cls,
        "name": name,
        "subtitle": subtitle,
        "trial_title": trial_title,
        "root_cause": base["root_cause"],
        "rule_to_remember": base["rule_to_remember"],
        "code_fix": base["code_fix"],
        "self_check": base["self_check"],
    }


def enrich_report_with_remediations(report: Dict[str, Any]) -> Dict[str, Any]:
    """Attaches remediation guides to a debrief report."""
    findings = report.get("findings") or []
    remediations = []
    seen = set()
    for f in findings:
        cls = f.get("class")
        if cls and cls not in seen and cls not in ("CORRECT", "OTHER"):
            seen.add(cls)
            remediations.append(get_remediation_for_finding(f))

    # If no findings (perfect score), provide a positive mastery note
    if not remediations:
        remediations.append({
            "class": "MASTERY",
            "name": "Full Invariant Mastery",
            "subtitle": "All attempted trials passed invariant verification",
            "trial_title": "Deep Space Trials",
            "root_cause": "Solid grasp of array memory models, pointer bounds, recursion base cases, and loop termination.",
            "rule_to_remember": "Keep verifying preconditions and edge cases on larger real-world datasets.",
            "code_fix": {
                "wrong": "// No bugs detected in attempted sessions",
                "right": "// Continue applying rigorous boundary checking",
            },
            "self_check": "Always ask: what happens on empty array, n = 1, and duplicate elements?",
        })

    report["remediations"] = remediations
    return report
