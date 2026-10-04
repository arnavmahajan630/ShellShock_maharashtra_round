"""Item choice for the Deep Space Trials (ml_plan/03 §8.5.4, §8.5.7).

Each item is a set of independent binary tests, one per class whose exposure
is at least ``EXAM_MIN_EXPOSURE``. Entropy is in bits. Trace exposure is
``P(belief answer | class)`` from ``likelihood`` (03 §6.2, §8.5.1).

The fixed blueprint is the §8.5.7 fallback. Two ids in the plan are not in
the bank B3 shipped: ``xt_bsearch_1`` and ``xt_arr_1``. See notes/D3.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ml.bayes.likelihood import likelihood
from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS
from ml.contracts.params import (
    ELO_ITEM,
    EXAM_COST,
    EXAM_MIN_EXPOSURE,
    EXAM_TARGET_PASS,
    EXAM_W_COVERAGE,
    EXAM_W_DIFFICULTY,
    EXAM_W_GHOST,
    FAIL_SIGNATURE_IF_NOT,
)
from ml.contracts.schemas import ExamItem

ML_ROOT = Path(__file__).resolve().parents[1]
DSA_DIR = ML_ROOT / "problems" / "dsa"
MAIN_DIR = ML_ROOT / "problems" / "main"
EXAM_ITEMS_PATH = ML_ROOT / "data" / "exam_items.json"

# 03 §8.5.7. ``xt_bsearch_1`` is not in exam_items.json; the searching traces
# are ``xt_search_*``, and ``xt_search_1`` is already in the list, so the last
# slot is ``xt_search_6``. ``xt_arr_1`` is ``xt_array_1`` in the bank.
BLUEPRINT_STRINGS = [
    "Q01", "xt_sort_1", "Q06", "xt_rec_1", "Q17",
    "xt_str_1", "Q14", "xt_search_1", "Q10", "xt_search_6",
]
BLUEPRINT_NO_STRINGS = [
    "Q01", "xt_sort_1", "Q06", "xt_rec_1", "Q17",
    "xt_search_1", "Q10", "xt_search_2", "Q03", "xt_array_1",
]

SECTOR_LABEL = {
    "searching": "Searching",
    "sorting": "Sorting",
    "array_tech": "Array tech",
    "strings": "Strings",
    "recursion": "Recursion",
}

CODING_PER_SECTOR = 2


def fixed_blueprint(*, strings=True):
    """Item ids for one fixed exam (03 §8.5.7). Strings stay unless cut."""
    return list(BLUEPRINT_STRINGS if strings else BLUEPRINT_NO_STRINGS)


def tie_key(learner_id, exam_id, label):
    """Stable tie break. The seed is (learner_id, exam_id); the label splits items."""
    text = f"{learner_id}:{exam_id}:{label}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def binary_entropy(p):
    """Shannon entropy of a coin with probability ``p``, in bits."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _posterior(p, like_active, like_not):
    numer = p * like_active
    denom = numer + (1.0 - p) * like_not
    if denom == 0.0:
        return p
    return numer / denom


def class_eig(p, exposure, fail_if_not=FAIL_SIGNATURE_IF_NOT):
    """Expected information gain of one class's binary fail test, in bits (03 §8.5.4)."""
    p = float(p)
    e = float(exposure)
    p_fail = p * e + (1.0 - p) * float(fail_if_not)
    post_fail = _posterior(p, e, float(fail_if_not))
    post_ok = _posterior(p, 1.0 - e, 1.0 - float(fail_if_not))
    gained = binary_entropy(p) - (
        p_fail * binary_entropy(post_fail) + (1.0 - p_fail) * binary_entropy(post_ok)
    )
    if gained < 0.0 and gained > -1e-12:
        return 0.0
    return gained


def class_eigs(p_active, exposure):
    """Per-class EIG for classes whose exposure clears ``EXAM_MIN_EXPOSURE``."""
    parts = {}
    for cls, raw in (exposure or {}).items():
        if cls not in MISCONCEPTIONS:
            continue
        e = float(raw)
        if e < EXAM_MIN_EXPOSURE:
            continue
        if cls in p_active:
            p = float(p_active[cls])
        else:
            p = 0.0
        parts[cls] = class_eig(p, e)
    return parts


def belief_exposure(options, correct, belief):
    """Exposure of a choice item: P(belief answer | class) from the §6.2 table."""
    table = likelihood(options, correct, belief)
    exposure = {}
    correct_s = str(correct)
    for cls, ans in (belief or {}).items():
        if cls not in MISCONCEPTIONS:
            continue
        ans_s = str(ans)
        row = table.get(cls)
        if row is None or ans_s == correct_s or ans_s not in row:
            continue
        exposure[cls] = row[ans_s]
    return exposure


def pass_probability(difficulty, theta):
    """Elo pass chance (03 §8.5.2)."""
    rating = ELO_ITEM[int(difficulty)]
    return 1.0 / (1.0 + 10 ** ((rating - float(theta)) / 400.0))


def load_pool(*, strings=True):
    """Coding items Q01–Q18 plus the exam trace bank. Options stay in file order."""
    items = []
    for path in sorted(DSA_DIR.glob("Q*.json")):
        problem = json.loads(path.read_text(encoding="utf-8"))
        if not strings and problem.get("sector") == "strings":
            continue
        items.append(_coding_item(problem))
    raw_items = json.loads(EXAM_ITEMS_PATH.read_text(encoding="utf-8"))
    for raw in raw_items:
        if not strings and raw.get("sector") == "strings":
            continue
        items.append(_trace_item(raw))
    return items


def practice_catalog():
    """Problem id, family, and sector for recommendation lookup. DSA, then main."""
    rows = []
    for folder in (DSA_DIR, MAIN_DIR):
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "problem_id": data["problem_id"],
                "family": data.get("family") or "",
                "sector": data.get("sector"),
            })
    return rows


def public_exam_item(item):
    """Fields an exam client may see. No exposure, correct answer, or belief map."""
    data = {
        "item_id": item["item_id"],
        "kind": item["kind"],
        "sector": item["sector"],
        "difficulty": int(item["difficulty"]),
        "markers": list(item.get("markers") or []),
        "sample_tests": list(item.get("sample_tests") or []),
        "options": list(item.get("options") or []),
    }
    if item["kind"] == "coding":
        data["prompt"] = item.get("prompt")
        data["signature"] = item.get("signature")
        data["starter"] = item.get("starter")
    else:
        data["code"] = item.get("code")
        data["question"] = item.get("question")
    return ExamItem.model_validate(data).model_dump()


def select_next(pool, session, n_coding, n_trace):
    """The next adaptive item as ``(item, reason, eig_bits)``, or None when the exam is full.

    Item 1 is a difficulty-1 coding item from the sector with the highest prior EIG.
    Later items maximise the §8.5.4 score under the composition constraints.
    """
    if session["n_coding"] >= n_coding and session["n_trace"] >= n_trace:
        return None
    if session["n_coding"] == 0 and session["n_trace"] == 0:
        return _opening(pool, session)
    # Quotas, the coding cap, consecutive sectors, and coverage (when the length
    # allows it) are hard. If none of the remaining items fit, the exam stops
    # and the rest is unobserved rather than breaking a rule.
    return _best(
        pool, session, n_coding, n_trace, session["sectors"],
        require_consecutive=True, require_coverage=True,
    )


def _coding_item(problem):
    exposure = {}
    for cls, raw in (problem.get("exposure") or {}).items():
        if cls in MISCONCEPTIONS:
            exposure[cls] = float(raw)
    return {
        "item_id": problem["problem_id"],
        "kind": "coding",
        "sector": problem["sector"],
        "difficulty": int(problem["difficulty"]),
        "name": problem.get("name") or problem["problem_id"],
        "exposure": exposure,
        "problem": problem,
        "prompt": problem.get("prompt"),
        "signature": problem.get("signature"),
        "starter": problem.get("starter"),
        "markers": list(problem.get("markers") or []),
        "sample_tests": _samples(problem),
        "options": [],
        "code": None,
        "question": None,
        "correct": None,
        "belief": {},
    }


def _trace_item(raw):
    exposure = belief_exposure(raw["options"], raw["correct"], raw.get("belief"))
    return {
        "item_id": raw["item_id"],
        "kind": "trace",
        "sector": raw["sector"],
        "difficulty": int(raw.get("difficulty") or 1),
        "name": raw["item_id"],
        "exposure": exposure,
        "problem": None,
        "prompt": None,
        "signature": None,
        "starter": None,
        "markers": [],
        "sample_tests": [],
        "options": list(raw["options"]),
        "code": raw.get("code") or "",
        "question": raw.get("question"),
        "correct": raw["correct"],
        "belief": dict(raw.get("belief") or {}),
    }


def _samples(problem):
    samples = []
    for test in problem.get("tests") or []:
        if test.get("sample"):
            samples.append({"args": test.get("args") or [], "expect": test.get("expect") or {}})
    return samples


def _opening(pool, session):
    """Difficulty-1 coding item from the sector whose best opening item has the most EIG."""
    per_sector = {}
    for item in pool:
        if item["kind"] != "coding" or int(item["difficulty"]) != 1:
            continue
        if item["sector"] not in session["sectors"]:
            continue
        eig = sum(class_eigs(session["p"], item["exposure"]).values())
        current = per_sector.get(item["sector"])
        if current is None or eig > current:
            per_sector[item["sector"]] = eig
    if not per_sector:
        raise RuntimeError("no difficulty-1 coding item in the pool")
    sector = max(
        per_sector,
        key=lambda name: (per_sector[name], -tie_key(session["learner_id"], session["exam_id"], name)),
    )
    candidates = [
        item for item in pool
        if item["kind"] == "coding" and int(item["difficulty"]) == 1 and item["sector"] == sector
    ]
    return _choose(candidates, session)


def _best(pool, session, n_coding, n_trace, sectors, *, require_consecutive, require_coverage):
    candidates = [
        item for item in pool
        if _allowed(
            item, session, n_coding, n_trace, sectors,
            require_consecutive=require_consecutive, require_coverage=require_coverage,
        )
    ]
    if not candidates:
        return None
    return _choose(candidates, session)


def _choose(candidates, session):
    best = None
    best_key = None
    best_eval = None
    for item in candidates:
        ev = evaluate(item, session)
        key = (ev["score"], -tie_key(session["learner_id"], session["exam_id"], item["item_id"]))
        if best is None or key > best_key:
            best = item
            best_key = key
            best_eval = ev
    reason = reason_line(
        best, best_eval["parts"],
        coverage=best_eval["coverage"],
        ghost_classes=best_eval["ghost_classes"],
        stable_since=session.get("stable_since") or {},
    )
    return best, reason, best_eval["eig"]


def evaluate(item, session):
    """§8.5.4 score. Coverage, difficulty, and ghost are the exam bonuses."""
    parts = class_eigs(session["p"], item["exposure"])
    eig = sum(parts.values())
    if eig < 0.0:
        eig = 0.0
    cost = EXAM_COST[item["kind"]]
    pp = pass_probability(item["difficulty"], session["theta"][item["sector"]])
    coverage = 0.0 if item["sector"] in session["covered"] else 1.0
    ghost_total, ghost_classes = _ghost(item, session)
    score = eig / cost
    score += EXAM_W_COVERAGE * coverage
    score -= EXAM_W_DIFFICULTY * abs(pp - EXAM_TARGET_PASS)
    score += EXAM_W_GHOST * ghost_total
    return {
        "score": score,
        "eig": eig,
        "parts": parts,
        "coverage": coverage,
        "ghost_classes": ghost_classes,
        "p_pass": pp,
    }


def _ghost(item, session):
    """Sum of exposure on STABLE classes this exam has not yet rechecked."""
    total = 0.0
    classes = []
    tested = session["tested"]
    states = session["states"]
    for cls, raw in item["exposure"].items():
        e = float(raw)
        if e <= 0.0 or cls in tested:
            continue
        if states.get(cls) == "STABLE":
            total += e
            classes.append(cls)
    return total, classes


def _allowed(item, session, n_coding, n_trace, sectors, *, require_consecutive, require_coverage):
    if item["item_id"] in session["used"]:
        return False
    if item["sector"] not in sectors:
        return False
    if item["kind"] == "coding":
        if session["n_coding"] >= n_coding:
            return False
        if session["coding_n"].get(item["sector"], 0) >= CODING_PER_SECTOR:
            return False
    elif session["n_trace"] >= n_trace:
        return False
    if require_consecutive and session["last"] == item["sector"]:
        return False
    if require_coverage and (n_coding + n_trace) >= len(sectors):
        slots_after = (n_coding - session["n_coding"]) + (n_trace - session["n_trace"]) - 1
        covered = set(session["covered"])
        covered.add(item["sector"])
        uncovered = sum(1 for sector in sectors if sector not in covered)
        if uncovered > slots_after:
            return False
    return True


def reason_line(item, parts, *, coverage, ghost_classes, stable_since):
    """One line: top classes by EIG contribution, then the bonus that fired."""
    ranked = sorted(parts, key=lambda cls: (-parts[cls], cls))
    ghost = set(ghost_classes)
    names = []
    for cls in ranked[:2]:
        label = CLASS_INFO[cls]["name"]
        if cls in ghost:
            since = stable_since.get(cls)
            if since:
                label = f"{label} (stable since {since} → recheck)"
            else:
                label = f"{label} (stable → recheck)"
        names.append(label)
    if len(names) == 2:
        about = f"{names[0]} and {names[1]}"
    elif len(names) == 1:
        about = names[0]
    else:
        about = "no class above the exposure floor"
    bonuses = []
    if coverage:
        bonuses.append(f"covers {SECTOR_LABEL.get(item['sector'], item['sector'])}")
    if ghost and not any(cls in ghost for cls in ranked[:2]):
        bonuses.append("rechecks a stable misconception")
    if not bonuses:
        bonuses.append("highest information per minute")
    head = item["item_id"]
    if item["kind"] == "coding" and item.get("name"):
        head = f"{item['item_id']} {item['name']}"
    return f"{head} — most informative about {about}; {'; '.join(bonuses)}."


def item_eig(p_active, item):
    """Bits of EIG for the adaptivity log, including a blueprint item."""
    total = sum(class_eigs(p_active, item["exposure"]).values())
    return 0.0 if total < 0.0 else total
