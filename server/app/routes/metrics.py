"""GET /metrics (package S3).

Assembles the 03 §9.4 document from eval cards that already exist:
`ml/eval/cards/*.json`, `ml/eval/*_card.json`, `ml/eval/out/e-b/*_card.json`,
and the red-rule table in `docs/eda.md`. A metric that is not in those files is the
string "missing". Numbers are copied, never filled in.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

log = logging.getLogger("relearn.metrics")
router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "ml" / "eval"
EDA = ROOT / "docs" / "eda.md"
EXPECTED = [f"E{number}" for number in range(1, 19)]
MISSING = "missing"
_CARD_ID = re.compile(r"^E\d+$")
_RULE = re.compile(r"^\| (10\.\d+) \| (.+?) \| (PASS|FAIL|INSPECT|SKIP) \| (.+?) \|$")


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 3)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("could not read %s", path.name)
        return None


def _cards():
    """Later directories win, so ml/eval/cards overrides an older copy of the same id."""
    found = {}
    sources = []
    groups = [
        list((EVAL / "out" / "e-b").glob("*_card.json")),
        list(EVAL.glob("*_card.json")),
        list((EVAL / "cards").glob("*.json")),
    ]
    for group in groups:
        for path in sorted(group):
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            card_id = data.get("id")
            if not isinstance(card_id, str) or not _CARD_ID.match(card_id):
                continue
            found[card_id] = data
            sources.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    ordered = []
    for card_id in EXPECTED:
        if card_id in found:
            ordered.append(found.pop(card_id))
        else:
            ordered.append({
                "id": card_id, "title": MISSING, "slice": MISSING, "n": MISSING,
                "metrics": MISSING, "missing": True,
            })
    for card_id in sorted(found):
        ordered.append(found[card_id])
    return ordered, sources


def _baselines(cards):
    card = next((row for row in cards if row.get("id") == "E8" and not row.get("missing")), None)
    metrics = (card or {}).get("metrics")
    if not isinstance(metrics, dict):
        return MISSING
    rows = []
    for name, splits in metrics.items():
        if not isinstance(splits, dict):
            continue
        row = {"name": name}
        for split, value in splits.items():
            if isinstance(value, dict) and "macro_f1" in value:
                row[split] = value["macro_f1"]
        if len(row) > 1:
            rows.append(row)
    return rows or MISSING


def _cross_domain(cards):
    card = next((row for row in cards if row.get("id") == "E15" and not row.get("missing")), None)
    metrics = (card or {}).get("metrics")
    if not isinstance(metrics, dict):
        return MISSING
    return {
        "m_recall_on_dsa_trained_main_only": metrics.get("m_recall_on_dsa_trained_main_only", MISSING),
        "shortcut_auc": metrics.get("shortcut_auc", MISSING),
    }


def _exam_sim(cards):
    card = next((row for row in cards if row.get("id") == "E14" and not row.get("missing")), None)
    if card is None:
        return MISSING
    rows = card.get("rows") if isinstance(card.get("rows"), list) else []
    final_f1, recheck = {}, {}
    for row in rows:
        if not isinstance(row, dict) or "policy" not in row:
            continue
        policy = row["policy"]
        final_f1[policy] = row.get("profile_f1", MISSING)
        recheck[policy] = row.get("stable_recheck", MISSING)
    return {
        "curves": card.get("curves", MISSING),
        "final_f1": final_f1 or MISSING,
        "stable_recheck_rate": recheck or MISSING,
        "caveat": card.get("caveat", MISSING),
    }


def _resolution(cards):
    card = next((row for row in cards if row.get("id") == "E10" and not row.get("missing")), None)
    if card is None or not isinstance(card.get("rows"), list):
        return MISSING
    return {"rows": card["rows"], "caveat": card.get("caveat", MISSING)}


def _failure_audit(cards):
    card = next((row for row in cards if row.get("id") == "E12" and not row.get("missing")), None)
    if card is None:
        return MISSING
    table = card.get("table")
    return table if isinstance(table, list) else MISSING


def _eda():
    if not EDA.is_file():
        return {"source": "docs/eda.md", "missing": True, "rules": MISSING, "rows": MISSING, "unique_hash": MISSING}
    text = EDA.read_text(encoding="utf-8")
    rules = []
    seen = set()
    for line in text.splitlines():
        match = _RULE.match(line.strip())
        if not match or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        rules.append({
            "id": match.group(1),
            "rule": match.group(2).strip(),
            "status": match.group(3),
            "detail": match.group(4).strip(),
        })
    rows = re.search(r"Rows: \*\*(\d+)\*\*", text)
    hashes = re.search(r"(\d+) unique hashes", text)
    return {
        "source": "docs/eda.md",
        "missing": False,
        "rules": rules or MISSING,
        "rows": int(rows.group(1)) if rows else MISSING,
        "train_rows": int(re.search(r"Train rows \(\`split=train\`\): \*\*(\d+)\*\*", text).group(1))
        if re.search(r"Train rows \(\`split=train\`\): \*\*(\d+)\*\*", text) else MISSING,
        "unique_hash": int(hashes.group(1)) if hashes else MISSING,
    }


def _generated_at(sources):
    mtimes = []
    for relative in sources:
        path = ROOT / relative
        if path.is_file():
            mtimes.append(path.stat().st_mtime)
    if EDA.is_file():
        mtimes.append(EDA.stat().st_mtime)
    if not mtimes:
        return MISSING
    return datetime.fromtimestamp(max(mtimes), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/metrics")
def metrics():
    started = time.perf_counter()
    cards, sources = _cards()
    version = MISSING
    for card in cards:
        if card.get("id") == "E1" and isinstance(card.get("model_version"), str):
            version = card["model_version"]
            break
    eda = _eda()
    e1 = next((card for card in cards if card.get("id") == "E1" and not card.get("missing")), None)
    return {
        "model_version": version,
        "generated_at": _generated_at(sources),
        "effective_n": {
            "rows": eda.get("rows", MISSING),
            "unique_hash": eda.get("unique_hash", MISSING),
            "unique_prog_op": MISSING,
            "train_rows": (e1 or {}).get("n", MISSING) if e1 else MISSING,
        },
        "cards": cards,
        "baselines": _baselines(cards),
        "domains": MISSING,
        "cross_domain": _cross_domain(cards),
        "exam_sim": _exam_sim(cards),
        "resolution": _resolution(cards),
        "failure_audit": _failure_audit(cards),
        "eda": eda,
        "sources": sources,
        "latency_ms": _ms(started),
    }
