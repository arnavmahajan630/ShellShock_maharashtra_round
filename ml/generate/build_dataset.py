"""Build ``ml/data/dataset.jsonl`` (03 §3.5–§3.6, §3.9).

Run: ``.venv/Scripts/python -m ml.generate.build_dataset``
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ml.contracts.schemas import DatasetRow
from ml.generate.ambiguity import ambiguity_pass, ast_hash
from ml.generate.augment import near_misses, style_variants
from ml.generate.card import write_card
from ml.generate.registry import (
    AMB_OPS,
    allowed,
    apply_op,
    find_sites,
    operators,
    predicate_holds,
    primary_label,
)
from ml.generate.verify import Checker, reference_iters

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "ml" / "data" / "dataset.jsonl"
_LUCK = _ROOT / "ml" / "data" / "passes_by_luck.jsonl"
_CARD = _ROOT / "docs" / "dataset_card.md"

_M = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10"]
_D = ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"]
_HIGH = {label: 550 for label in _M}
_HIGH.update({label: 400 for label in _D})
_HIGH["CORRECT"] = 1400
_HIGH["OTHER"] = 700
_AIM = {label: 300 for label in _M}
_AIM.update({label: 250 for label in _D})
_AIM["CORRECT"] = 1100
_AIM["OTHER"] = 500
_FLOOR = 80
_SEED = 0


def load_problems():
    found = []
    for folder in ("main", "dsa"):
        for path in sorted((_ROOT / "ml" / "problems" / folder).glob("*.json")):
            found.append(json.loads(path.read_text(encoding="utf-8")))
    return found


def _aug_limit(op_id, n_sites):
    if n_sites < 24 or op_id.startswith("d0") or op_id in {
        "m04_drop_cast", "m04_cast_late", "m04_int_literal", "m06_if_assign", "m06_while_assign",
    }:
        return 16
    if op_id.startswith("oth_") or n_sites < 40:
        return 6
    return 4


def _hash_for(key, cache):
    hit = cache.get(key)
    if hit is None:
        hit = ast_hash(key)
        cache[key] = hit
    return hit


def _row(problem, variant, source, op_id, aug, code, label, info, hashes, predicate, labels_all=None, soft=None, two=False, luck=False):
    return {
        "source": source,
        "problem_id": problem["problem_id"],
        "family": problem.get("family") or "",
        "split": problem.get("split") or "train",
        "variant": variant,
        "op_id": op_id,
        "op_variant": op_id,
        "aug": list(aug),
        "code": code,
        "label": label,
        "soft_label": soft,
        "labels_all": list(labels_all or [label]),
        "is_two_bug": two,
        "ambiguous_group": None,
        "ast_hash": _hash_for(info["key"], hashes),
        "verified": {
            "tests_failed": info["tests_failed"],
            "tests_total": info["tests_total"],
            "predicate": list(predicate),
            "passes_by_luck": luck,
        },
        "trace_summary": {"status": "ok", "events": [], "loop_iters_delta": None},
        "author": None,
        "rater2_label": None,
        "_key": info["key"],
    }


def _round_robin(groups, high, cap_po, cap_aug):
    keys = list(groups)
    index = {key: 0 for key in keys}
    picked = []
    used_po = Counter()
    used_aug = Counter()
    moved = True
    while len(picked) < high and moved:
        moved = False
        for key in keys:
            cursor = index[key]
            bucket = groups[key]
            while cursor < len(bucket):
                row = bucket[cursor]
                cursor += 1
                po = (row["problem_id"], row.get("op_id"))
                pv = (row["problem_id"], row.get("variant"), row.get("op_id"))
                if used_po[po] >= cap_po:
                    continue
                if row["aug"] and used_aug[pv] >= cap_aug:
                    continue
                picked.append(row)
                used_po[po] += 1
                if row["aug"]:
                    used_aug[pv] += 1
                moved = True
                break
            index[key] = cursor
            if len(picked) >= high:
                break
    return picked


def _select_label(rows, label, high, aim):
    """Keep at most ``high`` rows. The 15/6 caps stay when they can already reach ``aim``.

    A class whose sites cannot fill ``aim`` under those caps gets a higher cap, still
    spread across ``(problem, op_id)`` by round-robin, and never past ``high``.
    """
    if not rows:
        return [], None
    groups = defaultdict(list)
    for row in rows:
        groups[(row["problem_id"], row.get("op_id"))].append(row)
    group_count = max(len(groups), 1)
    under_default = sum(min(15, len(bucket)) for bucket in groups.values())
    relaxed = None
    if under_default < aim:
        cap_po = max(15, (aim + group_count - 1) // group_count + 2)
        cap_aug = max(16, cap_po)
        relaxed = (label, cap_po, cap_aug, len(groups))
    else:
        cap_po, cap_aug = 15, 6
    for bucket in groups.values():
        bucket.sort(key=lambda row: (
            0 if row["source"] == "AMB" else 1,
            0 if not row["aug"] else 1,
            0 if row["source"] == "E" else 1,
            row.get("variant") or "",
            row["code"],
        ))
    picked = _round_robin(groups, high, cap_po, cap_aug)
    if len(picked) < min(aim, len(rows)):
        cap_po = max(cap_po, (aim + group_count - 1) // max(1, sum(1 for bucket in groups.values() if len(bucket) >= 8)) + 2)
        cap_aug = max(cap_aug, 16, cap_po)
        picked = _round_robin(groups, high, cap_po, cap_aug)
        relaxed = (label, cap_po, cap_aug, len(groups))
    if relaxed and relaxed[1] <= 15 and relaxed[2] <= 6:
        relaxed = None
    return picked, relaxed


def _dedupe(rows):
    best = {}
    rank = {"AMB": 3, "E": 2, "A": 1}
    for row in rows:
        key = (row["label"], row["code"])
        prev = best.get(key)
        if prev is None or rank.get(row["source"], 0) > rank.get(prev["source"], 0):
            best[key] = row
    return list(best.values())


def _two_bug(problem, variant, singles, checker, rng, hashes):
    pool = [row for row in singles if row["label"] != "OTHER"]
    rng.shuffle(pool)
    kept = []
    tries = 0
    for left_i, left in enumerate(pool):
        if len(kept) >= 2 or tries >= 8:
            break
        for right in pool[left_i + 1:]:
            if len(kept) >= 2 or tries >= 8:
                break
            if left["op_id"] == right["op_id"]:
                continue
            tries += 1
            sites = find_sites(right["_op"], left["code"])
            if not sites:
                continue
            try:
                composed = apply_op(right["_op"], left["code"], sites[0])
            except Exception:
                continue
            if composed == left["code"]:
                continue
            if not predicate_holds(left["op_id"], composed) or not predicate_holds(right["op_id"], composed):
                continue
            decision, info = checker.judge(problem, composed, "class", left["op_id"])
            if decision != "keep":
                continue
            if not (info["fail_set"] & left["fail_set"]) or not (info["fail_set"] & right["fail_set"]):
                continue
            labels = []
            for lab in list(left["labels_all"]) + list(right["labels_all"]):
                if lab not in labels:
                    labels.append(lab)
            soft = None
            if len(labels) >= 2:
                weight = 1.0 / len(labels)
                soft = {lab: weight for lab in labels}
            kept.append(_row(
                problem, variant, "E", f"{left['op_id']}+{right['op_id']}", [], composed,
                labels[0], info, hashes, [left["op_id"], right["op_id"]],
                labels_all=labels, soft=soft, two=True,
            ))
    return kept


def _attach_traces(rows, problems, checker, ref_cache):
    by_id = {problem["problem_id"]: problem for problem in problems}
    for row in rows:
        problem = by_id[row["problem_id"]]
        if problem["problem_id"] not in ref_cache:
            ref_cache[problem["problem_id"]] = reference_iters(checker, problem)
        row["trace_summary"] = checker.summarize(problem, row["code"], ref_cache[problem["problem_id"]])
        row.pop("_key", None)
        row.pop("_op", None)


def _dump(rows):
    dumped = []
    for row in rows:
        dumped.append(DatasetRow.model_validate(row).model_dump())
    return dumped


def build():
    rng = random.Random(_SEED)
    problems = load_problems()
    ops = operators()
    checker = Checker()
    hashes = {}
    rows = []
    luck_rows = []
    attempts = Counter()
    kept_base = Counter()
    luck_base = Counter()
    drop_reason = Counter()
    aug_rejects = 0
    site_counts = Counter()

    matched = {}
    for problem in problems:
        use = [op for op in ops if allowed(op, problem.get("allowed_ops"))]
        matched[problem["problem_id"]] = use
        for code in problem["correct_variants"]:
            for op in use:
                site_counts[op.op_id] += len(find_sites(op, code))

    for problem in problems:
        use = matched[problem["problem_id"]]
        for index, code in enumerate(problem["correct_variants"]):
            variant = f"cv{index + 1}"
            decision, info = checker.judge(problem, code, "correct")
            if decision == "keep":
                rows.append(_row(
                    problem, variant, "A", None, [], code, "CORRECT", info, hashes, [],
                ))
                for names, text in style_variants(code, 4, rng):
                    aug_decision, aug_info = checker.judge(problem, text, "correct")
                    if aug_decision == "keep":
                        rows.append(_row(
                            problem, variant, "A", None, names, text, "CORRECT", aug_info, hashes, [],
                        ))
                    else:
                        aug_rejects += 1
            for op_id, text in near_misses(code):
                nm_decision, nm_info = checker.judge(problem, text, "correct")
                if nm_decision != "keep":
                    continue
                rows.append(_row(
                    problem, variant, "E", op_id, [], text, "CORRECT", nm_info, hashes, [],
                ))
                for names, aug_text in style_variants(text, 3, rng):
                    aug_decision, aug_info = checker.judge(problem, aug_text, "correct")
                    if aug_decision == "keep":
                        rows.append(_row(
                            problem, variant, "E", op_id, names, aug_text, "CORRECT", aug_info, hashes, [],
                        ))
                    else:
                        aug_rejects += 1
            singles = []
            for op in use:
                limit = _aug_limit(op.op_id, site_counts[op.op_id])
                label = primary_label(op)
                mode = "other" if label == "OTHER" else "class"
                labels_all = list(op.labels)
                for site in find_sites(op, code):
                    attempts[op.op_id] += 1
                    try:
                        mutant = apply_op(op, code, site)
                    except Exception:
                        drop_reason[(op.op_id, "apply")] += 1
                        continue
                    decision, info = checker.judge(problem, mutant, mode, op.op_id)
                    if decision == "luck":
                        luck_base[op.op_id] += 1
                        luck_rows.append(_row(
                            problem, variant, "A", op.op_id, [], mutant, label, info, hashes,
                            [op.op_id], labels_all=labels_all, luck=True,
                        ))
                        continue
                    if decision != "keep":
                        drop_reason[(op.op_id, decision)] += 1
                        continue
                    kept_base[op.op_id] += 1
                    row = _row(
                        problem, variant, "A", op.op_id, [], mutant, label, info, hashes,
                        [op.op_id], labels_all=labels_all,
                    )
                    rows.append(row)
                    singles.append({
                        "op_id": op.op_id,
                        "_op": op,
                        "code": mutant,
                        "label": label,
                        "labels_all": labels_all,
                        "fail_set": info["fail_set"],
                    })
                    for names, text in style_variants(mutant, limit, rng):
                        aug_decision, aug_info = checker.judge(problem, text, mode, op.op_id)
                        if aug_decision != "keep":
                            aug_rejects += 1
                            continue
                        rows.append(_row(
                            problem, variant, "A", op.op_id, names, text, label, aug_info, hashes,
                            [op.op_id], labels_all=labels_all,
                        ))
            rows.extend(_two_bug(problem, variant, singles, checker, rng, hashes))
        print(f"{problem['problem_id']} rows={len(rows)} luck={len(luck_rows)}", flush=True)

    rows, clashes = ambiguity_pass(rows, AMB_OPS)
    rows = _dedupe(rows)
    luck_rows = _dedupe(luck_rows)

    selected = []
    relaxed = []
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for label in list(_M) + list(_D) + ["OTHER"]:
        picked, note = _select_label(by_label.get(label, []), label, _HIGH[label], _AIM[label])
        selected.extend(picked)
        if note:
            relaxed.append(note)
    plain = [row for row in by_label.get("CORRECT", []) if row["source"] == "A"]
    near = [row for row in by_label.get("CORRECT", []) if row["source"] != "A"]
    picked_plain, note = _select_label(plain, "CORRECT", 700, 500)
    if note:
        relaxed.append(("CORRECT plain", note[1], note[2], note[3]))
    picked_near, note = _select_label(near, "CORRECT", 900, 400)
    if note:
        relaxed.append(note)
    if picked_plain and picked_near and len(picked_near) / (len(picked_plain) + len(picked_near)) < 0.30:
        while picked_plain and len(picked_near) / (len(picked_plain) + len(picked_near)) < 0.30:
            if len(picked_plain) <= _FLOOR:
                break
            picked_plain.pop()
    selected.extend(picked_plain)
    selected.extend(picked_near)

    have = {id(row) for row in selected}
    counts = Counter(row["label"] for row in selected)
    for label, bucket in by_label.items():
        need = _FLOOR - counts[label]
        if need <= 0:
            continue
        for row in bucket:
            if id(row) in have:
                continue
            selected.append(row)
            have.add(id(row))
            counts[label] += 1
            need -= 1
            if need <= 0:
                break
    if len(selected) < 4000:
        for row in rows:
            if id(row) in have:
                continue
            selected.append(row)
            have.add(id(row))
            if len(selected) >= 4000:
                break

    ref_cache = {}
    _attach_traces(selected, problems, checker, ref_cache)
    _attach_traces(luck_rows, problems, checker, ref_cache)
    selected.sort(key=lambda row: (
        row["problem_id"], row.get("variant") or "", row["label"], row.get("op_id") or "",
        ",".join(row["aug"]), row["source"], row["code"],
    ))
    luck_rows.sort(key=lambda row: (row["problem_id"], row.get("op_id") or "", row["code"]))
    source_n = Counter()
    for row in selected:
        source_n[row["source"]] += 1
        row["id"] = f"{row['source']}-{source_n[row['source']]:06d}"
    for index, row in enumerate(luck_rows, start=1):
        row["id"] = f"L-{index:06d}"

    dumped = _dump(selected)
    luck_dumped = _dump(luck_rows)
    _DATA.parent.mkdir(parents=True, exist_ok=True)
    _DATA.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dumped), encoding="utf-8")
    _LUCK.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in luck_dumped), encoding="utf-8")

    class_counts = Counter(row["label"] for row in dumped)
    amb_groups = len({row["ast_hash"] for row in dumped if row["source"] == "AMB"})
    operators_report = []
    zero_kept = []
    for op in ops:
        tries = attempts[op.op_id]
        kept = kept_base[op.op_id]
        luck = luck_base[op.op_id]
        rate = None if tries == 0 else (tries - kept) / tries
        operators_report.append((op.op_id, tries, kept, luck, rate))
        if kept == 0:
            if tries == 0:
                reason = "no site on a correct variant whose allowed_ops match"
            else:
                reason = f"{tries} sites, none kept (luck {luck})"
            zero_kept.append((op.op_id, reason))
    correct_rows = class_counts["CORRECT"]
    near_rows = sum(1 for row in dumped if row["label"] == "CORRECT" and row["source"] == "E")
    report = {
        "n_rows": len(dumped),
        "n_unique_hash": len({row["ast_hash"] for row in dumped}),
        "n_unique_pvo": len({(row["problem_id"], row["variant"], row["op_id"]) for row in dumped}),
        "domain_main": sum(1 for row in dumped if row["problem_id"].startswith("P")),
        "domain_dsa": sum(1 for row in dumped if row["problem_id"].startswith("Q")),
        "two_bug": sum(1 for row in dumped if row["is_two_bug"]),
        "luck_rows": len(luck_dumped),
        "luck_by_op": dict(luck_base),
        "operators": operators_report,
        "zero_kept": zero_kept,
        "clashes": clashes,
        "amb_groups": amb_groups,
        "aug_rejects": aug_rejects,
        "relaxed": relaxed,
        "near_share": (near_rows / correct_rows) if correct_rows else 0.0,
    }
    write_card(_CARD, dumped, report)
    _print_summary(dumped, class_counts, report, clashes)
    return report


def _print_summary(rows, class_counts, report, clashes):
    print(f"rows {len(rows)}")
    print("per-class " + " ".join(f"{label}={class_counts[label]}" for label in list(_M) + list(_D) + ["OTHER", "CORRECT"]))
    print(
        f"effective n_rows={report['n_rows']} n_unique_hash={report['n_unique_hash']} "
        f"n_unique_pvo={report['n_unique_pvo']}"
    )
    print(f"domain main={report['domain_main']} dsa={report['domain_dsa']}")
    print(f"sources A={sum(r['source']=='A' for r in rows)} E={sum(r['source']=='E' for r in rows)} AMB={sum(r['source']=='AMB' for r in rows)}")
    print(f"two_bug {report['two_bug']} luck {report['luck_rows']} near_miss_share {report['near_share']:.2f}")
    print(f"clashes {len(clashes)}")
    for line in clashes:
        print(line)
    print("zero_kept " + ", ".join(op_id for op_id, _ in report["zero_kept"]))


def main():
    build()


if __name__ == "__main__":
    main()
