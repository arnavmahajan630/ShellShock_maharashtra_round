"""C4: ml/data/code_items.json and ml/items/build_code_items.py (05 section 3.2).

The full re-run of every item is `python -m ml.items.build_code_items --check`; the last test here
runs it. The others are quick looks at the file and at the helpers.
"""
import json
import re
from collections import Counter

import pytest

from ml.contracts.classes import MISCONCEPTIONS
from ml.contracts.schemas import CodeItem, PublicCodeItem
from ml.items import build_code_items as C

ITEMS = json.loads(C.OUT.read_text(encoding="utf-8"))
PROBLEMS = C.load_problems()
BY_TYPE = {kind: [i for i in ITEMS if i["type"] == kind] for kind in ("complete_snippet", "fix_bug", "debug_line")}


def test_every_item_fits_the_contract_and_the_public_view():
    ids = [i["item_id"] for i in ITEMS]
    assert len(ids) == len(set(ids))
    for item in ITEMS:
        CodeItem.model_validate(item)
        public = PublicCodeItem.model_validate({k: item[k] for k in PublicCodeItem.model_fields if k in item})
        dumped = public.model_dump()
        for secret in ("planted", "bug_lines", "fix", "op_id", "explain", "exposes"):
            assert secret not in dumped


def test_targets_of_05_3_2():
    for pid in PROBLEMS:
        n = sum(1 for i in BY_TYPE["complete_snippet"] if i["problem_id"] == pid)
        assert 1 <= n <= 2, pid
    for cls in MISCONCEPTIONS:
        assert sum(1 for i in BY_TYPE["fix_bug"] if i["planted"] == cls) == 2, cls
        assert sum(1 for i in BY_TYPE["debug_line"] if i["planted"] == cls) == 2, cls
    nulls = [i for i in BY_TYPE["debug_line"] if i["planted"] is None]
    assert 0.15 <= len(nulls) / len(BY_TYPE["debug_line"]) <= 0.25
    assert all(i["bug_lines"] == [] and i["op_id"] is None and i["fix"] is None for i in nulls)


def test_op_ids_are_real_and_match_the_class():
    labels = C.op_table()
    for item in BY_TYPE["fix_bug"] + BY_TYPE["debug_line"]:
        if item["planted"] is None:
            continue
        assert item["op_id"] in labels
        assert item["planted"] in labels[item["op_id"]]
        assert not item["op_id"].startswith("amb_")          # ambiguous twins would plant two classes
        assert item["bug_lines"]


def test_bug_lines_are_the_diff_and_the_fix_passes():
    for kind in ("fix_bug", "debug_line"):
        for cls in MISCONCEPTIONS:
            item = next(i for i in BY_TYPE[kind] if i["planted"] == cls)
            problem = PROBLEMS[item["problem_id"]]
            code = item.get("starter") or item["code"]
            assert C.fails_cleanly(problem, code), item["item_id"]
            assert all(1 <= n <= len(code.split("\n")) for n in item["bug_lines"])
            if kind == "debug_line":
                assert C.passes(problem, item["fix"]["code"]), item["item_id"]
                assert C.diff_lines(item["fix"]["code"], code)[0] == item["bug_lines"]


def test_no_bug_items_are_correct_variants():
    for item in (i for i in BY_TYPE["debug_line"] if i["planted"] is None):
        problem = PROBLEMS[item["problem_id"]]
        assert C.passes(problem, item["code"])
        assert any(C.same_tree(item["code"], v) for v in problem["correct_variants"])


def test_complete_snippet_holes():
    sample = BY_TYPE["complete_snippet"][::5]
    assert sample
    for item in sample:
        problem = PROBLEMS[item["problem_id"]]
        assert item["starter"].count(C.HOLE) == len(item["holes"])
        answers = item["fix"]["answers"]
        pieces = item["starter"].split(C.HOLE)
        right = [answers[h["id"]] for h in item["holes"]]
        filled = "".join(p + (right[k] if k < len(right) else "") for k, p in enumerate(pieces))
        assert filled == item["fix"]["code"]
        assert C.passes(problem, filled), item["item_id"]
        for k, hole in enumerate(item["holes"]):
            assert HOLE_LINE_OK(item, hole)
            assert answers[hole["id"]] in hole["choices"]
            for wrong in (c for c in hole["choices"] if c != right[k]):
                values = list(right)
                values[k] = wrong
                code = "".join(p + (values[j] if j < len(values) else "") for j, p in enumerate(pieces))
                assert C.fails_cleanly(problem, code), (item["item_id"], hole["id"], wrong)
        assert item["exposes"] and set(item["exposes"]) <= set(MISCONCEPTIONS)


def HOLE_LINE_OK(item, hole):
    return C.HOLE in item["starter"].split("\n")[hole["line"] - 1]


def test_tidy_keeps_the_syntax_tree():
    for pid in ("P03", "Q03", "Q06", "Q12"):
        for variant in PROBLEMS[pid]["correct_variants"]:
            printed = C.reprint(variant)
            neat = C.tidy(printed)
            assert C.same_tree(neat, printed)
            assert "\n\n" not in neat
            assert not re.search(r"\)\s*\n\s*\{", neat)           # braces are on the line of their statement


def test_diff_lines():
    a = "int f() {\n    int t = 0;\n    return t;\n}"
    b = "int f() {\n    int t;\n    return t;\n}"
    assert C.diff_lines(a, b) == ([2], [2])
    c = "int f() {\n    int t = 0;\n    t = 0;\n    return t;\n}"
    assert C.diff_lines(a, c)[0] == [3]


def test_find_holes_on_p03():
    kinds = {h["kind"] for h in C.find_holes(PROBLEMS["P03"]["correct_variants"][0])}
    assert {"acc_init", "loop_init", "loop_rel", "index"} <= kinds


def test_s2_reads_these_items():
    """server/app/pipeline.code_item_rule is the consumer; it must accept what we wrote."""
    pipeline = pytest.importorskip("server.app.pipeline")
    item = BY_TYPE["fix_bug"][0]
    problem_ok = item["problem_id"]
    assert problem_ok in PROBLEMS
    ok = pipeline.code_item_rule(item, item["starter"], {"top": []}, True)
    assert ok["response"] == "correct" and ok["apply_diagnosis"] is False
    same = pipeline.code_item_rule(item, item["starter"], {"top": []}, False)
    assert same["response"] == "wrong" and same["bug_lines_unchanged"] is True
    other = "x = 1;\n" + item["starter"]
    moved = pipeline.code_item_rule(item, other, {"top": []}, False)
    assert moved["response"] == "wrong" and moved["bug_lines_unchanged"] is False
    snippet = pipeline.code_item_rule(BY_TYPE["complete_snippet"][0], "", {}, False)
    assert snippet["apply_diagnosis"] is True


def test_check_passes():
    """Re-runs every item on the interpreter and gcc (about a minute)."""
    assert C.check(verbose=False) == 0
