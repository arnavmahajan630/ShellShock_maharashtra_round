"""Package B4: the quiz bank, its builder and its verifier."""
import copy

import pytest

from ml.contracts.classes import MISCONCEPTIONS
from ml.items import build_quiz as B
from ml.items import verify_quiz as V
from ml.items.draft_quiz_llm import GUIDE, draft_messages, explain_key
from ml.oracle import gcc

try:
    gcc.gcc_path()
    HAVE_GCC = True
except gcc.GccUnavailable:
    HAVE_GCC = False

ITEMS, CHECKS = V.load()


def changed(item_id, **fields):
    items = copy.deepcopy(ITEMS)
    next(item for item in items if item["item_id"] == item_id).update(fields)
    return items


def first(kind):
    return next(item for item in ITEMS if item["type"] == kind and len(item["classes"]) == 1)


# ---------------------------------------------------------------- the bank

def test_structure_is_sound():
    assert V.structure_problems(ITEMS, CHECKS, V.other_bank_code()) == []


def test_targets_of_the_plan_are_met():
    assert V.target_problems(ITEMS) == []


def test_every_answer_holds_on_the_interpreter():
    problems, summary = V.run_problems(ITEMS, CHECKS, backends=("interp",))
    assert problems == []
    assert summary["run"] == summary["items"] - len(summary["manual"])
    assert summary["collisions"] >= V.MIN_COLLISIONS


@pytest.mark.skipif(not HAVE_GCC, reason="gcc is not installed")
def test_every_answer_holds_on_gcc_where_c_defines_it():
    problems, summary = V.run_problems(ITEMS, CHECKS)
    assert problems == []
    assert summary["on_gcc"] + summary["interp_only"] == summary["run"]


def test_no_item_is_marked_manual_without_reason():
    assert [item["item_id"] for item in ITEMS if item["verified"]["manual"]] == []


def test_public_fields_do_not_leak_answers():
    from ml.contracts.schemas import PublicQuizItem
    public = set(PublicQuizItem.model_fields)
    assert not public & {"correct", "belief", "explain", "classes", "expected"}


# ---------------------------------------------------------------- the verifier catches

def test_a_wrong_correct_answer_is_caught():
    item = first("predict_output")
    other = next(o for o in item["options"] if o != item["correct"] and o not in item["belief"].values())
    problems, _ = V.run_problems(changed(item["item_id"], correct=other), CHECKS, backends=("interp",))
    assert any(item["item_id"] in line and "correct" in line for line in problems)


def test_a_wrong_belief_answer_is_caught():
    item = first("mcq")
    cls = item["classes"][0]
    other = next(o for o in item["options"] if o not in (item["correct"], item["belief"][cls]))
    problems, _ = V.run_problems(changed(item["item_id"], belief={cls: other}), CHECKS, backends=("interp",))
    assert any(item["item_id"] in line and f"belief of {cls}" in line for line in problems)


def test_a_wrong_state_query_is_caught():
    item = first("next_state")
    query = dict(item["state_query"], hit=item["state_query"]["hit"] + 40)
    problems, _ = V.run_problems(changed(item["item_id"], state_query=query), CHECKS, backends=("interp",))
    assert any(item["item_id"] in line for line in problems)


def test_structure_rules_catch():
    item = first("predict_output")
    item_id, cls = item["item_id"], item["classes"][0]
    cases = [
        ({"options": item["options"][:2]}, "different options"),
        ({"correct": "no such option"}, "is not an option"),
        ({"belief": {cls: item["correct"]}}, "equals the correct answer"),
        ({"belief": {"M99": item["options"][0]}}, "is not in classes"),
        ({"belief": {}}, "no belief answer"),
        ({"concept": "pointers"}, "unknown concept"),
        ({"explain": " "}, "no explanation"),
        ({"classes": []}, "non-empty list"),
        ({"state_query": {"var": "i", "after_line": 1, "hit": 1}}, "only to them"),
        ({"surprise": 1}, "does not fit QuizItem"),
    ]
    for fields, expect in cases:
        problems = V.structure_problems(changed(item_id, **fields), CHECKS)
        assert any(expect in line for line in problems), (fields, problems)


def test_reasoning_items_have_no_options():
    item = first("reasoning")
    problems = V.structure_problems(changed(item["item_id"], options=["1", "2", "3"]), CHECKS)
    assert any("has no options" in line for line in problems)
    problems = V.structure_problems(changed(item["item_id"], expected="M01"), CHECKS)
    assert any("expects CORRECT_REASON" in line for line in problems)


def test_an_item_with_no_check_must_be_marked_manual():
    item = first("mcq")
    checks = {k: v for k, v in CHECKS.items() if k != item["item_id"]}
    assert any("not marked manual" in line for line in V.structure_problems(ITEMS, checks))
    flags = dict(item["verified"], manual=True)
    assert V.structure_problems(changed(item["item_id"], verified=flags), checks) == []


def test_quiz_code_may_not_repeat_a_probe_trap_or_exam_item():
    item = first("mcq")
    problems = V.structure_problems(ITEMS, CHECKS, {" ".join(item["code"].split()): "trap_M01"})
    assert any("same code as trap_M01" in line for line in problems)


def test_missing_targets_are_reported():
    without = [item for item in ITEMS if not (item["type"] == "next_state" and item["classes"] == ["M03"])]
    assert any("M03 has 0 next_state" in line for line in V.target_problems(without))
    singles = [item for item in ITEMS if len(item["classes"]) == 1]
    assert any("collision items" in line for line in V.target_problems(singles))


def test_collision_means_one_wrong_option_shared_by_two_classes():
    assert V.collision_answers({"correct": "4", "belief": {"M01": "3", "M08": "3"}}) == ["3"]
    assert V.collision_answers({"correct": "4", "belief": {"M01": "3", "M08": "5"}}) == []
    assert V.collision_answers({"correct": "3", "belief": {"M01": "3", "M08": "3"}}) == []


# ---------------------------------------------------------------- the builder

DRAFT = {"draft_id": "t1", "class": "M03", "type": "mcq", "question": "What is total after the loop?",
         "tail": 'printf("%d\\n", total);', "distractors": ["6", "0"], "difficulty": 1,
         "code": "int total = 0;\nfor (int i = 1; i <= 3; i++) {\n    total = 0;\n    total = total + i;\n}",
         "belief_code": "int total = 0;\nfor (int i = 1; i <= 3; i++) {\n    total = total + i;\n}"}


def test_builder_takes_answers_from_the_runs_not_from_the_draft():
    cand, reason = B.normalise(dict(DRAFT, correct="999", belief={"M03": "999"}))
    assert reason is None
    B.run_all([cand])
    assert cand["problem"] is None
    assert cand["correct"] == "3" and cand["belief"] == {"M03": "6"}
    options = B.options_for(cand)
    assert sorted(options) == ["0", "3", "6"]
    assert options == B.options_for(cand)                 # same order every time


@pytest.mark.parametrize("fields, expect", [
    ({"belief_code": DRAFT["code"]}, "same answer"),
    ({"code": "int total = ;", "belief_code": "int total = ;"}, "does not run"),
    ({"tail": "total = 5;"}, "not one printf"),
    ({"code": DRAFT["code"] + '\nprintf("%d", total);'}, "mcq code prints"),
    ({"code": "\n".join(["int x = 1;"] * 20)}, "too long"),
    ({"belief_code": "int z = 9;\nwhile (z > 0) {\n    z = z - 2;\n}\nopen_door();"}, "different program"),
])
def test_builder_drops_bad_drafts(fields, expect):
    cand, reason = B.normalise(dict(DRAFT, **fields))
    if cand is not None:
        B.run_all([cand])
        reason = cand["problem"]
    assert reason and expect in reason


def test_builder_wants_proof_that_the_mistake_is_in_the_code():
    draft = dict(DRAFT, **{"class": "M06"})                 # this code has no `=` in a condition
    cand, _ = B.normalise(draft)
    B.run_all([cand])
    assert "assign_in_cond" in cand["problem"]


def test_next_state_line_must_be_a_plain_statement():
    draft = {"draft_id": "t2", "class": "M03", "type": "next_state", "var": "total", "hit": 2,
             "code": DRAFT["code"], "belief_code": DRAFT["belief_code"], "after_line": 2, "belief_after_line": 2}
    assert B.normalise(draft) == (None, "next_state line is not a plain statement")
    cand, reason = B.normalise(dict(draft, after_line=4, belief_after_line=3))
    assert reason is None
    B.run_all([cand])
    assert cand["correct"] == "2" and cand["belief"] == {"M03": "3"}
    assert cand["question"] == "What is total right after line 4 has run for the 2nd time?"


def test_reasoning_question_states_the_real_output():
    assert B.reasoning_question("1", "5") == 'This prints "1". A classmate expected "5". In one sentence: why?'
    assert "never stops" in B.reasoning_question("never stops", "3")
    assert "cannot be predicted" in B.reasoning_question("unpredictable", "6")


# ---------------------------------------------------------------- the drafting prompts (no calls are made)

def test_every_class_has_a_drafting_guide_and_prompt():
    assert set(GUIDE) == set(MISCONCEPTIONS)
    for cls in MISCONCEPTIONS:
        for kind in ("predict_output", "mcq", "next_state"):
            text = draft_messages(cls, kind)[1]["content"]
            assert GUIDE[cls]["shows"] in text and "Do not tell me any result" in text


def test_explanations_are_keyed_by_content_not_by_id():
    item = first("mcq")
    assert explain_key(item) == explain_key(dict(item, item_id="renamed"))
