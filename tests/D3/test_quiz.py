"""D3: quiz selection and ask_reason on a collision option."""
import json
from pathlib import Path

import pytest

from ml.contracts import schemas as S
from ml.contracts.params import ITEM_GUESS_SLIP
from ml.learner.knowledge import update_p
from ml.quiz.select import next_item, score_answer

ROOT = Path(__file__).resolve().parents[2]

COLLISION = {
    "item_id": "qz_hand_collision",
    "type": "mcq",
    "concept": "loops",
    "classes": ["M01", "M08"],
    "code": "int x = 0;",
    "question": "What is x?",
    "options": ["0", "1", "2"],
    "correct": "0",
    "belief": {"M01": "1", "M08": "1"},
    "explain": "x was set to 0.",
    "difficulty": 1,
}

SINGLE = {
    **COLLISION,
    "item_id": "qz_hand_single",
    "belief": {"M01": "1", "M08": "2"},
}

PREDICT = {
    **COLLISION,
    "item_id": "qz_hand_predict",
    "type": "predict_output",
    "classes": ["M01"],
    "belief": {"M01": "1"},
}

STATE = {"learner_id": "ada", "p_active": {"M01": 0.4, "M08": 0.4}}


def test_collision_option_asks_for_a_reason():
    scored = score_answer(STATE, COLLISION, "1")
    assert scored["ask_reason"] is True
    assert scored["correct"] is False
    for update in scored["updates"]:
        S.ClassUpdate.model_validate(update)


def test_single_class_option_does_not_ask_for_a_reason():
    shared = score_answer(STATE, SINGLE, "1")
    correct = score_answer(STATE, SINGLE, "0")
    assert shared["ask_reason"] is False
    assert correct["ask_reason"] is False
    assert correct["correct"] is True


def test_predict_output_with_a_belief_uses_the_trap_row():
    predicted = score_answer(STATE, PREDICT, "1")
    asked = score_answer(STATE, {**PREDICT, "type": "mcq", "item_id": "qz_hand_mcq"}, "1")
    trap = update_p(0.4, "trap", correct=False)
    mcq = update_p(0.4, "mcq", correct=False, belief={"M01": "1"}, class_id="M01")
    assert predicted["updates"][0]["p_after"] == pytest.approx(trap)
    assert asked["updates"][0]["p_after"] == pytest.approx(mcq)
    assert ITEM_GUESS_SLIP["trap"] != ITEM_GUESS_SLIP["belief_mcq"]
    assert predicted["updates"][0]["p_after"] != pytest.approx(asked["updates"][0]["p_after"])
    assert predicted["ask_reason"] is False


def test_next_item_is_public_and_names_the_classes():
    public, reason = next_item(STATE, [COLLISION])
    S.PublicQuizItem.model_validate(public)
    assert public["options"] == ["0", "1", "2"]
    assert "correct" not in public
    assert "belief" not in public
    assert "explain" not in public
    assert "Boundary Drift" in reason
    assert "Index Origin Fault" in reason
    filtered, _ = next_item(STATE, [COLLISION, SINGLE], type="mcq", concept="loops")
    assert filtered["item_id"] in ("qz_hand_collision", "qz_hand_single")


def test_real_quiz_bank_if_present():
    path = ROOT / "ml" / "data" / "quiz_items.json"
    if not path.exists():
        pytest.skip("quiz bank absent")
    items = json.loads(path.read_text(encoding="utf-8"))
    public, reason = next_item(
        {"learner_id": "ada", "p_active": {"M01": 0.5, "M08": 0.5}},
        items,
    )
    S.PublicQuizItem.model_validate(public)
    assert "correct" not in public
    assert "belief" not in public
    assert "explain" not in public
    assert reason.strip()
    source = next(item for item in items if item["item_id"] == public["item_id"])
    assert public["options"] == source.get("options") or []
