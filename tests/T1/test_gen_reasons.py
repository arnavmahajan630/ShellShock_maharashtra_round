"""Checks the reason-sentence generator and its output without calling DeepSeek.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/T1
"""
import json
from collections import Counter, defaultdict

from ml.contracts import schemas as S
from ml.contracts.classes import MISCONCEPTIONS, REASON_LABELS
from ml.text import gen_reasons


def load_contexts():
    return json.loads(gen_reasons.CONTEXTS.read_text(encoding="utf-8"))


def test_six_contexts_per_class_with_a_real_wrong_answer():
    data = load_contexts()
    per_class = Counter(c["cls"] for c in data["contexts"])
    assert set(per_class) == set(MISCONCEPTIONS) and set(per_class.values()) == {6}
    assert set(data["correct_rules"]) == set(MISCONCEPTIONS)
    assert len({c["id"] for c in data["contexts"]}) == len(data["contexts"])
    for context in data["contexts"]:
        assert context["chosen"] != context["correct"], context["id"]


def test_plan_covers_every_class_context_and_voice():
    data = load_contexts()
    calls = gen_reasons.plan(data["contexts"], data["correct_rules"])
    wrong = [c for c in calls if c[0] != "CORRECT_REASON"]
    assert len(wrong) == 17 * 6 * len(gen_reasons.VOICES)
    assert {(c[0], c[1]["id"], c[2]) for c in wrong} == {
        (ctx["cls"], ctx["id"], voice) for ctx in data["contexts"] for voice in gen_reasons.VOICES}
    correct_splits = {gen_reasons.SPLIT_BY_CONTEXT_NUMBER[int(c[1]["id"].rsplit("_c", 1)[1])]
                      for c in calls if c[0] == "CORRECT_REASON"}
    assert correct_splits == {"train", "val", "test"}


def test_prompt_gives_the_belief_but_never_the_class_id():
    data = load_contexts()
    for label, context, _, messages in gen_reasons.plan(data["contexts"], data["correct_rules"])[:40]:
        text = messages[1]["content"]
        assert context["code"] in text and "JSON" in text
        assert label not in text


def test_filters():
    assert gen_reasons.keep("because 4", "M01") == "under 3 words"
    assert gen_reasons.keep("this is the boundary drift thing again", "M01") is not None
    assert gen_reasons.keep("the variable is uninitialized so garbage", "M05") is not None
    assert gen_reasons.keep("Local variables start at 0", "M05") == "repeats the description"
    assert gen_reasons.keep("c starts at zero so after plus plus it is 1", "M05") is None


def test_generated_sentences():
    rows = [S.ReasonRow.model_validate(json.loads(line))
            for line in gen_reasons.OUT.read_text(encoding="utf-8").splitlines()]
    per_label, contexts, split_of = Counter(), defaultdict(set), {}
    for row in rows:
        per_label[row.label] += 1
        contexts[row.label].add(row.context_id)
        assert split_of.setdefault(row.context_id, row.split) == row.split, "a context is in two splits"
        assert row.source == "deepseek"
    assert set(per_label) == set(REASON_LABELS)
    for label in MISCONCEPTIONS:
        assert per_label[label] >= 150 and len(contexts[label]) == 6
    assert len({(r.label, r.text) for r in rows}) == len(rows)

    bundle = json.loads(gen_reasons.BUNDLE.read_text(encoding="utf-8"))
    assert bundle["labels"] == REASON_LABELS and set(bundle["descriptions"]) == set(REASON_LABELS)
    assert len(bundle["rows"]) == len(rows)
