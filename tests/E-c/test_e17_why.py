"""E17 ("why?" on collision items), package E-c.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/E-c

No __init__.py in this folder on purpose ("E-c" is not a package name), so test file names must
be unique across the repo.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ml.bayes import update
from ml.contracts.classes import LABELS, REASON_LABELS
from ml.eval import e17_why as e17

ROOT = Path(__file__).resolve().parents[2]
ITEMS = json.loads((ROOT / "ml" / "data" / "quiz_items.json").read_text(encoding="utf-8"))


class Fake:
    """A reader that is sure of the label written in the sentence ("M01 ..."), or sure of nothing."""
    version = "fake"

    def __init__(self, threshold=0.5, sure=True):
        self.threshold, self.sure = threshold, sure

    def probs(self, texts):
        out = np.full((len(texts), len(REASON_LABELS)), 1.0 / len(REASON_LABELS))
        if self.sure:
            for i, text in enumerate(texts):
                out[i] = 0.01
                out[i, REASON_LABELS.index(text.split()[0])] = 1.0 - 0.01 * (len(REASON_LABELS) - 1)
        return out


def pool(classes, per=4):
    return [{"id": f"{c}-{i}", "label": c, "context_id": f"{c}_c{i % 2}", "text": f"{c} sentence {i}"} for c in classes for i in range(per)]


def groups():
    return e17.collision_groups(ITEMS)


def test_there_are_collision_groups_with_two_or_more_mistakes():
    found = groups()
    assert len({item["item_id"] for item, _, _ in found}) >= 12      # 05 §3.1 target
    for item, answer, group in found:
        assert len(group) >= 2 and answer != item["correct"] and answer in item["options"]
        assert all(item["belief"][c] == answer for c in group)


def test_option_alone_cannot_separate_the_group():
    for item, answer, group in groups():
        post = e17.posterior_after_option(item, answer)
        tied = [post[c] for c in group]
        assert max(tied) - min(tied) < 1e-12
        assert e17.tie_credit(post, group[0], group) == pytest.approx(1.0 / len(group))
        assert abs(sum(post.values()) - 1.0) < 1e-9


def test_posterior_uses_the_bayes_layer_update():
    item, answer, _ = groups()[0]
    post = e17.posterior_after_option(item, answer)
    again = update(e17.prior({label: 1.0 / len(LABELS) for label in LABELS}), answer,
                   correct=item["correct"], belief=item["belief"], options=item["options"])
    assert post == again


def test_tie_credit():
    post = {label: 0.0 for label in LABELS}
    post["M01"] = post["M08"] = 0.5
    assert e17.tie_credit(post, "M01") == 0.5
    assert e17.tie_credit(post, "M02") == 0.0
    post["M01"] = 0.6
    post["M08"] = 0.4
    assert e17.tie_credit(post, "M01") == 1.0
    assert e17.tie_credit(post, "M08") == 0.0
    assert e17.tie_credit(post, "M08", ["M08", "M02"]) == 1.0 or e17.tie_credit(post, "M08", ["M08", "M02"]) == 0.0


def test_text_probs_maps_correct_reason_to_correct_and_covers_all_labels():
    row = [0.0] * len(REASON_LABELS)
    row[-1] = 1.0
    mapped = e17.text_probs(row)
    assert set(mapped) == set(LABELS)
    assert mapped["CORRECT"] == 1.0 and mapped["OTHER"] == 0.0


def test_a_sure_reader_settles_every_pair():
    found = groups()
    classes = sorted({c for _, _, g in found for c in g})
    trials = e17.run_trials(Fake(sure=True), found, {"p": pool(classes)})
    assert trials and all(t["status"] == "matched" for t in trials)
    summary = e17.summarise(trials)
    assert summary["option_group"] == pytest.approx(0.5)
    assert summary["both_group"] == pytest.approx(1.0)
    assert summary["gain_group"] == pytest.approx(0.5)
    assert summary["sentence_only_group"] == pytest.approx(1.0)


def test_an_unsure_reader_changes_nothing():
    found = groups()
    classes = sorted({c for _, _, g in found for c in g})
    trials = e17.run_trials(Fake(sure=False, threshold=0.9), found, {"p": pool(classes)})
    assert all(t["status"] == "unsure" for t in trials)
    assert all(t["both_group"] == t["option_group"] for t in trials)
    assert e17.summarise(trials)["gain_group"] == 0.0


def test_trials_only_use_sentences_of_the_classes_in_the_group():
    found = groups()
    classes = sorted({c for _, _, g in found for c in g})
    trials = e17.run_trials(Fake(), found, {"p": pool(classes + ["D04", "D03"])})
    assert all(t["truth"] in t["group"] for t in trials)
    assert len(trials) == sum(len(g) for _, _, g in found) * 4


def test_summary_carries_the_three_intervals():
    found = groups()
    classes = sorted({c for _, _, g in found for c in g})
    s = e17.summarise(e17.run_trials(Fake(), found, {"p": pool(classes)}))
    for key in ("gain_group_ci_contexts", "gain_group_ci_items", "gain_group_ci_trials"):
        assert len(s[key]) == 2 and s[key][0] <= s[key][1]


def test_card_on_disk_is_consistent_and_says_the_sentences_are_llm_written():
    path = ROOT / "ml" / "eval" / "e17_card.json"
    if not path.exists():
        pytest.skip("run python -m ml.eval.e17_why first")
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["id"] == "E17"
    assert "LLM-written" in card["caveat"] and "not collected" not in card["caveat"].lower() or "no real student" in card["caveat"]
    loaded = [r for r in card["summary"] if r.get("loaded")]
    assert loaded
    for r in loaded:
        assert r["option_group"] == pytest.approx(0.5)          # a tie among two mistakes
        assert r["gain_group"] == pytest.approx(r["both_group"] - r["option_group"], abs=2e-4)
    assert card["n_items"] >= 12
    assert any("not written for" in line.lower() or "written for these items" in line.lower() for line in card["not_measured"])
