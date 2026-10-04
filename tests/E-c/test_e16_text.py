"""E16 (sentence reader experiment), package E-c.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/E-c

The folder has no __init__.py on purpose: "E-c" is not a valid package name, and pytest then
imports this file by its own name.
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from ml.contracts.classes import MISCONCEPTIONS, REASON_LABELS
from ml.eval import e16_text as e16
from ml.text import serve

ROOT = Path(__file__).resolve().parents[2]
INDEX = {label: i for i, label in enumerate(REASON_LABELS)}


def fake_embed(texts, dim=64):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        padded = f" {str(text).lower()} "
        for start in range(len(padded) - 2):
            out[row, zlib.crc32(padded[start:start + 3].encode("utf-8")) % dim] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class Keyword:
    """A reader that looks for a label's id written in the sentence."""
    version = "fake"

    def __init__(self, name, threshold=0.5, confidence=0.9):
        self.name, self.threshold, self.confidence = name, threshold, confidence

    def probs(self, texts):
        out = np.full((len(texts), len(REASON_LABELS)), (1 - self.confidence) / (len(REASON_LABELS) - 1))
        for row, text in enumerate(texts):
            found = [i for label, i in INDEX.items() if label in text]
            if found:
                out[row, found[0]] = self.confidence
            else:
                out[row] = 1 / len(REASON_LABELS)
        return out


def row(label, text, voice="confident", split="test", context=None, code=""):
    return {"id": f"x-{label}-{abs(hash(text)) % 10_000}", "label": label, "context_id": context or f"{label}_c6",
            "code": code, "chosen": "1", "text": text, "voice": voice, "source": "deepseek", "split": split}


@pytest.fixture()
def small():
    test = [row(label, f"this shows {label}", voice) for label in REASON_LABELS for voice in ("confident", "hinglish")]
    test += [row("M05", "nothing useful", "typos"), row("M02", "this shows M03", "typos")]
    persona = [row(label, f"this shows {label}", voice, context=f"{label}_c5")
               for label in REASON_LABELS[:6] for voice in ("persona_friend", "persona_topper", "persona_topper_told")]
    mohler = [{"question_id": "1.1", "question": "q", "reference": "r", "answer": answer, "score": score}
              for answer, score in [("a stack is last in first out", 5.0), ("looks like M08 to me", 2.0),
                                    ("CORRECT_REASON maybe", 4.5), ("a queue", 3.0)]]
    return test, persona, mohler


# ---------------------------------------------------------------- pieces

def test_loco_numbers_are_the_ones_in_the_notes():
    notes = (ROOT / "notes" / "T1.md").read_text(encoding="utf-8")
    line = next(l for l in notes.splitlines() if l.startswith("Per class top-1:"))
    in_notes = {label: float(value) for label, value in re.findall(r"([MD]\d\d) (\d\.\d+)", line)}
    assert in_notes == e16.LOCO_TOP1 and set(in_notes) == set(MISCONCEPTIONS)
    assert "| Mean over 17 classes | 0.120 | 0.283 |" in notes
    assert e16.LOCO_MEAN == {"top1": 0.120, "top3": 0.283}
    assert round(sum(e16.LOCO_TOP1.values()) / 17, 3) == 0.120
    assert "| Best (D02) | 0.311 | 0.600 |" in notes and e16.LOCO_TOP3["D02"] == 0.600
    assert "| Worst (M01, M07) | 0.000 | 0.017 / 0.067 |" in notes
    assert (e16.LOCO_TOP3["M01"], e16.LOCO_TOP3["M07"]) == (0.017, 0.067)


def test_language_and_base_voice():
    assert e16.language_of("hinglish") == "Hinglish" and e16.language_of("persona_friend_told") == "Hinglish"
    assert e16.language_of("typos") == "English" and e16.language_of("persona_topper") == "English"
    assert e16.base_voice("persona_friend_told") == "persona_friend" and e16.base_voice("typos") == "typos"
    persona = e16.load_jsonl(e16.PERSONA)
    friend = [r for r in persona if r["voice"].startswith("persona_friend")]
    assert len(friend) == 45 and any("hai" in r["text"].lower().split() for r in friend)


def test_measure_counts_answers_and_accuracy(small):
    test, _, _ = small
    reader = Keyword("tfidf")
    out = e16.measure(reader.probs([r["text"] for r in test]), test, reader.threshold, interval=True)
    assert out["n"] == 38
    assert out["accuracy"] == pytest.approx(36 / 38, abs=1e-6)      # two rows were written to be wrong
    assert out["answers"] == pytest.approx(37 / 38, abs=1e-6)       # "nothing useful" stays under the threshold
    assert out["right_when_it_answers"] == pytest.approx(36 / 37, abs=1e-6)
    low, high = out["accuracy_ci95"]
    assert low <= out["accuracy"] <= high
    assert e16.measure(np.zeros((0, 18)), [], 0.5) == {"n": 0}


def test_persona_slices_split_on_told(small):
    _, persona, _ = small
    slices = {name: keep for name, _, keep in e16.slices_of("persona", persona)}
    assert len(slices["persona"]) == 18 and len(slices["persona_told"]) == 6 and len(slices["persona_picked_by_itself"]) == 12
    assert all(persona[i]["voice"].endswith("_told") for i in slices["persona_told"])
    assert all(label.startswith(e16.PERSONA_LABEL) for _, label, _ in e16.slices_of("persona", persona))


def test_mohler_row_counts_the_three_answers(small):
    _, _, mohler = small
    reader = Keyword("tfidf")
    out = e16.mohler_row("tfidf", reader.probs([m["answer"] for m in mohler]), reader.threshold, mohler)
    assert (out["unsure"], out["matched_wrongly"], out["correct_reasoning"]) == (0.5, 0.25, 0.25)
    assert out["most_often_matched"] == [["M08", 0.25]]
    assert out["graded_4_or_more"] == {"n": 2, "unsure": 0.5, "matched_wrongly": 0.0}
    assert out["graded_under_4"] == {"n": 2, "unsure": 0.5, "matched_wrongly": 0.5}


def test_nearest_description_finds_a_sentence_that_repeats_its_description():
    descriptions = {label: f"{label} " + " ".join(chr(97 + (i * 7 + k) % 26) * 5 for k in range(4))
                    for i, label in enumerate(REASON_LABELS)}
    rows = [row(label, descriptions[label], split=split) for label in REASON_LABELS for split in ("train", "test")]
    per_class, overall = e16.nearest_description(fake_embed, descriptions, rows)
    assert all(per_class[label]["top1_all"] == 1.0 and per_class[label]["n_all"] == 2 for label in REASON_LABELS)
    assert all(per_class[label]["n_test"] == 1 for label in REASON_LABELS)
    assert overall["top1_test"] == 1.0 and overall["n_test"] == 18 and overall["n_all"] == 36


def test_code_masking_row_uses_the_readers_own_masking():
    code = "int total(int a[], int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) s += a[i];\n    return s;\n}"
    rows = [row("M01", "x", code=code), row("M01", "y", code="for (i = 0; i < 3; i++) fire();")]
    probs = np.zeros((2, 18))
    probs[:, INDEX["D05"]], probs[:, INDEX["M01"]] = 0.6, 0.4
    masked, parsed = e16.masked_by_code(probs, rows)
    assert parsed == 1
    assert masked[0, INDEX["D05"]] == 0.0 and masked[0, INDEX["M01"]] == pytest.approx(1.0)
    assert masked[1].tolist() == probs[1].tolist()


# ---------------------------------------------------------------- the card

def test_card_from_fake_readers_has_every_row(small):
    test, persona, mohler = small
    descriptions = {label: f"description of {label}" for label in REASON_LABELS}
    readers = {"tfidf": Keyword("tfidf"), "biencoder": None, "frozen": Keyword("frozen", threshold=0.95)}
    card = e16.build_card(readers, test, persona, mohler, embed=fake_embed, descriptions=descriptions,
                          descriptions_from="a test", all_rows=test + persona)
    assert {"id", "title", "slice", "n", "metrics", "per_class", "plots", "caveat"} <= set(card)     # 03 §9.4
    assert card["id"] == "E16" and card["n"] == 38 and card["default_reader"] == "tfidf"
    assert card["metrics"]["reader"] == "tfidf" and card["metrics"]["accuracy"] == pytest.approx(36 / 38, abs=1e-6)
    assert card["readers"]["biencoder"] == {"loaded": False} and card["readers"]["frozen"]["threshold"] == 0.95
    assert {(r["reader"], r["set"]) for r in card["rows"]} == {
        (reader, name) for reader in ("tfidf", "frozen")
        for name in ("test", "persona", "persona_picked_by_itself", "persona_told")}
    frozen_test = next(r for r in card["rows"] if r["reader"] == "frozen" and r["set"] == "test")
    assert frozen_test["answers"] == 0.0 and frozen_test["right_when_it_answers"] is None     # threshold above its confidence
    assert {v["voice"] for v in card["per_voice"] if v["set"] == "persona"} == {"persona_friend", "persona_topper"}
    assert {(l["set"], l["language"]) for l in card["language"]} == {
        (name, language) for name in ("test", "persona") for language in ("English", "Hinglish")}
    assert set(card["per_class"]) == set(REASON_LABELS) and card["per_class"]["M03"]["n"] == 2
    assert [m["reader"] for m in card["mohler"]["rows"]] == ["tfidf", "frozen"] and card["mohler"]["n"] == 4
    loco = card["leave_one_class_out"]
    assert loco["source"] == "run on Colab on 2026-10-04, copied from notes/T1.md"
    assert set(loco["per_class"]) == set(MISCONCEPTIONS)
    assert loco["per_class"]["M04"]["finetuned_top1"] == 0.217 and "unchanged_top1_all" in loco["per_class"]["M04"]
    assert card["not_run"][0]["row"] == "DeepSeek zero-shot"
    json.dumps(card)

    text = e16.to_markdown(card)
    for needed in ("LLM-written stand-in for classmates' sentences", "run on Colab on 2026-10-04, copied from notes/T1.md",
                   "DeepSeek zero-shot", "Not run", "Only Mohler is real student text", "Hinglish", "biencoder | no"):
        assert needed in text, needed


def test_card_without_the_big_files_says_what_was_skipped(small):
    test, persona, _ = small
    card = e16.build_card({"tfidf": Keyword("tfidf"), "biencoder": None, "frozen": None}, test, persona, [])
    assert card["mohler"]["rows"] == [] and card["mohler"]["what"].startswith("not run")
    assert card["leave_one_class_out"]["unchanged"].startswith("not computed")
    assert card["leave_one_class_out"]["per_class"]["D02"] == {"finetuned_top1": 0.311, "finetuned_top3": 0.6}
    assert "not computed" in e16.to_markdown(card)


def test_card_with_no_reader_at_all_still_builds(small):
    test, persona, mohler = small
    card = e16.build_card({"tfidf": None, "biencoder": None, "frozen": None}, test, persona, mohler)
    assert card["default_reader"] is None and card["rows"] == [] and card["n"] == 0
    assert "E16" in e16.to_markdown(card)


# ---------------------------------------------------------------- the committed card

@pytest.fixture(scope="module")
def committed():
    return json.loads(e16.CARD_JSON.read_text(encoding="utf-8"))


def test_committed_card_says_what_the_sets_are(committed):
    assert committed["id"] == "E16" and committed["default_reader"] == "tfidf"
    assert committed["caveat"] == e16.CAVEAT and "LLM-written" in committed["caveat"] and "Mohler" in committed["caveat"]
    for entry in committed["rows"]:
        if entry["set"].startswith("persona"):
            assert entry["set_label"].startswith("LLM-written stand-in for classmates' sentences")
    assert committed["leave_one_class_out"]["source"] == "run on Colab on 2026-10-04, copied from notes/T1.md"
    assert [item["row"] for item in committed["not_run"]] == ["DeepSeek zero-shot"]
    text = e16.CARD_MD.read_text(encoding="utf-8")
    assert "human" not in text.lower() and "human" not in json.dumps(committed).lower()
    assert committed["caveat"] in text


def test_committed_card_matches_the_committed_word_count_head(committed):
    """The card is not stale: tfidf's rows are what the committed head gives now."""
    reader = serve.load_reader("tfidf")
    assert committed["readers"]["tfidf"]["version"] == reader.version
    assert committed["readers"]["tfidf"]["threshold"] == pytest.approx(reader.threshold)
    test = [r for r in e16.load_jsonl(e16.DATA) if r["split"] == "test"]
    persona = e16.load_jsonl(e16.PERSONA)
    for name, rows in (("test", test), ("persona", persona)):
        now = e16.measure(reader.probs([r["text"] for r in rows]), rows, reader.threshold)
        then = next(r for r in committed["rows"] if r["reader"] == "tfidf" and r["set"] == name)
        for key in ("n", "accuracy", "macro_f1", "top3", "answers", "right_when_it_answers"):
            assert then[key] == pytest.approx(now[key], abs=1e-6), (name, key)


def test_committed_card_agrees_with_the_earlier_check(committed):
    """notes/T1.md, "Readers on both sets": the same readers on the same sets give the same numbers."""
    expected = {("tfidf", "test"): (539, 0.720, 0.876), ("tfidf", "persona"): (270, 0.730, 0.900)}
    if committed["readers"]["biencoder"]["loaded"]:
        expected.update({("biencoder", "test"): (539, 0.718, 0.891), ("biencoder", "persona"): (270, 0.707, 0.919),
                         ("biencoder", "persona_picked_by_itself"): (163, 0.638, 0.902)})
    for (reader, name), (n, accuracy, top3) in expected.items():
        entry = next(r for r in committed["rows"] if r["reader"] == reader and r["set"] == name)
        assert entry["n"] == n
        assert entry["accuracy"] == pytest.approx(accuracy, abs=0.0006) and entry["top3"] == pytest.approx(top3, abs=0.0006)
