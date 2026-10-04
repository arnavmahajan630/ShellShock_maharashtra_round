"""`read()` with the word-count reader, the load order, odd input and the response helper.

Run from the repo root:  .venv\\Scripts\\python -m pytest tests/T2
"""
import json

import numpy as np
import pytest

from ml.contracts import schemas as S
from ml.contracts.classes import REASON_LABELS
from ml.text import check_readers, fit_heads, serve
from tests.T2.conftest import check_shape

INDEX = {label: i for i, label in enumerate(REASON_LABELS)}


# ---------------------------------------------------------------- the 60-row sample

def test_tfidf_reads_the_sample(sample_rows, sample_heads, empty_dir):
    right = 0
    for row in sample_rows:
        result = check_shape(serve.read(row["code"], row["text"], heads_dir=sample_heads, artifacts_dir=empty_dir))
        assert result["reader"] == "tfidf"
        right += int(np.argmax(result["probs"])) == INDEX[row["label"]]
    assert right >= 0.8 * len(sample_rows)


def test_labels_missing_from_the_training_data_get_zero(sample_rows, sample_heads, empty_dir):
    seen = {row["label"] for row in sample_rows}
    result = serve.read(None, sample_rows[0]["text"], heads_dir=sample_heads, artifacts_dir=empty_dir)
    for label, p in zip(REASON_LABELS, result["probs"]):
        assert (p > 0) == (label in seen), label


def test_sample_head_uses_the_default_threshold(sample_heads):
    """The sample has no val split, so no threshold can be chosen there."""
    meta = json.loads((sample_heads / "tfidf.json").read_text(encoding="utf-8"))
    assert meta["threshold"] == fit_heads.DEFAULT_THRESHOLD and meta["val"] is None
    assert meta["threshold_rule"].startswith("default")


def test_status_follows_the_top_label_and_the_threshold(sample_rows, sample_heads, empty_dir):
    statuses = set()
    for row in sample_rows:
        result = serve.read(None, row["text"], heads_dir=sample_heads, artifacts_dir=empty_dir)
        top = int(np.argmax(result["probs"]))
        if result["probs"][top] < result["threshold"]:
            assert result["status"] == "unsure"
        elif REASON_LABELS[top] == "CORRECT_REASON":
            assert result["status"] == "correct_reasoning"
        else:
            assert result["status"] == "matched"
        statuses.add(result["status"])
    assert {"matched", "correct_reasoning"} <= statuses


def test_status_of():
    row = np.full(serve.N, 0.01)
    row[3] = 0.83
    assert serve.status_of(row, 0.8) == "matched"
    assert serve.status_of(row, 0.9) == "unsure"
    assert serve.status_of(row, serve.NEVER) == "unsure"
    row[3], row[serve.CORRECT_INDEX] = 0.01, 0.83
    assert serve.status_of(row, 0.8) == "correct_reasoning"
    assert serve.status_of(np.full(serve.N, np.nan), 0.0) == "unsure"


def test_pick_threshold_is_the_smallest_level_that_reaches_the_target():
    confidence = np.array([0.2] * 10 + [0.5] * 10 + [0.9] * 10)
    right = np.array([0] * 10 + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] + [1] * 10)
    probs = np.tile(((1 - confidence) / 9)[:, None], (1, 10))
    probs[:, 0] = confidence                                   # column 0 is always the top answer
    truth = np.where(right == 1, 0, 1)
    assert serve.pick_threshold(probs, truth) == 0.5          # 18 of 20 accepted are right
    assert serve.pick_threshold(probs, truth, target=0.95) == pytest.approx(0.9)
    assert serve.pick_threshold(probs, np.full(30, 2)) == serve.NEVER
    assert serve.pick_threshold(probs[:5], truth[:5]) == serve.NEVER     # fewer than 10 accepted


# ---------------------------------------------------------------- the committed head

def test_committed_tfidf_head_is_the_default_reader():
    result = check_shape(serve.read(None, "6 minus 2 is 4 so four shots"))
    assert result["reader"] == "tfidf"
    assert 0.0 < result["threshold"] < 1.0
    assert serve.model_version(result) == f"reason_tfidf_{result['version']}"


def test_committed_heads_are_plain_small_and_current():
    for name in ("tfidf", "frozen"):
        meta = json.loads((serve.HEADS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert meta["labels"] == REASON_LABELS
        assert meta["data"] == "ml/data/reasons.jsonl"
        assert meta["data_sha256"] == fit_heads.data_sha256(fit_heads.DATA), f"refit {name}: the sentences changed"
        assert meta["sentences"]["train"] == sum(r["split"] == "train" for r in fit_heads.load_rows(fit_heads.DATA))
        assert meta["val"]["right_when_it_answers"] >= serve.TARGET_ACCURACY or meta["threshold"] == serve.NEVER
        arrays = np.load(serve.HEADS_DIR / f"{name}.npz", allow_pickle=False)     # no pickles inside
        assert set(arrays["classes"]) == set(range(len(REASON_LABELS)))
        assert (serve.HEADS_DIR / f"{name}.npz").stat().st_size < 3_000_000


def test_committed_tfidf_head_gives_the_fitted_pipelines_numbers():
    """The stored arrays are the same reader as the pipeline in check_readers.py."""
    rows = fit_heads.load_rows(fit_heads.DATA)
    reference, _ = check_readers.tfidf_reader([r for r in rows if r["split"] == "train"])
    texts = [r["text"] for r in rows if r["split"] == "val"][:120] + ["", "loop apne aap ruk jayega", "ζ ω 数组"]
    ours = serve.TfidfReader().probs(texts)
    assert np.abs(ours - reference(texts)).max() < 1e-4


# ---------------------------------------------------------------- odd input

ODD = [None, "", "   \n\t ", "a", "?", "why " * 50_000, "x" * 300_000, "लूप अपने आप रुक जाता है",
       "loop apne aap ruk jayega na bhai, isliye 4", "数组从1开始 🙂", "\x00\x01�", 12345, 3.5, b"bytes",
       ["a", "list"], {"a": 1}]


@pytest.mark.parametrize("text", ODD, ids=[f"odd{i}" for i in range(len(ODD))])
def test_read_never_raises_on_odd_text(text):
    result = check_shape(serve.read(None, text))
    assert result["reader"] == "tfidf"


@pytest.mark.parametrize("text", [None, "", "   \n\t "])
def test_an_empty_sentence_is_unsure(text):
    result = serve.read(None, text)
    assert result["status"] == "unsure" and result["reader"] == "tfidf"
    assert serve.top_classes(result)[0]["p"] == pytest.approx(1 / serve.N, abs=1e-4)


ODD_CODE = [12, b"int", ["x"], "", "   ", "this is not C {{{", "int main( {", "x" * 100_000,
            "for (i = 2; i <= 6; i++) fire();", "// only a comment", "#include <stdio.h>"]


@pytest.mark.parametrize("code", ODD_CODE, ids=[f"code{i}" for i in range(len(ODD_CODE))])
def test_read_never_raises_on_odd_code_and_then_does_not_mask(code):
    plain = serve.read(None, "the loop stops before 6 so 2 3 4 5 only")
    result = check_shape(serve.read(code, "the loop stops before 6 so 2 3 4 5 only"))
    assert result["masked"] == [] and result["probs"] == plain["probs"]


def test_a_very_long_sentence_is_cut(monkeypatch):
    assert len(serve.clean("ab " * 10_000)) <= serve.MAX_CHARS
    assert serve.read(None, "the loop stops before 6 " + "z" * 100_000)["reader"] == "tfidf"


def test_a_reader_that_breaks_while_reading_gives_none(monkeypatch):
    class Broken:
        name, threshold, version = "tfidf", 0.5, "x"

        def probs(self, texts):
            raise RuntimeError("boom")
    monkeypatch.setattr(serve, "get_reader", lambda *args, **kwargs: Broken())
    result = check_shape(serve.read(None, "anything"))
    assert result["reader"] == "none" and result["status"] == "unsure"


# ---------------------------------------------------------------- load order

def test_order_is_tfidf_then_biencoder_then_frozen():
    assert serve.READER_ORDER == ("tfidf", "biencoder", "frozen")
    assert serve.reader_order() == ("tfidf", "biencoder", "frozen")
    assert serve.reader_order("frozen") == ("frozen", "tfidf", "biencoder")
    assert serve.reader_order("none") == ()
    assert serve.reader_order("no-such-reader") == serve.READER_ORDER


def test_nothing_loads_gives_none(empty_dir):
    result = check_shape(serve.read(None, "6 minus 2 is 4", heads_dir=empty_dir, artifacts_dir=empty_dir))
    assert result["reader"] == "none" and result["status"] == "unsure"
    assert result["probs"] == pytest.approx([1 / serve.N] * serve.N)
    assert serve.top_classes(result) == [] and serve.model_version(result) == "reason_none"


def test_named_reader_goes_first(sample_heads, empty_dir, fake_frozen):
    text = "6 minus 2 is 4 so four shots"
    assert serve.read(None, text, heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"
    assert serve.read(None, text, "frozen", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "frozen"
    assert serve.read(None, text, "none", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "none"


def test_environment_variable_names_the_reader(sample_heads, empty_dir, fake_frozen, monkeypatch):
    text = "6 minus 2 is 4 so four shots"
    monkeypatch.setenv(serve.ENV_READER, "frozen")
    assert serve.read(None, text, heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "frozen"
    assert serve.read(None, text, "tfidf", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"
    monkeypatch.setenv(serve.ENV_READER, "NONE")
    assert serve.read(None, text, heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "none"
    monkeypatch.setenv(serve.ENV_READER, "something else")
    assert serve.read(None, text, heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"


def test_a_named_reader_that_cannot_load_is_skipped(sample_heads, empty_dir, no_frozen_model):
    text = "6 minus 2 is 4 so four shots"
    assert serve.read(None, text, "biencoder", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"
    assert serve.read(None, text, "frozen", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"


def test_a_broken_head_file_is_skipped(tmp_path, empty_dir):
    (tmp_path / "tfidf.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "tfidf.npz").write_bytes(b"not an npz")
    assert serve.load_reader("tfidf", tmp_path, empty_dir) is None
    assert serve.read(None, "x y z", heads_dir=tmp_path, artifacts_dir=empty_dir)["reader"] == "none"


def test_a_head_for_other_labels_is_refused(sample_heads, tmp_path, empty_dir):
    meta = json.loads((sample_heads / "tfidf.json").read_text(encoding="utf-8"))
    meta["labels"] = meta["labels"][:-1]
    (tmp_path / "tfidf.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "tfidf.npz").write_bytes((sample_heads / "tfidf.npz").read_bytes())
    assert serve.load_reader("tfidf", tmp_path, empty_dir) is None


def test_readers_are_loaded_once(sample_heads, empty_dir):
    first = serve.get_reader(heads_dir=sample_heads, artifacts_dir=empty_dir)
    assert serve.get_reader(heads_dir=sample_heads, artifacts_dir=empty_dir) is first
    serve.reset()
    assert serve.get_reader(heads_dir=sample_heads, artifacts_dir=empty_dir) is not first


# ---------------------------------------------------------------- what the server builds from a result

def test_top_classes_fits_the_reason_response(sample_rows, sample_heads, empty_dir):
    for row in sample_rows[::7]:
        result = serve.read(None, row["text"], heads_dir=sample_heads, artifacts_dir=empty_dir)
        top = serve.top_classes(result)
        assert 1 <= len(top) <= 3
        assert [entry["p"] for entry in top] == sorted((entry["p"] for entry in top), reverse=True)
        assert top[0]["id"] == REASON_LABELS[int(np.argmax(result["probs"]))]
        response = S.ReasonResponse(status=result["status"], top=top, reader=result["reader"], updates=[],
                                    model_version=serve.model_version(result), latency_ms=1.0)
        assert response.top[0].name


def test_top_classes_names_correct_reasoning_and_drops_tiny_runners_up():
    probs = [0.0] * serve.N
    probs[serve.CORRECT_INDEX], probs[0], probs[1] = 0.97, 0.025, 0.005
    top = serve.top_classes({"probs": probs, "reader": "tfidf"})
    assert [entry["id"] for entry in top] == ["CORRECT_REASON", "M01"]
    assert top[0]["name"] == "Correct reasoning" and top[0]["band"] == "Likely"
    assert top[1]["name"] == "Boundary Drift" and top[1]["band"] == "Unsure"
