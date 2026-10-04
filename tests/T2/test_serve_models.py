"""The two readers that need a large model file: `biencoder` (fine-tuned) and `frozen`.

Each is tested twice: with a small fake embedder (always runs) and with the real file
(skipped when the file is not on this machine).
"""
import json

import numpy as np
import pytest

from ml.contracts.classes import REASON_LABELS
from ml.text import fit_heads, serve
from tests.T2.conftest import check_shape, fake_embedder

INDEX = {label: i for i, label in enumerate(REASON_LABELS)}
TEXT = "6 minus 2 is 4 so four shots"


# ---------------------------------------------------------------- frozen, fake embedder

def test_frozen_reads_the_sample_with_a_fake_embedder(sample_rows, sample_heads, empty_dir, fake_frozen):
    right = 0
    for row in sample_rows:
        result = check_shape(serve.read(row["code"], row["text"], "frozen", heads_dir=sample_heads, artifacts_dir=empty_dir))
        assert result["reader"] == "frozen"
        right += int(np.argmax(result["probs"])) == INDEX[row["label"]]
    assert right >= 0.7 * len(sample_rows)


def test_frozen_head_matches_the_fitted_logistic_regression(sample_rows, tmp_path):
    from sklearn.linear_model import LogisticRegression
    fit_heads.fit_frozen(sample_rows, tmp_path, embedder=fake_embedder, revision="fake")
    texts, truth = fit_heads.split(sample_rows, "train")
    reference = LogisticRegression(**fit_heads.LOGREG).fit(fake_embedder(texts), truth)
    ours = serve.FrozenReader(tmp_path, embedder=fake_embedder).probs(texts)
    assert np.abs(ours[:, reference.classes_] - reference.predict_proba(fake_embedder(texts))).max() < 1e-4
    arrays = np.load(tmp_path / "frozen.npz", allow_pickle=False)
    assert arrays["coef"].shape == (96, len(reference.classes_))


def test_frozen_is_used_when_it_is_the_only_reader_that_loads(sample_heads, tmp_path, empty_dir, fake_frozen):
    for name in ("frozen.json", "frozen.npz"):
        (tmp_path / name).write_bytes((sample_heads / name).read_bytes())
    assert check_shape(serve.read(None, TEXT, heads_dir=tmp_path, artifacts_dir=empty_dir))["reader"] == "frozen"


def test_frozen_is_skipped_when_the_model_is_not_in_the_cache(sample_heads, tmp_path, empty_dir, no_frozen_model):
    for name in ("frozen.json", "frozen.npz"):
        (tmp_path / name).write_bytes((sample_heads / name).read_bytes())
    assert serve.load_reader("frozen", tmp_path, empty_dir) is None
    assert serve.read(None, TEXT, heads_dir=tmp_path, artifacts_dir=empty_dir)["reader"] == "none"


def test_serving_never_downloads_the_model(sample_heads, empty_dir, monkeypatch):
    """Loading `frozen` asks the Hugging Face cache only; no network call is allowed."""
    import huggingface_hub
    calls = []

    def recorded(repo, filename, **kwargs):
        calls.append(kwargs.get("local_files_only"))
        raise FileNotFoundError("not in the cache")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", recorded)
    assert serve.load_reader("frozen", sample_heads, empty_dir) is None
    assert calls == [True]


def test_binary_head_is_stored_with_one_row_per_class(sample_rows, tmp_path):
    two = [row for row in sample_rows if row["label"] in ("M01", "M08")]
    fit_heads.fit_frozen(two, tmp_path, embedder=fake_embedder)
    reader = serve.FrozenReader(tmp_path, embedder=fake_embedder)
    probs = reader.probs([row["text"] for row in two])
    assert set(np.flatnonzero(probs.sum(0))) == {INDEX["M01"], INDEX["M08"]}
    assert np.allclose(probs.sum(1), 1.0)


# ---------------------------------------------------------------- biencoder, fake artifact

@pytest.fixture()
def fake_artifact(tmp_path, monkeypatch):
    """An artifact folder with the real layout and a fake model behind it."""
    root = tmp_path / "artifacts"
    folder = root / "reason_fake0001"
    folder.mkdir(parents=True)
    texts = [f"description of {label}" for label in REASON_LABELS]
    (folder / "model.onnx").write_bytes(b"")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"id": "fake0001", "max_len": 64, "temperature": 0.05,
                                                  "threshold": 0.4, "pooling": "cls", "normalize": True}), encoding="utf-8")
    (folder / "descriptions.json").write_text(json.dumps({
        "labels": REASON_LABELS, "texts": texts, "embeddings": fake_embedder(texts).tolist()}), encoding="utf-8")

    class FakeOnnx:
        def __init__(self, model_path, tokenizer_path, max_len):
            assert str(model_path).endswith("model.onnx") and max_len == 64

        def __call__(self, texts, batch_size=64):
            return fake_embedder(texts)
    monkeypatch.setattr(serve, "OnnxEmbedder", FakeOnnx)
    return root


def test_biencoder_is_used_when_the_artifact_is_present(fake_artifact, empty_dir):
    result = check_shape(serve.read(None, "description of D03", heads_dir=empty_dir, artifacts_dir=fake_artifact))
    assert result["reader"] == "biencoder" and result["version"] == "fake0001"
    assert REASON_LABELS[int(np.argmax(result["probs"]))] == "D03"
    assert result["status"] == "matched" and result["threshold"] == 0.4


def test_biencoder_is_skipped_when_the_artifact_folder_is_empty(sample_heads, empty_dir, fake_frozen):
    assert serve.load_reader("biencoder", sample_heads, empty_dir) is None
    assert serve.read(None, TEXT, "biencoder", heads_dir=sample_heads, artifacts_dir=empty_dir)["reader"] == "tfidf"
    assert serve.read(None, TEXT, heads_dir=empty_dir, artifacts_dir=empty_dir)["reader"] == "none"


def test_same_call_with_and_without_the_artifact(fake_artifact, sample_heads, tmp_path, empty_dir, fake_frozen):
    """Heads folder holding only `frozen`: biencoder comes before it when present, frozen is used when not."""
    for name in ("frozen.json", "frozen.npz"):
        (tmp_path / name).write_bytes((sample_heads / name).read_bytes())
    assert serve.read(None, TEXT, heads_dir=tmp_path, artifacts_dir=fake_artifact)["reader"] == "biencoder"
    assert serve.read(None, TEXT, heads_dir=tmp_path, artifacts_dir=empty_dir)["reader"] == "frozen"
    assert serve.read(None, TEXT, heads_dir=sample_heads, artifacts_dir=fake_artifact)["reader"] == "tfidf"
    assert serve.read(None, TEXT, "biencoder", heads_dir=sample_heads, artifacts_dir=fake_artifact)["reader"] == "biencoder"


def test_an_artifact_for_other_labels_is_skipped(fake_artifact, empty_dir):
    path = fake_artifact / "reason_fake0001" / "descriptions.json"
    described = json.loads(path.read_text(encoding="utf-8"))
    described["labels"] = described["labels"][::-1]
    path.write_text(json.dumps(described), encoding="utf-8")
    assert serve.load_reader("biencoder", empty_dir, fake_artifact) is None


def test_a_broken_model_file_is_skipped(tmp_path, empty_dir):
    """A real loader on a file that is not a model: skipped, never raised."""
    folder = tmp_path / "reason_broken"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"not a model")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"max_len": 64, "temperature": 0.05, "threshold": 0.5}), encoding="utf-8")
    (folder / "descriptions.json").write_text(json.dumps({"labels": REASON_LABELS, "texts": [], "embeddings": []}), encoding="utf-8")
    assert serve.read(None, TEXT, heads_dir=empty_dir, artifacts_dir=tmp_path)["reader"] == "none"


# ---------------------------------------------------------------- the real files (skipped on a fresh clone)

def test_real_biencoder_is_used_when_present_and_skipped_when_not(real_artifacts, empty_dir):
    present = check_shape(serve.read(None, TEXT, heads_dir=empty_dir, artifacts_dir=real_artifacts))
    assert present["reader"] == "biencoder"
    assert present["threshold"] == pytest.approx(json.loads(
        (serve.load_reader("biencoder", empty_dir, real_artifacts).folder / "meta.json").read_text(encoding="utf-8"))["threshold"])
    assert serve.read(None, TEXT, "biencoder")["reader"] == "biencoder"
    assert serve.read(None, TEXT)["reader"] == "tfidf"                       # the default order still starts with tfidf
    serve.reset()
    assert serve.read(None, TEXT, heads_dir=empty_dir, artifacts_dir=empty_dir)["reader"] == "none"
    assert serve.read(None, TEXT, "biencoder", artifacts_dir=empty_dir)["reader"] == "tfidf"


def test_real_biencoder_handles_odd_text(real_artifacts):
    for text in ["", "why " * 5000, "लूप अपने आप रुक जाता है", "loop apne aap ruk jayega", "🙂", None]:
        assert check_shape(serve.read(None, text, "biencoder"))["reader"] == "biencoder"


def test_real_frozen_reads_the_sample(real_frozen, sample_rows, tmp_path, empty_dir):
    """Fit on the 48 sample training sentences with the real model, then read all 60."""
    fit_heads.fit_frozen(sample_rows, tmp_path, download=False)
    meta = json.loads((tmp_path / "frozen.json").read_text(encoding="utf-8"))
    assert meta["model"] == serve.FROZEN_MODEL and meta["revision"]
    right = 0
    for row in sample_rows:
        result = check_shape(serve.read(row["code"], row["text"], heads_dir=tmp_path, artifacts_dir=empty_dir))
        assert result["reader"] == "frozen"
        right += int(np.argmax(result["probs"])) == INDEX[row["label"]]
    assert right >= 0.8 * len(sample_rows)


def test_real_frozen_with_the_committed_head(real_frozen):
    for text in [TEXT, "", "why " * 5000, "लूप अपने आप रुक जाता है", "loop apne aap ruk jayega", None]:
        result = check_shape(serve.read(None, text, "frozen"))
        assert result["reader"] == "frozen" and 0.0 < result["threshold"] < 1.0


def test_real_frozen_vectors_are_unit_length_cls_vectors(real_frozen):
    embed, revision = serve.frozen_embedder()
    vectors = embed(["i thought arrays start from 1", "loop apne aap ruk jayega", "ok"])
    assert vectors.shape == (3, 768) and revision
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    alone = embed(["ok"])                       # padding inside a batch must not change a sentence's vector
    assert np.abs(alone[0] - vectors[2]).max() < 1e-4
