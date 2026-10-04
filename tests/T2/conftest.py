"""Shared pieces for the T2 tests.

Nothing here needs the big files. A test that needs the fine-tuned artifact or the unchanged
model in the Hugging Face cache asks for `real_artifacts` / `real_frozen` and is skipped when
they are absent, so a fresh clone stays green.
"""
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from ml.text import fit_heads, serve

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "reasons_sample.jsonl"


def fake_embedder(texts, dim=96):
    """A stand-in for the embedding model: hashed character 3-grams, unit length. Deterministic."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        padded = f" {str(text).lower()} "
        for start in range(len(padded) - 2):
            out[row, zlib.crc32(padded[start:start + 3].encode("utf-8")) % dim] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


@pytest.fixture(scope="session")
def sample_rows():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="session")
def sample_heads(tmp_path_factory, sample_rows):
    """Both heads fitted on the 60-row sample; `frozen` uses the fake embedder."""
    folder = tmp_path_factory.mktemp("heads")
    fit_heads.fit_tfidf(sample_rows, folder)
    fit_heads.fit_frozen(sample_rows, folder, embedder=fake_embedder, revision="fake")
    return folder


@pytest.fixture()
def empty_dir(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture(autouse=True)
def fresh_readers(monkeypatch):
    """Every test starts with nothing loaded and no reader named in the environment."""
    monkeypatch.delenv(serve.ENV_READER, raising=False)
    serve.reset()
    yield
    serve.reset()


@pytest.fixture()
def fake_frozen(monkeypatch):
    """Serve `frozen` with the fake embedder instead of the 436 MB model."""
    monkeypatch.setattr(serve, "frozen_embedder", lambda *args, **kwargs: (fake_embedder, "fake"))


@pytest.fixture()
def no_frozen_model(monkeypatch):
    """Behave like a machine whose Hugging Face cache does not hold the model."""
    def missing(*args, **kwargs):
        raise FileNotFoundError("not in the cache")
    monkeypatch.setattr(serve, "frozen_embedder", missing)


@pytest.fixture(scope="session")
def real_artifacts():
    if not glob.glob(str(serve.ARTIFACTS_DIR / "reason_*" / "model.onnx")):
        pytest.skip("the fine-tuned artifact ml/artifacts/reason_*/ is not on this machine")
    return serve.ARTIFACTS_DIR


@pytest.fixture(scope="session")
def real_frozen():
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(serve.FROZEN_MODEL, serve.FROZEN_ONNX_FILE)
    except Exception:
        found = None
    if not isinstance(found, str):
        pytest.skip("BAAI/bge-base-en-v1.5 onnx/model.onnx is not in the Hugging Face cache")
    return found


def check_shape(result):
    """What every `read` result must look like."""
    assert set(result) >= {"probs", "status", "reader"}
    assert len(result["probs"]) == serve.N
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in result["probs"])
    assert abs(sum(result["probs"]) - 1.0) < 1e-6
    assert result["status"] in ("matched", "correct_reasoning", "unsure")
    assert result["reader"] in ("tfidf", "biencoder", "frozen", "none")
    return result
