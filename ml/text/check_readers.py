"""Scores the word-count reader and the fine-tuned reader on the two sentence test sets.

  test      the held-out contexts of ml/data/reasons.jsonl (same generator as training)
  persona   ml/data/reasons_persona.jsonl (a different model and prompt; see gen_persona_reasons.py)

Both sets are LLM-written. A quick check, not the E16 experiment (package E-c writes that).

Run from the repo root:  python -m ml.text.check_readers
"""
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union

from ml.contracts.classes import REASON_LABELS

ROOT = Path(__file__).resolve().parents[2]
INDEX = {label: i for i, label in enumerate(REASON_LABELS)}


def load(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines()]


def scores(probs, truth, threshold=None):
    predicted = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    out = {"n": len(truth), "accuracy": float((predicted == truth).mean()),
           "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
           "top3": float(np.mean([t in row for t, row in zip(truth, top3)]))}
    if threshold is not None:
        accepted = probs.max(1) >= threshold
        out["answers"] = float(accepted.mean())
        out["right_when_it_answers"] = float((predicted[accepted] == truth[accepted]).mean()) if accepted.any() else None
    return out


def tfidf_reader(train_rows):
    model = make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        LogisticRegression(max_iter=2000, C=10.0))
    model.fit([r["text"] for r in train_rows], [INDEX[r["label"]] for r in train_rows])

    def read(texts):
        wide = np.zeros((len(texts), len(REASON_LABELS)))
        wide[:, model.classes_] = model.predict_proba(texts)
        return wide
    return read, None


def biencoder_reader():
    """The fine-tuned reader from ml/artifacts/reason_*/, run with onnxruntime. None if it is not there."""
    found = sorted(glob.glob(str(ROOT / "ml" / "artifacts" / "reason_*" / "model.onnx")))
    if not found:
        return None, None
    import onnxruntime
    from tokenizers import Tokenizer
    folder = Path(found[-1]).parent
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    described = json.loads((folder / "descriptions.json").read_text(encoding="utf-8"))
    assert described["labels"] == REASON_LABELS
    anchors = np.array(described["embeddings"], dtype=np.float32)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=meta["max_len"])
    tokenizer.enable_padding()
    session = onnxruntime.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])

    def read(texts):
        out = []
        for start in range(0, len(texts), 64):
            encoded = tokenizer.encode_batch(texts[start:start + 64])
            vectors = session.run(None, {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                                         "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64)})[0]
            logits = vectors @ anchors.T / meta["temperature"]
            exp = np.exp(logits - logits.max(1, keepdims=True))
            out.append(exp / exp.sum(1, keepdims=True))
        return np.concatenate(out)
    return read, meta["threshold"]


def main():
    rows = load("ml/data/reasons.jsonl")
    persona = load("ml/data/reasons_persona.jsonl")
    sets = {
        "test (same generator)": [r for r in rows if r["split"] == "test"],
        "persona (different model and prompt)": persona,
        "  persona, picked by itself": [r for r in persona if not r["voice"].endswith("_told")],
        "  persona, told which option": [r for r in persona if r["voice"].endswith("_told")],
    }
    readers = {"tfidf": tfidf_reader([r for r in rows if r["split"] == "train"]), "biencoder": biencoder_reader()}
    results = {}
    for name, (read, threshold) in readers.items():
        if read is None:
            print(f"{name}: no artifact found, skipped")
            continue
        for set_name, items in sets.items():
            truth = np.array([INDEX[r["label"]] for r in items])
            results[(name, set_name)] = scores(read([r["text"] for r in items]), truth, threshold)
    print(f"{'reader':10} {'set':40} {'n':>4} {'acc':>6} {'F1':>6} {'top3':>6} {'answers':>8} {'right then':>10}")
    for (name, set_name), s in results.items():
        extra = f"{s['answers']:8.0%} {s['right_when_it_answers']:10.0%}" if "answers" in s else ""
        print(f"{name:10} {set_name:40} {s['n']:4} {s['accuracy']:6.3f} {s['macro_f1']:6.3f} {s['top3']:6.3f} {extra}")
    print("Both sets are LLM-written; neither is real student text.")


if __name__ == "__main__":
    main()
