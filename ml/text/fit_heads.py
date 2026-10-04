"""Fits the two small heads of the sentence reader and saves them as plain arrays (package T2).

    python -m ml.text.fit_heads                    # tfidf and frozen, from ml/data/reasons.jsonl
    python -m ml.text.fit_heads --readers tfidf    # one of them
    python -m ml.text.fit_heads --data tests/fixtures/reasons_sample.jsonl --out <folder>

Each head is fitted on the train split; its "unsure" threshold is chosen on the val split so
that accepted answers are at least 90% right there (05 §6). The test split is not read.

Output per head, in ml/text/heads/ (committed; no pickles, so nothing breaks when
scikit-learn changes):
    <name>.json   labels, threshold, where it came from, validation scores, version
    <name>.npz    the arrays ml/text/serve.py rebuilds the predictor from

`frozen` needs the unchanged BAAI/bge-base-en-v1.5 as an ONNX file (about 436 MB). This is
the only command that downloads it; it lands in the normal Hugging Face cache, outside the
repo. The server never downloads: without the file in the cache it skips `frozen`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ml.contracts.classes import REASON_LABELS
from ml.text import serve
from ml.text.check_readers import scores

ROOT = serve.ROOT
DATA = ROOT / "ml" / "data" / "reasons.jsonl"
INDEX = {label: i for i, label in enumerate(REASON_LABELS)}

# The word-count pipeline of ml/text/check_readers.py and kaggle_train.py, block by block.
TFIDF_BLOCKS = [{"key": "word", "analyzer": "word", "ngram_range": [1, 2]},
                {"key": "char", "analyzer": "char_wb", "ngram_range": [3, 5]}]
LOGREG = {"C": 10.0, "max_iter": 2000}
# Used only when there are too few validation sentences to choose a threshold (the 60-row
# sample has none). It is TEXT_MATCH_P of ml/contracts/params.py, not a measured value.
DEFAULT_THRESHOLD = 0.6


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def data_sha256(path):
    """SHA-256 of the file with Windows line endings folded, so every checkout gives the same value."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def split(rows, name):
    chosen = [r for r in rows if r["split"] == name]
    return [r["text"] for r in chosen], np.array([INDEX[r["label"]] for r in chosen], dtype=np.int64)


def _softmax_parts(model):
    """(coef, intercept) with one row per fitted class, so softmax(x @ coef.T + intercept) is predict_proba."""
    coef, intercept = model.coef_, model.intercept_
    if coef.shape[0] == 1 and len(model.classes_) == 2:     # two classes: sklearn keeps one row
        coef, intercept = np.vstack([np.zeros_like(coef), coef]), np.array([0.0, intercept[0]])
    return coef, intercept


def _version(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()[:8]


def _threshold(reader_probs, texts, truth):
    """(threshold, how it was chosen, validation scores)."""
    if len(texts) < serve.MIN_ACCEPTED:
        return DEFAULT_THRESHOLD, "default: too few validation sentences to choose one", None
    probs = reader_probs(texts)
    threshold = serve.pick_threshold(probs, truth)
    rule = (f"smallest confidence with accepted answers >= {serve.TARGET_ACCURACY:.0%} right on val"
            if threshold < serve.NEVER else
            f"no confidence level reaches {serve.TARGET_ACCURACY:.0%} on val: always unsure")
    return threshold, rule, scores(probs, truth, threshold)


def _save(out_dir, name, meta, arrays, reader_factory, val, source):
    """Write the arrays, reload them the way the server does, choose the threshold with that predictor."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"reader": name, "labels": REASON_LABELS, "threshold": serve.NEVER, **meta,
            "version": _version(arrays), **source}
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    threshold, rule, val_scores = _threshold(reader_factory().probs, *val)
    meta.update({"threshold": threshold, "threshold_rule": rule, "val": val_scores})
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["bytes"] = {f.name: f.stat().st_size for f in (path, out_dir / f"{name}.npz")}
    return meta


def fit_tfidf(rows, out_dir, source=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    texts, truth = split(rows, "train")
    started = time.perf_counter()
    model = make_pipeline(
        make_union(*[TfidfVectorizer(analyzer=b["analyzer"], ngram_range=tuple(b["ngram_range"]), sublinear_tf=True)
                     for b in TFIDF_BLOCKS]),
        LogisticRegression(**LOGREG))
    model.fit(texts, truth)
    seconds = time.perf_counter() - started
    union, head = model.steps[0][1], model.steps[1][1]
    coef, intercept = _softmax_parts(head)
    arrays = {"classes": head.classes_.astype(np.int64), "intercept": intercept.astype(np.float32)}
    start = 0
    for block, (_, vectorizer) in zip(TFIDF_BLOCKS, union.transformer_list):
        terms = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
        arrays[f"vocab_{block['key']}"] = np.frombuffer(json.dumps(terms, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        arrays[f"idf_{block['key']}"] = vectorizer.idf_.astype(np.float32)
        arrays[f"coef_{block['key']}"] = coef[:, start:start + len(terms)].T.astype(np.float32)
        start += len(terms)
    meta = {"blocks": TFIDF_BLOCKS, "logreg": LOGREG, "n_features": int(start),
            "sentences": {"train": len(texts)}, "fit_seconds": round(seconds, 2)}
    saved = _save(out_dir, "tfidf", meta, arrays, lambda: serve.TfidfReader(out_dir), split(rows, "val"), source or {})
    # The stored arrays must give the fitted pipeline's own probabilities.
    check = texts[:200]
    difference = float(np.abs(serve.TfidfReader(out_dir).probs(check)[:, head.classes_] - model.predict_proba(check)).max())
    if difference > 1e-4:
        raise RuntimeError(f"rebuilt word-count reader differs from the fitted pipeline by {difference:.6f}")
    saved["max_difference_from_pipeline"] = difference
    return saved


def fit_frozen(rows, out_dir, source=None, embedder=None, revision=None, download=True):
    """`embedder` is for tests; by default the real model is used (downloaded if `download`)."""
    from sklearn.linear_model import LogisticRegression
    if embedder is None:
        embedder, revision = serve.frozen_embedder(download=download)
    texts, truth = split(rows, "train")
    started = time.perf_counter()
    vectors = embedder(texts)
    embed_seconds = time.perf_counter() - started
    started = time.perf_counter()
    head = LogisticRegression(**LOGREG).fit(vectors, truth)
    seconds = time.perf_counter() - started
    coef, intercept = _softmax_parts(head)
    arrays = {"classes": head.classes_.astype(np.int64), "coef": coef.T.astype(np.float32),
              "intercept": intercept.astype(np.float32)}
    meta = {"model": serve.FROZEN_MODEL, "revision": revision, "file": serve.FROZEN_ONNX_FILE,
            "max_len": serve.FROZEN_MAX_LEN, "pooling": "cls", "normalize": True, "logreg": LOGREG,
            "sentences": {"train": len(texts)}, "embed_seconds": round(embed_seconds, 1),
            "fit_seconds": round(seconds, 2)}
    return _save(out_dir, "frozen", meta, arrays, lambda: serve.FrozenReader(out_dir, embedder=embedder),
                 split(rows, "val"), source or {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=serve.HEADS_DIR)
    parser.add_argument("--readers", nargs="+", default=["tfidf", "frozen"], choices=["tfidf", "frozen"])
    parser.add_argument("--no-download", action="store_true",
                        help="frozen: use the model only if it is already in the Hugging Face cache")
    args = parser.parse_args(argv)
    rows = load_rows(args.data)
    try:
        shown = args.data.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = args.data.name
    source = {"data": shown, "data_sha256": data_sha256(args.data)}
    status = 0
    for name in args.readers:
        try:
            meta = (fit_tfidf(rows, args.out, source) if name == "tfidf"
                    else fit_frozen(rows, args.out, source, download=not args.no_download))
        except Exception as error:
            print(f"{name}: not fitted ({type(error).__name__}: {error})")
            status = 1
            continue
        val = meta["val"]
        line = (f"val n {val['n']}  acc {val['accuracy']:.3f}  F1 {val['macro_f1']:.3f}  top3 {val['top3']:.3f}  "
                f"answers {val['answers']:.0%}  right then "
                f"{'-' if val['right_when_it_answers'] is None else format(val['right_when_it_answers'], '.0%')}"
                if val else "no validation sentences")
        print(f"{name}: threshold {meta['threshold']:.3f}  {line}  version {meta['version']}  "
              f"files {meta['bytes']}  fit {meta['fit_seconds']} s"
              + (f"  embed {meta['embed_seconds']} s" if "embed_seconds" in meta else ""))
    serve.reset()
    return status


if __name__ == "__main__":
    sys.exit(main())
