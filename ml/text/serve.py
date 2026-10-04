"""Sentence reader, serving side (ml_plan/05 §6), package T2.

    from ml.text.serve import read, top_classes
    result = read(code, text)
    # {"probs":  one probability per label, in REASON_LABELS order,
    #  "status": "matched" | "correct_reasoning" | "unsure",
    #  "reader": "tfidf" | "biencoder" | "frozen" | "none",
    #  "threshold", "version", "masked"}          (extras; see below)
    top_classes(result)   # [{id, name, subtitle, p, band}] for the /reason response

The sentence alone is what a reader sees. `code` is optional and is used only to remove the
classes the code makes impossible, with the diagnoser's own masking table
(MASK_PRECONDITIONS through ml/model/mask.py and ml/features/ast_feats.py). Masking is applied
only to whole, self-contained code (see `allowed_labels`); `read(..., keep=[...])` names
labels that must never be removed.

Three readers, one interface (`probs(texts) -> (n, 18)`, `threshold`, `version`):

  tfidf      word 1-2-grams + character 3-5-grams, logistic regression. Rebuilt from plain
             arrays in ml/text/heads/tfidf.{json,npz}; nothing is fitted at load.
  biencoder  the fine-tuned model in ml/artifacts/reason_<id>/ (onnxruntime + tokenizers).
  frozen     the unchanged BAAI/bge-base-en-v1.5 (its ONNX file, read from the Hugging Face
             cache, never downloaded here) + the logistic head in ml/text/heads/frozen.{json,npz}.

Load order: tfidf, then biencoder, then frozen, then none. This is the project lead's
decision and differs from the order written in 05 §6: on held-out contexts tfidf and the
fine-tuned reader are level (0.72), frozen is behind (0.60), and the fine-tuned reader's
threshold did not hold on the second test set (notes/T1.md, notes/T2.md).
A reader named by the `reader` argument or by the RELEARN_REASON_READER environment variable
is tried first. A reader that cannot load (missing file, missing package) is skipped silently.

`read` never raises. Odd input (None, empty, very long, any script) gives a normal result;
an empty sentence is `unsure`.

    python -m ml.text.serve "i thought arrays start from 1"
    python -m ml.text.serve --bench          # load time and per-sentence time of each reader
"""
from __future__ import annotations

import functools
import glob
import json
import os
import threading
from pathlib import Path

import numpy as np

from ml.contracts.classes import CLASS_INFO, REASON_LABELS, band
from ml.contracts.feature_names import GROUP_A
from ml.contracts.params import PROB_FLOOR

ROOT = Path(__file__).resolve().parents[2]
HEADS_DIR = ROOT / "ml" / "text" / "heads"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

READER_ORDER = ("tfidf", "biencoder", "frozen")
ENV_READER = "RELEARN_REASON_READER"
CORRECT_LABEL = "CORRECT_REASON"

FROZEN_MODEL = "BAAI/bge-base-en-v1.5"
FROZEN_ONNX_FILE = "onnx/model.onnx"
FROZEN_TOKENIZER_FILE = "tokenizer.json"
FROZEN_MAX_LEN = 64                 # same as the fine-tuned artifact's meta.json

TARGET_ACCURACY = 0.90              # accepted answers must be this often right on validation (05 §6)
MIN_ACCEPTED = 10
NEVER = 1.01                        # threshold meaning "always unsure"
MAX_CHARS = 2000                    # a "why" box holds one sentence; anything longer is cut

N = len(REASON_LABELS)
CORRECT_INDEX = REASON_LABELS.index(CORRECT_LABEL)
# The display text of CORRECT_REASON is not in CLASS_INFO (it is not a diagnoser class).
CORRECT_DISPLAY = {"name": "Correct reasoning", "subtitle": None}


# ---------------------------------------------------------------- small shared pieces

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def pick_threshold(probs, truth, target=TARGET_ACCURACY, min_accepted=MIN_ACCEPTED):
    """Smallest confidence at which accepted answers are at least `target` right.

    The same rule as ml/text/kaggle_train.py, so all three readers' thresholds mean the same.
    Returns NEVER (always unsure) when no confidence level reaches the target.
    """
    probs, truth = np.asarray(probs), np.asarray(truth)
    confidence, right = probs.max(1), probs.argmax(1) == truth
    for threshold in np.unique(np.round(confidence, 3)):
        accepted = confidence >= threshold
        if accepted.sum() >= min_accepted and right[accepted].mean() >= target:
            return float(threshold)
    return NEVER


def status_of(row, threshold):
    """`unsure` below the threshold; otherwise what the top label says."""
    top = int(np.argmax(row))
    if not np.isfinite(row).all() or row[top] < threshold:
        return "unsure"
    return "correct_reasoning" if top == CORRECT_INDEX else "matched"


def _widen(narrow, classes):
    """Probabilities over the labels a head was fitted on -> all 18 labels (missing ones get 0)."""
    wide = np.zeros((len(narrow), N))
    wide[:, classes] = narrow
    return wide


def _load_head(folder, name):
    """(meta, arrays) of one committed head. Raises when a file is missing or the labels moved."""
    folder = Path(folder)
    meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    if meta["labels"] != REASON_LABELS:
        raise ValueError(f"{name} head was fitted for other labels")
    with np.load(folder / f"{name}.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return meta, arrays


# ---------------------------------------------------------------- text -> vectors with ONNX

class OnnxEmbedder:
    """Sentences -> vectors with onnxruntime + tokenizers (no torch).

    A model that returns one vector per sentence (the fine-tuned export) is used as it is.
    A model that returns one vector per token (the stock bge export) is pooled the way bge is
    trained: the [CLS] token, L2-normalised.
    """

    def __init__(self, model_path, tokenizer_path, max_len):
        import onnxruntime
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=int(max_len))
        self.tokenizer.enable_padding()
        self.session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.inputs = {entry.name for entry in self.session.get_inputs()}

    def __call__(self, texts, batch_size=64):
        out = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(list(texts[start:start + batch_size]))
            feed = {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            vectors = self.session.run(None, feed)[0]
            if vectors.ndim == 3:
                vectors = vectors[:, 0]
                vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            out.append(vectors)
        return np.concatenate(out)


def frozen_embedder(model=FROZEN_MODEL, max_len=FROZEN_MAX_LEN, revision=None, download=False):
    """The unchanged embedding model from the Hugging Face cache.

    download=False (serving): raises if the files are not already in the cache; no network.
    download=True (only `python -m ml.text.fit_heads`): fetches them into the normal cache.
    Returns (embedder, revision).
    """
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(model, name, revision=revision, local_files_only=not download)
             for name in (FROZEN_ONNX_FILE, FROZEN_TOKENIZER_FILE)]
    parts = Path(paths[0]).parts
    found = parts[parts.index("snapshots") + 1] if "snapshots" in parts else revision
    return OnnxEmbedder(paths[0], paths[1], max_len), found


# ---------------------------------------------------------------- the three readers

class TfidfReader:
    """Word and character counts + logistic regression, rebuilt from stored arrays.

    The counting is scikit-learn's CountVectorizer with a fixed vocabulary; the tf-idf weighting
    (1 + log tf, times idf, L2 per block) and the logistic step are done here, so the head does
    not depend on how scikit-learn stores a fitted model.
    """
    name = "tfidf"

    def __init__(self, heads_dir=HEADS_DIR):
        from sklearn.feature_extraction.text import CountVectorizer
        meta, arrays = _load_head(heads_dir, "tfidf")
        self.meta, self.threshold, self.version = meta, float(meta["threshold"]), meta["version"]
        self.classes = arrays["classes"]
        self.intercept = arrays["intercept"].astype(np.float64)
        self.blocks = []
        for block in meta["blocks"]:
            key = block["key"]
            vocabulary = json.loads(arrays[f"vocab_{key}"].tobytes().decode("utf-8"))
            counter = CountVectorizer(vocabulary={term: i for i, term in enumerate(vocabulary)},
                                      analyzer=block["analyzer"], ngram_range=tuple(block["ngram_range"]))
            self.blocks.append((counter, arrays[f"idf_{key}"].astype(np.float64),
                                arrays[f"coef_{key}"].astype(np.float64)))

    def probs(self, texts):
        from sklearn.preprocessing import normalize
        logits = np.tile(self.intercept, (len(texts), 1))
        for counter, idf, coef in self.blocks:
            counts = counter.transform(texts).astype(np.float64)
            counts.data = 1.0 + np.log(counts.data)
            weighted = normalize(counts.multiply(idf).tocsr(), norm="l2", copy=False)
            logits += weighted @ coef
        return _widen(softmax(logits), self.classes)


class BiencoderReader:
    """The fine-tuned reader: a sentence lands next to the description of its mistake."""
    name = "biencoder"

    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        found = sorted(glob.glob(str(Path(artifacts_dir) / "reason_*" / "model.onnx")))
        if not found:
            raise FileNotFoundError("no reason_*/model.onnx")
        folder = Path(found[-1]).parent
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        described = json.loads((folder / "descriptions.json").read_text(encoding="utf-8"))
        if described["labels"] != REASON_LABELS:
            raise ValueError("the artifact was trained for other labels")
        self.meta, self.folder = meta, folder
        self.descriptions = described["texts"]
        self.anchors = np.array(described["embeddings"], dtype=np.float32)
        self.temperature, self.threshold = float(meta["temperature"]), float(meta["threshold"])
        self.version = str(meta.get("id") or folder.name.replace("reason_", ""))
        self.embed = OnnxEmbedder(folder / "model.onnx", folder / "tokenizer.json", meta["max_len"])

    def probs(self, texts):
        return softmax(self.embed(texts) @ self.anchors.T / self.temperature)


class FrozenReader:
    """The unchanged embedding model + a logistic head on its vectors."""
    name = "frozen"

    def __init__(self, heads_dir=HEADS_DIR, embedder=None):
        meta, arrays = _load_head(heads_dir, "frozen")
        self.meta, self.threshold, self.version = meta, float(meta["threshold"]), meta["version"]
        self.classes = arrays["classes"]
        self.coef = arrays["coef"].astype(np.float64)
        self.intercept = arrays["intercept"].astype(np.float64)
        self.embed = embedder or frozen_embedder(meta["model"], meta["max_len"], meta.get("revision"))[0]

    def probs(self, texts):
        return _widen(softmax(self.embed(texts) @ self.coef + self.intercept), self.classes)


# ---------------------------------------------------------------- choosing a reader

_cache: dict = {}
_lock = threading.RLock()


def _build(name, heads_dir, artifacts_dir):
    if name == "tfidf":
        return TfidfReader(heads_dir)
    if name == "biencoder":
        return BiencoderReader(artifacts_dir)
    if name == "frozen":
        return FrozenReader(heads_dir)
    raise KeyError(name)


def load_reader(name, heads_dir=None, artifacts_dir=None):
    """The named reader, or None when it cannot load. Loaded once per folder; failures are remembered."""
    heads_dir, artifacts_dir = Path(heads_dir or HEADS_DIR), Path(artifacts_dir or ARTIFACTS_DIR)
    key = (name, str(heads_dir), str(artifacts_dir))
    with _lock:
        if key not in _cache:
            try:
                _cache[key] = _build(name, heads_dir, artifacts_dir)
            except Exception:           # missing file, missing package, bad file: skip this reader
                _cache[key] = None
        return _cache[key]


def reader_order(prefer=None):
    """Names to try. `prefer` (argument, else the environment variable) goes first; "none" tries nothing."""
    prefer = (prefer or os.environ.get(ENV_READER) or "").strip().lower()
    if prefer == "none":
        return ()
    if prefer in READER_ORDER:
        return (prefer,) + tuple(name for name in READER_ORDER if name != prefer)
    return READER_ORDER


def get_reader(prefer=None, heads_dir=None, artifacts_dir=None):
    """The first reader of the order that loads, or None."""
    for name in reader_order(prefer):
        reader = load_reader(name, heads_dir, artifacts_dir)
        if reader is not None:
            return reader
    return None


def reset():
    """Forget loaded readers (after new heads were fitted or an artifact was copied in)."""
    with _lock:
        _cache.clear()
    allowed_labels.cache_clear()


# ---------------------------------------------------------------- masking by the code

# Calls that cannot hide a loop, a swap or a recursion of the learner's own.
LIBRARY_CALLS = frozenset({"printf", "scanf", "puts", "putchar", "getchar", "strlen", "strcmp", "strcpy", "abs"})


def self_contained(code):
    """True when the code parses and every function it calls is defined in it (library calls aside).

    A snippet that calls a helper it does not show (`swap(a, j, j + 1)`) is not proof that a
    structure is absent: the array writes are inside the helper.
    """
    from pycparser import c_ast
    from ml.features.ast_feats import parse
    tree = parse(code)
    if tree is None:
        return False
    defined = {node.decl.name for node in tree.ext if isinstance(node, c_ast.FuncDef)}
    called = set()

    class Calls(c_ast.NodeVisitor):
        def visit_FuncCall(self, node):
            called.add(node.name.name if isinstance(node.name, c_ast.ID) else None)
            self.generic_visit(node)
    Calls().visit(tree)
    return called <= defined | LIBRARY_CALLS


@functools.lru_cache(maxsize=256)
def allowed_labels(code):
    """Per label: False when the code makes the class impossible. None = no masking.

    Masking needs whole code, as the diagnoser gets it. So None (no masking) for:
      * no code, or code that does not parse as a C file with at least one function. A bare
        fragment (a few statements, as most quiz snippets are) is not masked, on purpose:
        wrapping fragments in an empty function was tried and removed the true class on 10 of
        the 85 train and val contexts (a swap of two plain variables has no array write;
        `low = mid;` alone has no mid computation);
      * code that calls a function it does not define (see `self_contained`).
    CORRECT_REASON has no precondition, so it is never removed. See notes/T2.md.
    """
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        from ml.features.ast_feats import ast_features
        from ml.model.mask import allowed_classes
        feats, meta = ast_features(code)
        if not meta.get("parse_ok") or not self_contained(code):
            return None
        row = [feats[name] for name in GROUP_A]
        return tuple(bool(ok) for ok in allowed_classes(row, features=GROUP_A, labels=REASON_LABELS)[0])
    except Exception:
        return None


def mask_probs(probs, allowed):
    """Zero the impossible classes and renormalise (ml/model/mask.py). `allowed` None = unchanged."""
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    if allowed is None:
        return probs
    from ml.model.mask import apply_mask
    return apply_mask(probs, np.atleast_2d(np.asarray(allowed, dtype=bool)))


# ---------------------------------------------------------------- the one call

def _result(probs, status, reader, threshold=None, version=None, masked=()):
    return {"probs": [float(p) for p in probs], "status": status, "reader": reader,
            "threshold": threshold, "version": version, "masked": list(masked)}


def _nothing(reader="none", threshold=None, version=None):
    return _result(np.full(N, 1.0 / N), "unsure", reader, threshold, version)


def clean(text):
    """What a reader is given: a string, cut to MAX_CHARS, outer blanks removed."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text[:MAX_CHARS].strip()


def read(code, text, reader=None, heads_dir=None, artifacts_dir=None, keep=()):
    """Which mistake does this one sentence show?

    code    the code the sentence is about, or None. Only used to remove impossible classes.
    text    the learner's sentence.
    reader  optional: "tfidf", "biencoder", "frozen" (tried first) or "none".
    keep    labels that must never be removed, e.g. a quiz item's own `classes`.
    """
    try:
        chosen = get_reader(reader, heads_dir, artifacts_dir)
        if chosen is None:
            return _nothing()
        sentence = clean(text)
        if not sentence:
            return _nothing(chosen.name, chosen.threshold, chosen.version)
        raw = chosen.probs([sentence])
        allowed = allowed_labels(code) if isinstance(code, str) else None
        if allowed and isinstance(keep, (list, tuple, set, frozenset)) and keep:
            allowed = tuple(ok or label in keep for label, ok in zip(REASON_LABELS, allowed))
        row = mask_probs(raw, allowed)[0]
        if row.shape != (N,) or not np.isfinite(row).all():
            return _nothing(chosen.name, chosen.threshold, chosen.version)
        masked = [label for label, ok in zip(REASON_LABELS, allowed) if not ok] if allowed else []
        return _result(row, status_of(row, chosen.threshold), chosen.name, chosen.threshold,
                       chosen.version, masked)
    except Exception:
        return _nothing()


def top_classes(result, k=3, floor=PROB_FLOOR):
    """The `top` list of the /reason response (schemas.TopClass): best first, at most k,
    runners-up below `floor` dropped. Empty when no reader is loaded."""
    if result.get("reader") == "none":
        return []
    probs = np.asarray(result["probs"], dtype=np.float64)
    out = []
    for index in np.argsort(-probs, kind="stable")[:k]:
        p = float(probs[index])
        if out and p < floor:
            break
        label = REASON_LABELS[index]
        info = CORRECT_DISPLAY if label == CORRECT_LABEL else CLASS_INFO[label]
        out.append({"id": label, "name": info["name"], "subtitle": info["subtitle"],
                    "p": round(p, 4), "band": band(p)})
    return out


def model_version(result):
    """The `model_version` string for the response envelope, e.g. reason_tfidf_1a2b3c4d."""
    if result.get("reader") == "none" or not result.get("version"):
        return "reason_none"
    return f"reason_{result['reader']}_{result['version']}"


# ---------------------------------------------------------------- command line

def bench(repeats=50):
    """Load time and per-sentence time of each reader on this machine."""
    import time
    sentences = ["i thought arrays start from 1", "loop apne aap ruk jayega",
                 "6 minus 2 is 4 so four shots", "the recursive call adds it up on its own i think",
                 "because = checks if they are equal"]
    code = "int sum(int a[], int n) {\n    int s = 0;\n    for (int i = 1; i <= n; i++) s += a[i];\n    return s;\n}"
    rows = []
    for name in READER_ORDER:
        reset()
        started = time.perf_counter()
        reader = load_reader(name)
        load_s = time.perf_counter() - started
        if reader is None:
            rows.append({"reader": name, "loaded": False})
            continue
        read(None, sentences[0], name)                      # first call warms the session
        times = []
        for index in range(repeats):
            started = time.perf_counter()
            read(None, sentences[index % len(sentences)], name)
            times.append(time.perf_counter() - started)
        allowed_labels.cache_clear()
        started = time.perf_counter()
        allowed_labels(code)
        mask_ms = (time.perf_counter() - started) * 1000
        rows.append({"reader": name, "loaded": True, "load_s": round(load_s, 3),
                     "ms_median": round(float(np.median(times)) * 1000, 2),
                     "ms_p95": round(float(np.percentile(times, 95)) * 1000, 2),
                     "mask_first_ms": round(mask_ms, 2), "threshold": reader.threshold,
                     "version": reader.version})
    reset()
    return rows


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Read one sentence, or time the readers.")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--code")
    parser.add_argument("--reader")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args(argv)
    if args.bench:
        for row in bench():
            print(row)
        return 0
    result = read(args.code, args.text or "", args.reader)
    # ASCII only: a Windows console cannot print every character in the class subtitles.
    print(json.dumps({"status": result["status"], "reader": result["reader"],
                      "threshold": result["threshold"], "masked": result["masked"],
                      "top": top_classes(result)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
