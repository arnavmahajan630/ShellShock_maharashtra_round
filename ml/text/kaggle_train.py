"""Fine-tunes the sentence reader on a cloud GPU (Colab or Kaggle). ml_plan/05 §6.

Self-contained: it imports nothing from this repo. Upload this file and
ml/data/reasons_bundle.json, then run

    python kaggle_train.py --data reasons_bundle.json

What it does:
  1. trains BAAI/bge-base-en-v1.5 so a student's sentence lands next to the description
     of the mistake it shows (cross-entropy over cosine similarity to all descriptions);
  2. compares it on the held-out contexts with three baselines (word counts, frozen
     embeddings + logistic regression, frozen nearest description);
  3. picks the confidence threshold below which the reader says "unsure";
  4. exports the model to ONNX and checks the export against PyTorch;
  5. writes reason_<id>.zip. Unzip it on the laptop into ml/artifacts/reason_<id>/.

  --lomo   also runs the leave-one-class-out test (17 retrains, about 1-2 hours on a T4).
"""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def ensure(packages):
    """Install anything missing (Colab and Kaggle already have torch and transformers)."""
    import importlib.util
    missing = [pip for module, pip in packages if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure([("torch", "torch"), ("transformers", "transformers"), ("sklearn", "scikit-learn"),
        ("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("onnxscript", "onnxscript")])

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline, make_union  # noqa: E402
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(torch.nn.Module):
    """Text -> one unit-length vector (the [CLS] token, as bge models are trained)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(hidden[:, 0], dim=-1)


def tokens(tokenizer, texts, max_len):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)


@torch.no_grad()
def embed(encoder, tokenizer, texts, max_len, batch_size=128):
    encoder.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        out.append(encoder(*tokens(tokenizer, texts[start:start + batch_size], max_len)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def train(encoder, tokenizer, rows, labels, descriptions, args, log=True):
    """Cross-entropy over cosine similarity between each sentence and every description."""
    index = {label: i for i, label in enumerate(labels)}
    texts = [r["text"] for r in rows]
    targets = torch.tensor([index[r["label"]] for r in rows])
    description_texts = [descriptions[label] for label in labels]

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(rows) + args.batch - 1) // args.batch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    order = list(range(len(rows)))
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        encoder.train()
        rng.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
                sentences = encoder(*tokens(tokenizer, [texts[i] for i in batch], args.max_len))
                anchors = encoder(*tokens(tokenizer, description_texts, args.max_len))
                logits = sentences.float() @ anchors.float().T / args.temperature
                loss = F.cross_entropy(logits, targets[batch].to(DEVICE))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, seen = total + loss.item() * len(batch), seen + len(batch)
        if log:
            print(f"  epoch {epoch + 1}/{args.epochs}  loss {total / seen:.4f}", flush=True)


def probabilities(encoder, tokenizer, texts, labels, descriptions, args):
    anchors = embed(encoder, tokenizer, [descriptions[label] for label in labels], args.max_len)
    return torch.softmax(embed(encoder, tokenizer, texts, args.max_len) @ anchors.T / args.temperature, dim=-1).numpy()


def scores(probs, truth, threshold=None):
    """Accuracy, macro-F1 and top-3; with a threshold, also how many answers are accepted and how many of those are right."""
    predicted = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    result = {
        "n": int(len(truth)),
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "top3": float(np.mean([t in row for t, row in zip(truth, top3)])),
    }
    if threshold is not None:
        accepted = probs.max(1) >= threshold
        result["accepted_share"] = float(accepted.mean())
        result["accepted_accuracy"] = float((predicted[accepted] == truth[accepted]).mean()) if accepted.any() else None
    return result


def pick_threshold(probs, truth, target=0.90):
    """Smallest confidence at which accepted answers are at least `target` right on validation."""
    confidence, right = probs.max(1), probs.argmax(1) == truth
    for threshold in np.unique(np.round(confidence, 3)):
        accepted = confidence >= threshold
        if accepted.sum() >= 10 and right[accepted].mean() >= target:
            return float(threshold)
    return 1.01     # never confident enough: always "unsure"


def baselines(split_rows, labels, descriptions, tokenizer, args):
    """Three readers that need no fine-tuning, scored on val and test."""
    index = {label: i for i, label in enumerate(labels)}
    x = {s: [r["text"] for r in rows] for s, rows in split_rows.items()}
    y = {s: np.array([index[r["label"]] for r in rows]) for s, rows in split_rows.items()}
    out = {}

    tfidf = make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        LogisticRegression(max_iter=2000, C=10.0))
    tfidf.fit(x["train"], y["train"])
    classes = tfidf.classes_

    def full(p):    # a class missing from train gets probability 0
        wide = np.zeros((len(p), len(labels)))
        wide[:, classes] = p
        return wide
    out["tfidf"] = {s: scores(full(tfidf.predict_proba(x[s])), y[s]) for s in ("val", "test")}

    frozen = Encoder(AutoModel.from_pretrained(args.model)).to(DEVICE)
    vectors = {s: embed(frozen, tokenizer, x[s], args.max_len).numpy() for s in x}
    head = LogisticRegression(max_iter=2000, C=10.0).fit(vectors["train"], y["train"])
    out["frozen"] = {s: scores(full(head.predict_proba(vectors[s])), y[s]) for s in ("val", "test")}
    out["frozen_nearest_description"] = {
        s: scores(probabilities(frozen, tokenizer, x[s], labels, descriptions, args), y[s]) for s in ("val", "test")}
    del frozen
    return out


def leave_one_class_out(rows, labels, descriptions, tokenizer, args):
    """For each mistake: train without any of its sentences, then see if its description alone finds them."""
    index = {label: i for i, label in enumerate(labels)}
    result = {}
    for held in [label for label in labels if label != "CORRECT_REASON"]:
        started = time.time()
        encoder = Encoder(AutoModel.from_pretrained(args.model)).to(DEVICE)
        train(encoder, tokenizer, [r for r in rows if r["label"] != held and r["split"] == "train"],
              labels, descriptions, args, log=False)
        unseen = [r for r in rows if r["label"] == held]
        probs = probabilities(encoder, tokenizer, [r["text"] for r in unseen], labels, descriptions, args)
        result[held] = scores(probs, np.full(len(unseen), index[held]))
        print(f"  {held}: top-1 {result[held]['accuracy']:.3f}  top-3 {result[held]['top3']:.3f}  "
              f"({len(unseen)} sentences, {time.time() - started:.0f} s)", flush=True)
        del encoder
    result["mean_top1"] = float(np.mean([v["accuracy"] for v in result.values()]))
    result["mean_top3"] = float(np.mean([v["top3"] for k, v in result.items() if k != "mean_top1"]))
    return result


def export_onnx(encoder, tokenizer, out_dir, args):
    """Write model.onnx and check it gives the same vectors as PyTorch."""
    import onnxruntime
    cpu = encoder.to("cpu").float().eval()
    sample = ["i thought arrays start from 1", "loop apne aap ruk jayega"]
    batch = tokenizer(sample, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
    path = out_dir / "model.onnx"
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["embedding"], opset_version=17,
                  dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                                "embedding": {0: "batch"}})
    inputs = (batch["input_ids"], batch["attention_mask"])
    try:
        torch.onnx.export(cpu, inputs, str(path), dynamo=False, **kwargs)
    except TypeError:       # older torch has no `dynamo` argument
        torch.onnx.export(cpu, inputs, str(path), **kwargs)

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    check = tokenizer(sample + ["ok"], padding=True, truncation=True, max_length=args.max_len, return_tensors="np")
    onnx_out = session.run(None, {"input_ids": check["input_ids"].astype(np.int64),
                                  "attention_mask": check["attention_mask"].astype(np.int64)})[0]
    with torch.no_grad():
        torch_out = cpu(torch.tensor(check["input_ids"]), torch.tensor(check["attention_mask"])).numpy()
    difference = float(np.abs(onnx_out - torch_out).max())
    if difference > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {difference:.5f}")
    return difference, path.stat().st_size + sum(p.stat().st_size for p in out_dir.glob("model.onnx*") if p != path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="reasons_bundle.json")
    parser.add_argument("--out", default="reason_model")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64, dest="max_len")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="use only this many training sentences (quick trial)")
    parser.add_argument("--lomo", action="store_true", help="also run the leave-one-class-out test (slow)")
    parser.add_argument("--no-export", action="store_true", dest="no_export")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else
                                 "  -- no GPU: in Colab choose Runtime > Change runtime type > T4 GPU"))

    raw = Path(args.data).read_bytes()
    bundle = json.loads(raw)
    labels, descriptions, rows = bundle["labels"], bundle["descriptions"], bundle["rows"]
    split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        random.Random(args.seed).shuffle(split_rows["train"])
        split_rows["train"] = split_rows["train"][:args.limit]
    print("sentences:", {s: len(r) for s, r in split_rows.items()}, "| labels:", len(labels))
    index = {label: i for i, label in enumerate(labels)}
    truth = {s: np.array([index[r["label"]] for r in split_rows[s]]) for s in split_rows}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("baselines (no fine-tuning) ...", flush=True)
    results = baselines(split_rows, labels, descriptions, tokenizer, args)

    print("fine-tuning ...", flush=True)
    train_started = time.time()
    encoder = Encoder(AutoModel.from_pretrained(args.model)).to(DEVICE)
    train(encoder, tokenizer, split_rows["train"], labels, descriptions, args)
    train_seconds = time.time() - train_started
    probs = {s: probabilities(encoder, tokenizer, [r["text"] for r in split_rows[s]], labels, descriptions, args)
             for s in ("val", "test")}
    threshold = pick_threshold(probs["val"], truth["val"])
    results["biencoder"] = {s: scores(probs[s], truth[s], threshold) for s in ("val", "test")}

    voices = sorted({r["voice"] for r in split_rows["test"]})
    predicted = probs["test"].argmax(1)
    results["biencoder"]["test_by_voice"] = {
        v: float(np.mean([predicted[i] == truth["test"][i] for i, r in enumerate(split_rows["test"]) if r["voice"] == v]))
        for v in voices}

    print(f"\n{'reader':28} {'val acc':>8} {'val F1':>7} {'test acc':>9} {'test F1':>8} {'test top3':>10}")
    for name in ("tfidf", "frozen", "frozen_nearest_description", "biencoder"):
        v, t = results[name]["val"], results[name]["test"]
        print(f"{name:28} {v['accuracy']:8.3f} {v['macro_f1']:7.3f} {t['accuracy']:9.3f} {t['macro_f1']:8.3f} {t['top3']:10.3f}")
    t = results["biencoder"]["test"]
    print(f"\nthreshold {threshold:.3f}: on test it answers {t['accepted_share']:.0%} of sentences "
          f"and is right on {t['accepted_accuracy'] if t['accepted_accuracy'] is None else format(t['accepted_accuracy'], '.0%')} of those")
    print("test accuracy by voice:", {k: round(v, 3) for k, v in results["biencoder"]["test_by_voice"].items()})
    print(f"training took {train_seconds:.0f} s")
    print("Note: every sentence here was written by an LLM. Only real student sentences show real accuracy.")

    if args.lomo:
        print("\nleave-one-class-out (17 retrains) ...", flush=True)
        results["leave_one_class_out"] = leave_one_class_out(rows, labels, descriptions, tokenizer, args)
        print(f"mean top-1 {results['leave_one_class_out']['mean_top1']:.3f}, "
              f"mean top-3 {results['leave_one_class_out']['mean_top3']:.3f}")

    config = {k: getattr(args, k) for k in ("model", "epochs", "lr", "batch", "max_len", "temperature", "seed")}
    run_id = hashlib.sha256(raw + json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    anchors = embed(encoder, tokenizer, [descriptions[label] for label in labels], args.max_len).numpy()
    (out_dir / "descriptions.json").write_text(json.dumps({
        "labels": labels, "texts": [descriptions[label] for label in labels],
        "embeddings": [[round(float(x), 6) for x in row] for row in anchors]}), encoding="utf-8")
    meta = {"id": run_id, "reader": "biencoder", **config, "pooling": "cls", "normalize": True,
            "threshold": threshold, "data_sha256": hashlib.sha256(raw).hexdigest(),
            "sentences": {s: len(r) for s, r in split_rows.items()}, "device": DEVICE,
            "train_seconds": round(train_seconds, 1), "torch": torch.__version__, "results": results,
            "caveat": "All sentences are LLM-written; results on real student sentences are not measured here."}

    if not args.no_export:
        tokenizer.save_pretrained(out_dir)
        try:
            difference, size = export_onnx(encoder, tokenizer, out_dir, args)
            meta["onnx"] = {"max_abs_difference": difference, "bytes": size}
            print(f"\nONNX export checked: max difference from PyTorch {difference:.2e}, {size / 1e6:.0f} MB")
        except Exception as error:      # keep the trained weights even if the export fails
            encoder.model.save_pretrained(out_dir / "hf")
            meta["onnx"] = {"error": f"{type(error).__name__}: {error}"}
            print(f"\nONNX export FAILED ({type(error).__name__}: {error}). PyTorch weights saved in {out_dir}/hf instead.")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.no_export:
        # No model inside: name it differently so it is never unzipped over a real model folder.
        archive = shutil.make_archive(f"reason_{run_id}_results_only", "zip", out_dir)
        print(f"\nDone in {time.time() - started:.0f} s. No model was exported. The results are in {out_dir}/meta.json "
              f"(also in {Path(archive).name}); keep that file as ml/artifacts/reason_{run_id}/lomo_meta.json.")
    else:
        archive = shutil.make_archive(f"reason_{run_id}", "zip", out_dir)
        print(f"\nDone in {time.time() - started:.0f} s. Download {Path(archive).name} "
              f"({Path(archive).stat().st_size / 1e6:.0f} MB) and unzip it into ml/artifacts/reason_{run_id}/")


if __name__ == "__main__":
    main()
