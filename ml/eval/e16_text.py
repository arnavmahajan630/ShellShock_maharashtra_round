"""E16: how well does the sentence reader work? (plans/05 §9), package E-c.

    python -m ml.eval.e16_text            # writes ml/eval/e16_card.json and ml/eval/e16_card.md

For each reader that loads (tfidf, biencoder, frozen; ml/text/serve.py) it scores:
  1. held-out contexts: the test split of ml/data/reasons.jsonl;
  2. ml/data/reasons_persona.jsonl, the LLM-written stand-in for classmates' sentences, as a
     whole and split into "picked by itself" / "told which option";
  3. every voice of both sets, and Hinglish against English;
  4. the Mohler answers (real students, none of our classes): how often it stays unsure;
  5. leave-one-class-out: the fine-tuned reader's numbers are copied from notes/T1.md (17
     retrains on a cloud GPU); next to them, computed here, the unchanged model's
     nearest-description accuracy per class, which also uses no training sentence of the class.
The DeepSeek zero-shot row is not run. Nothing here is tuned: thresholds come from the heads
(chosen on the val split) and from the artifact's meta.json.

`build_card(...)` returns the card as a dict in the shape of 03 §9.4 (id, title, slice, n,
metrics, per_class, plots, caveat) with the E16 tables as extra keys, so eval/run_all.py can
call it or read the JSON file.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

from ml.contracts.classes import MISCONCEPTIONS, REASON_LABELS
from ml.text import serve
from ml.text.check_readers import scores

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "data" / "reasons.jsonl"
PERSONA = ROOT / "ml" / "data" / "reasons_persona.jsonl"
BUNDLE = ROOT / "ml" / "data" / "reasons_bundle.json"
CARD_JSON = ROOT / "ml" / "eval" / "e16_card.json"
CARD_MD = ROOT / "ml" / "eval" / "e16_card.md"
INDEX = {label: i for i, label in enumerate(REASON_LABELS)}

TEST_LABEL = "Held-out contexts (test split of reasons.jsonl; LLM-written, same generator as training)"
PERSONA_LABEL = "LLM-written stand-in for classmates' sentences"
MOHLER_LABEL = "Mohler short answers (real CS students; none of our classes applies)"
CAVEAT = ("The training sentences are LLM-written, and so are both test sets. Only Mohler is real student "
          "text, and it has no labels for our classes, so it can only show how often a reader stays unsure. "
          "No number on this card measures accuracy on real students' sentences.")
LOCO_SOURCE = "run on Colab on 2026-10-04, copied from notes/T1.md"
# Fine-tuned reader, leave-one-class-out: share of the held-out class's sentences whose nearest
# description is the held-out one (notes/T1.md, "Per class top-1"). Scored on every sentence of
# the class, all splits. tests/E-c checks these against the notes file.
LOCO_TOP1 = {"M01": 0.000, "M02": 0.067, "M03": 0.106, "M04": 0.217, "M05": 0.155, "M06": 0.011, "M07": 0.000,
             "M08": 0.044, "M10": 0.272, "D01": 0.144, "D02": 0.311, "D03": 0.250, "D04": 0.267, "D05": 0.011,
             "D06": 0.011, "D07": 0.139, "D08": 0.028}
LOCO_TOP3 = {"D02": 0.600, "M01": 0.017, "M07": 0.067}          # the only per-class top-3 values in the notes
LOCO_MEAN = {"top1": 0.120, "top3": 0.283}
NOT_RUN = [{"row": "DeepSeek zero-shot", "why": "Optional row of 05 §9. Not run: no LLM API call was made for this card."}]
BOOTSTRAP = 1000


# ---------------------------------------------------------------- loading

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_mohler():
    """The Mohler answers, or [] when the dataset is not on this machine.

    Looks in this checkout first, then where ml/external/common.py keeps downloads.
    One row per student answer; the whole answer (median 15 words) is read as one sentence.
    """
    try:
        from ml.external import common, mohler
        for root in (ROOT / "ml" / "data" / "external" / "mohler", common.default_target() / "mohler"):
            if (root / "data" / "raw").is_dir():
                return [row for row in mohler.load(root) if row["answer"].strip()]
    except Exception:
        pass
    return []


def load_descriptions(artifacts_dir=None):
    """({label: description text}, where it came from). The artifact's own texts when it is present."""
    reader = serve.load_reader("biencoder", artifacts_dir=artifacts_dir)
    if reader is not None:
        return dict(zip(REASON_LABELS, reader.descriptions)), f"ml/artifacts/{reader.folder.name}/descriptions.json"
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return {label: bundle["descriptions"][label] for label in REASON_LABELS}, "ml/data/reasons_bundle.json"


def language_of(voice):
    """Hinglish voices: `hinglish` in the training generator, the `friend` persona in the second set."""
    return "Hinglish" if voice == "hinglish" or voice.startswith("persona_friend") else "English"


def base_voice(voice):
    return voice[:-len("_told")] if voice.endswith("_told") else voice


# ---------------------------------------------------------------- scoring

def truth_of(rows):
    return np.array([INDEX[row["label"]] for row in rows], dtype=np.int64)


def accuracy_interval(right, groups, resamples=BOOTSTRAP, seed=0):
    """95% interval for accuracy, resampling whole contexts (sentences of one context are not independent)."""
    right, groups = np.asarray(right, dtype=float), np.asarray(groups)
    names = np.unique(groups)
    if len(names) < 2:
        return None
    sums = np.array([right[groups == name].sum() for name in names])
    counts = np.array([(groups == name).sum() for name in names])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(resamples, len(names)))
    values = sums[picks].sum(1) / counts[picks].sum(1)
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def measure(probs, rows, threshold, keep=None, interval=False):
    """The E16 metrics of one reader on one slice. `keep` selects rows; None = all."""
    keep = np.arange(len(rows)) if keep is None else np.asarray(keep, dtype=np.int64)
    if len(keep) == 0:
        return {"n": 0}
    truth = truth_of(rows)[keep]
    out = scores(probs[keep], truth, threshold)
    if interval:
        out["accuracy_ci95"] = accuracy_interval(probs[keep].argmax(1) == truth, [rows[i]["context_id"] for i in keep])
    return {key: (round(value, 6) if isinstance(value, float) else value) for key, value in out.items()}


def per_class(probs, rows):
    from sklearn.metrics import precision_recall_fscore_support
    truth = truth_of(rows)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, probs.argmax(1), labels=list(range(len(REASON_LABELS))), zero_division=0)
    return {label: {"n": int(support[i]), "precision": round(float(precision[i]), 3),
                    "recall": round(float(recall[i]), 3), "f1": round(float(f1[i]), 3)}
            for i, label in enumerate(REASON_LABELS)}


def masked_by_code(probs, rows):
    """The same probabilities after `read` has removed what each row's code makes impossible."""
    out = np.vstack([serve.mask_probs(probs[i], serve.allowed_labels(row["code"])) for i, row in enumerate(rows)])
    return out, int(sum(serve.allowed_labels(row["code"]) is not None for row in rows))


def slices_of(name, rows):
    """[(slice id, label, row indexes)] for the main table."""
    every = list(range(len(rows)))
    if name == "test":
        return [("test", TEST_LABEL, every)]
    told = [i for i, row in enumerate(rows) if row["voice"].endswith("_told")]
    alone = [i for i, row in enumerate(rows) if not row["voice"].endswith("_told")]
    return [("persona", PERSONA_LABEL, every),
            ("persona_picked_by_itself", PERSONA_LABEL + ": the writer picked the option by itself", alone),
            ("persona_told", PERSONA_LABEL + ": the writer was told which option (voice ends in _told)", told)]


def voice_tables(reader_name, set_name, rows, probs, threshold):
    """(per voice, Hinglish against English) for one reader on one set."""
    voices, languages = [], []
    for voice in sorted({base_voice(row["voice"]) for row in rows}):
        keep = [i for i, row in enumerate(rows) if base_voice(row["voice"]) == voice]
        voices.append({"reader": reader_name, "set": set_name, "voice": voice, "language": language_of(voice),
                       **measure(probs, rows, threshold, keep)})
    for language in ("English", "Hinglish"):
        keep = [i for i, row in enumerate(rows) if language_of(row["voice"]) == language]
        languages.append({"reader": reader_name, "set": set_name, "language": language,
                          **measure(probs, rows, threshold, keep)})
    return voices, languages


def mohler_row(reader_name, probs, threshold, rows):
    """How a reader answers sentences that show none of our mistakes. `matched` is always wrong here."""
    statuses = np.array([serve.status_of(row, threshold) for row in probs])
    top = probs.argmax(1)
    matched = statuses == "matched"
    wrong = sorted(((REASON_LABELS[k], float((top[matched] == k).sum() / len(rows))) for k in set(top[matched])),
                   key=lambda item: -item[1])[:5]
    out = {"reader": reader_name, "n": len(rows), "unsure": round(float((statuses == "unsure").mean()), 4),
           "matched_wrongly": round(float(matched.mean()), 4),
           "correct_reasoning": round(float((statuses == "correct_reasoning").mean()), 4),
           "most_often_matched": [[label, round(share, 4)] for label, share in wrong]}
    graded = np.array([row["score"] if row.get("score") is not None else np.nan for row in rows], dtype=float)
    for key, keep in (("graded_4_or_more", graded >= 4.0), ("graded_under_4", graded < 4.0)):
        if keep.any():
            out[key] = {"n": int(keep.sum()), "unsure": round(float((statuses[keep] == "unsure").mean()), 4),
                        "matched_wrongly": round(float(matched[keep].mean()), 4)}
    return out


def nearest_description(embed, descriptions, rows):
    """The unchanged model with no training: each sentence goes to the description it is closest to.

    Per class, on every sentence of the class (the rows the leave-one-class-out run scored) and
    on its test-split sentences alone.
    """
    anchors = embed([descriptions[label] for label in REASON_LABELS])
    similarity = embed([row["text"] for row in rows]) @ anchors.T
    truth = truth_of(rows)
    order = np.argsort(-similarity, axis=1)
    top1, top3 = order[:, 0] == truth, (order[:, :3] == truth[:, None]).any(1)
    test = np.array([row["split"] == "test" for row in rows])
    out = {}
    for label in REASON_LABELS:
        mine = truth == INDEX[label]
        out[label] = {"n_all": int(mine.sum()), "top1_all": round(float(top1[mine].mean()), 3),
                      "top3_all": round(float(top3[mine].mean()), 3), "n_test": int((mine & test).sum()),
                      "top1_test": round(float(top1[mine & test].mean()), 3) if (mine & test).any() else None,
                      "top3_test": round(float(top3[mine & test].mean()), 3) if (mine & test).any() else None}
    overall = {"n_test": int(test.sum()), "top1_test": round(float(top1[test].mean()), 4) if test.any() else None,
               "top3_test": round(float(top3[test].mean()), 4) if test.any() else None,
               "n_all": len(rows), "top1_all": round(float(top1.mean()), 4), "top3_all": round(float(top3.mean()), 4)}
    return out, overall


def leave_one_class_out(nearest, overall, descriptions_from):
    """The copied fine-tuned numbers next to the unchanged model's, per class."""
    block = {"source": LOCO_SOURCE,
             "finetuned": "trained without any sentence of the class, then asked to find them from its description",
             "unchanged": None, "per_class": {}, "mean": {"finetuned_top1": LOCO_MEAN["top1"], "finetuned_top3": LOCO_MEAN["top3"]}}
    for label in MISCONCEPTIONS:
        block["per_class"][label] = {"finetuned_top1": LOCO_TOP1[label], "finetuned_top3": LOCO_TOP3.get(label)}
    if nearest is None:
        block["unchanged"] = "not computed: the unchanged model is not in the Hugging Face cache on this machine"
        return block
    block["unchanged"] = (f"{serve.FROZEN_MODEL} with no training at all: nearest description (texts from "
                          f"{descriptions_from}), computed here on the same sentences (every sentence of the class)")
    for label in MISCONCEPTIONS:
        block["per_class"][label].update({f"unchanged_{key}": value for key, value in nearest[label].items()})
    for key in ("top1_all", "top3_all", "top1_test", "top3_test"):
        block["mean"][f"unchanged_{key}"] = round(float(np.mean([nearest[label][key] for label in MISCONCEPTIONS])), 3)
    block["unchanged_on_the_whole_test_split"] = overall
    return block


# ---------------------------------------------------------------- the card

def build_card(readers, test_rows, persona_rows, mohler_rows, embed=None, descriptions=None,
               descriptions_from=None, all_rows=None, log=lambda message: None):
    """readers: {name: reader or None} in the order to report. `embed` is the unchanged model (or None)."""
    sets = {"test": test_rows, "persona": persona_rows}
    card_readers, table, voices, languages, mohler_table, classes, masking = {}, [], [], [], [], {}, []
    for name, reader in readers.items():
        if reader is None:
            card_readers[name] = {"loaded": False}
            continue
        card_readers[name] = {"loaded": True, "threshold": round(float(reader.threshold), 4), "version": reader.version,
                              "threshold_from": "val split, accepted answers at least 90% right there"}
        for set_name, rows in sets.items():
            if not rows:
                continue
            started = time.perf_counter()
            probs = reader.probs([row["text"] for row in rows])
            log(f"{name}: {set_name} ({len(rows)} sentences, {time.perf_counter() - started:.1f} s)")
            for slice_id, label, keep in slices_of(set_name, rows):
                table.append({"reader": name, "set": slice_id, "set_label": label,
                              **measure(probs, rows, reader.threshold, keep, interval=True)})
            by_voice, by_language = voice_tables(name, set_name, rows, probs, reader.threshold)
            voices += by_voice
            languages += by_language
            if set_name == "test":
                classes[name] = per_class(probs, rows)
            with_code, n_masked = masked_by_code(probs, rows)
            masking.append({"reader": name, "set": set_name, "rows_with_usable_code": n_masked,
                            **measure(with_code, rows, reader.threshold)})
        if mohler_rows:
            started = time.perf_counter()
            probs = reader.probs([serve.clean(row["answer"]) for row in mohler_rows])
            log(f"{name}: mohler ({len(mohler_rows)} answers, {time.perf_counter() - started:.1f} s)")
            mohler_table.append(mohler_row(name, probs, reader.threshold, mohler_rows))

    nearest = overall = None
    if embed is not None and descriptions and all_rows:
        started = time.perf_counter()
        nearest, overall = nearest_description(embed, descriptions, all_rows)
        log(f"unchanged model, nearest description ({len(all_rows)} sentences, {time.perf_counter() - started:.1f} s)")

    loaded = [name for name, info in card_readers.items() if info["loaded"]]
    default = next((name for name in serve.READER_ORDER if name in loaded), None)
    headline = next((row for row in table if row["reader"] == default and row["set"] == "test"), {})
    return {
        "id": "E16",
        "title": "How well does the sentence reader work?",
        "slice": TEST_LABEL,
        "n": headline.get("n", 0),
        "metrics": {"reader": default, **{key: headline.get(key) for key in
                                          ("macro_f1", "accuracy", "top3", "answers", "right_when_it_answers")}},
        "per_class": classes.get(default, {}),
        "plots": [],
        "caveat": CAVEAT,
        "generated_at": datetime.date.today().isoformat(),
        "default_reader": default,
        "reader_order": list(serve.READER_ORDER),
        "readers": card_readers,
        "columns": {"accuracy": "top answer is right, no threshold", "macro_f1": "over the 18 labels, no threshold",
                    "top3": "right label among the best three",
                    "answers": "share of sentences at or above the reader's threshold (the rest are 'unsure')",
                    "right_when_it_answers": "accuracy on the sentences it answers",
                    "accuracy_ci95": f"{BOOTSTRAP} resamples of whole contexts"},
        "rows": table,
        "per_voice": voices,
        "language": languages,
        "with_code_masking": {"what": "The same sentences with each row's code passed to read(). Only whole, "
                                      "self-contained code is used for masking (it parses as C functions and calls "
                                      "nothing it does not define), so bare quiz fragments change nothing",
                              "rows": masking},
        "per_class_by_reader": classes,
        "mohler": {"set_label": MOHLER_LABEL, "n": len(mohler_rows), "rows": mohler_table,
                   "what": "Every 'matched' is wrong: these answers are about other questions and show none of our "
                           "17 mistakes. 'correct_reasoning' is listed apart; it updates nothing about a mistake."
                   if mohler_rows else "not run: the Mohler files are not on this machine (python -m ml.external.mohler)"},
        "leave_one_class_out": leave_one_class_out(nearest, overall, descriptions_from),
        "not_run": NOT_RUN,
    }


# ---------------------------------------------------------------- Markdown

def _pct(value):
    return "-" if value is None else f"{value:.0%}"


def _num(value):
    return "-" if value is None else f"{value:.3f}"


def _table(header, lines):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(line) + " |" for line in lines] + [""]


def to_markdown(card):
    out = ["# E16: how well does the sentence reader work?", "", f"Generated {card['generated_at']} by `python -m ml.eval.e16_text`.", "",
           f"**Caveat.** {card['caveat']}", ""]
    out += ["## Readers", ""]
    out += _table(["Reader", "Loaded", "Threshold", "Version"],
                  [[name, "yes" if info["loaded"] else "no", _num(info.get("threshold")), str(info.get("version", "-"))]
                   for name, info in card["readers"].items()])
    out += [f"Serving order: {', '.join(card['reader_order'])}, then none. Default here: `{card['default_reader']}`. "
            "Thresholds were chosen on the val split; nothing was tuned on the sets below.", ""]

    out += ["## 1 and 2. Held-out contexts and the second test set", "",
            f"`test` = {TEST_LABEL}. `persona` = **{PERSONA_LABEL}** (`reasons_persona.jsonl`, a different model and "
            "prompt from the training sentences). It is not text written by students.", ""]
    out += _table(["Reader", "Set", "n", "Accuracy", "95% interval", "Macro-F1", "Top-3", "Answers", "Right when it answers"],
                  [[row["reader"], row["set"], str(row["n"]), _num(row.get("accuracy")),
                    "-" if not row.get("accuracy_ci95") else f"{row['accuracy_ci95'][0]:.2f} to {row['accuracy_ci95'][1]:.2f}",
                    _num(row.get("macro_f1")), _num(row.get("top3")), _pct(row.get("answers")),
                    _pct(row.get("right_when_it_answers"))] for row in card["rows"]])
    out += ["Intervals resample whole contexts. \"Answers\" is the share at or above the threshold; the rest are `unsure`.", ""]

    out += ["## 3. By voice", ""]
    out += _table(["Reader", "Set", "Voice", "Language", "n", "Accuracy", "Top-3", "Answers", "Right when it answers"],
                  [[row["reader"], row["set"], row["voice"], row["language"], str(row["n"]), _num(row.get("accuracy")),
                    _num(row.get("top3")), _pct(row.get("answers")), _pct(row.get("right_when_it_answers"))]
                   for row in card["per_voice"]])
    out += ["Persona voices merge the `_told` rows with the rest. Hinglish against English:", ""]
    out += _table(["Reader", "Set", "Language", "n", "Accuracy", "Top-3", "Answers", "Right when it answers"],
                  [[row["reader"], row["set"], row["language"], str(row["n"]), _num(row.get("accuracy")),
                    _num(row.get("top3")), _pct(row.get("answers")), _pct(row.get("right_when_it_answers"))]
                   for row in card["language"]])

    mohler = card["mohler"]
    out += ["## 4. Mohler: should say unsure", "", f"{MOHLER_LABEL}. n = {mohler['n']}. {mohler['what']}", ""]
    if mohler["rows"]:
        out += _table(["Reader", "n", "Unsure", "Wrongly `matched`", "`correct_reasoning`", "Most often matched to"],
                      [[row["reader"], str(row["n"]), _pct(row["unsure"]), _pct(row["matched_wrongly"]),
                        _pct(row["correct_reasoning"]),
                        ", ".join(f"{label} {share:.1%}" for label, share in row["most_often_matched"]) or "-"]
                       for row in mohler["rows"]])

    loco = card["leave_one_class_out"]
    out += ["## 5. Leave-one-class-out", "",
            f"Fine-tuned reader: **{loco['source']}**. {loco['finetuned'].capitalize()}.", "",
            f"Unchanged model: {loco['unchanged']}.", ""]
    lines = []
    for label, row in loco["per_class"].items():
        lines.append([label, _num(row["finetuned_top1"]), _num(row.get("finetuned_top3")), _num(row.get("unchanged_top1_all")),
                      _num(row.get("unchanged_top3_all")), str(row.get("unchanged_n_all", "-")),
                      _num(row.get("unchanged_top1_test")), str(row.get("unchanged_n_test", "-"))])
    mean = loco["mean"]
    lines.append(["**mean of 17**", _num(mean["finetuned_top1"]), _num(mean["finetuned_top3"]), _num(mean.get("unchanged_top1_all")),
                  _num(mean.get("unchanged_top3_all")), "", _num(mean.get("unchanged_top1_test")), ""])
    out += _table(["Class", "Fine-tuned, class held out: top-1", "top-3", "Unchanged model: top-1 (all sentences)",
                   "top-3", "n", "Unchanged model: top-1 (test split)", "n"], lines)
    out += ["The fine-tuned top-3 per class is in the notes for three classes only.", ""]

    masking = card["with_code_masking"]
    out += ["## Extra: with the code passed for masking", "", masking["what"] + ".", ""]
    out += _table(["Reader", "Set", "Rows with usable code", "n", "Accuracy", "Top-3", "Answers", "Right when it answers"],
                  [[row["reader"], row["set"], str(row["rows_with_usable_code"]), str(row["n"]), _num(row.get("accuracy")),
                    _num(row.get("top3")), _pct(row.get("answers")), _pct(row.get("right_when_it_answers"))]
                   for row in masking["rows"]])

    out += ["## 6. Not run", ""] + [f"- **{item['row']}**: {item['why']}" for item in card["not_run"]] + [""]
    return "\n".join(out)


# ---------------------------------------------------------------- command line

def run(log=print):
    rows = load_jsonl(DATA)
    readers = {name: serve.load_reader(name) for name in serve.READER_ORDER}
    for name, reader in readers.items():
        log(f"{name}: {'loaded' if reader is not None else 'not loaded, skipped'}")
    frozen = readers.get("frozen")
    embed = frozen.embed if frozen is not None else None
    if embed is None:
        try:
            embed = serve.frozen_embedder()[0]
        except Exception:
            embed = None
    descriptions, descriptions_from = load_descriptions()
    mohler_rows = load_mohler()
    log(f"mohler: {len(mohler_rows)} answers" if mohler_rows else "mohler: not on this machine, skipped")
    return build_card(readers, [row for row in rows if row["split"] == "test"], load_jsonl(PERSONA), mohler_rows,
                      embed=embed, descriptions=descriptions, descriptions_from=descriptions_from, all_rows=rows, log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=Path, default=CARD_JSON)
    parser.add_argument("--md", type=Path, default=CARD_MD)
    args = parser.parse_args(argv)
    card = run()
    args.json.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md.write_text(to_markdown(card), encoding="utf-8")
    print(f"wrote {args.json} and {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
