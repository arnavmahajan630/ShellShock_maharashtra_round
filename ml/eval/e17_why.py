"""E17: does asking "why?" add anything? (ml_plan/05 §9), package E-c.

    python -m ml.eval.e17_why            # writes ml/eval/e17_card.json and ml/eval/e17_card.md

Collision items only: items where one wrong option is the belief answer of two or more mistakes
(`verify_quiz.collision_answers`). The option alone cannot separate those mistakes, so this is the
case the "why?" follow-up exists for (05 §4, `ask_reason`).

For each collision item, each wrong answer shared by mistakes {a, b, ...}, and each mistake c in the
group, one trial per sentence written for c:

  option alone      posterior after the 03 §6.2 update for the shared answer (ml.bayes.update).
                    The mistakes in the group are tied, so naming one is a coin flip.
  option + sentence the same posterior, then the 05 §4 text update (ml.bayes.apply_text) with the
                    reader's probabilities, unless the reader says unsure (then nothing changes).
  sentence alone    the reader's most probable mistake inside the group, ignoring the option (ablation).

Score = does the posterior's top label equal c (a tie counts as 1/size of the tie). Reported two
ways: over all 19 labels, and over the group only ("the learner is one of these").

THE SENTENCES ARE NOT WRITTEN FOR THESE ITEMS. No sentence exists that answers a collision item
(the training contexts are separate hand-written questions, and DeepSeek was not called). Each trial
uses a sentence written for mistake c in the reader's held-out sets, so the sentence is about some
other question. The reader sees the sentence alone anyway (05 §6), but a real "why?" sentence would
refer to the item. All sentences are LLM-written. Two pools: the test split of reasons.jsonl and the
persona set.
"""
from __future__ import annotations

import ml.eval._ec_common as ec  # noqa: F401  (sets the thread limits before numpy loads)

import argparse
import json
import sys

import numpy as np

from ml.bayes import apply_text, prior, update
from ml.contracts.classes import LABELS, REASON_LABELS
from ml.eval._ec_common import LLM_CAVEAT, ci_text, cluster_bootstrap, md_table, read_jsonl, row_bootstrap, write_card
from ml.items.verify_quiz import collision_answers
from ml.text import serve

POOLS = {
    "test": ("ml/data/reasons.jsonl", lambda row: row["split"] == "test",
             "test split of reasons.jsonl: held-out contexts, LLM-written, same generator as training"),
    "persona": ("ml/data/reasons_persona.jsonl", lambda row: True,
                "reasons_persona.jsonl: 270 role-played sentences, LLM-written, different model and prompt"),
}
HEADLINE_READER, HEADLINE_POOL = "tfidf", "test"


# ---------------------------------------------------------------- items and groups

def collision_groups(items):
    """[(item, shared wrong answer, [classes that believe it])] for every collision."""
    out = []
    for item in items:
        if item["type"] == "reasoning":
            continue
        for answer in collision_answers(item):
            group = [cls for cls, value in item["belief"].items() if value == answer]
            out.append((item, answer, group))
    return out


def posterior_after_option(item, answer):
    """03 §6.2 update from a uniform model prior and a population learner prior."""
    post = prior({label: 1.0 / len(LABELS) for label in LABELS})
    return update(post, answer, correct=item["correct"], belief=item["belief"], options=item["options"])


def text_probs(row):
    """Reader probabilities as P_text over the 19 labels: CORRECT_REASON is the posterior's CORRECT."""
    out = {label: 0.0 for label in LABELS}
    for label, p in zip(REASON_LABELS, row):
        out["CORRECT" if label == "CORRECT_REASON" else label] = float(p)
    return out


def tie_credit(posterior, truth, among=None, tol=1e-9):
    """1 if `truth` is the single top label, 1/k if it is tied with k-1 others, else 0."""
    labels = list(among) if among is not None else LABELS
    best = max(posterior[label] for label in labels)
    top = [label for label in labels if posterior[label] >= best - tol]
    return (1.0 / len(top)) if truth in top else 0.0


# ---------------------------------------------------------------- trials

def run_trials(reader, groups, pools):
    """Per-trial records for one reader."""
    classes = sorted({cls for _, _, group in groups for cls in group})
    raw = {}
    for pool_name, rows in pools.items():
        use = [r for r in rows if r["label"] in classes]
        raw[pool_name] = (use, reader.probs([r["text"] for r in use]))
    trials = []
    for item, answer, group in groups:
        base = posterior_after_option(item, answer)
        allowed = serve.allowed_labels(item["code"])
        if allowed:
            allowed = tuple(ok or label in item["classes"] for label, ok in zip(REASON_LABELS, allowed))
        for pool_name, (rows, probs) in raw.items():
            for row, p in zip(rows, probs):
                if row["label"] not in group:
                    continue
                masked = serve.mask_probs(p, allowed)[0]
                status = serve.status_of(masked, reader.threshold)
                after = apply_text(base, text_probs(masked), unsure=(status == "unsure"))
                soft = apply_text(base, text_probs(masked), unsure=False)     # variant: ignores the threshold
                inside = [cls for cls in group]
                sentence_only = max(inside, key=lambda cls: masked[REASON_LABELS.index(cls)])
                trials.append({
                    "item": item["item_id"], "pool": pool_name, "truth": row["label"], "group": tuple(sorted(group)),
                    "status": status, "sentence_id": row["id"], "context": row["context_id"],
                    "option_all": tie_credit(base, row["label"]),
                    "option_group": tie_credit(base, row["label"], inside),
                    "both_all": tie_credit(after, row["label"]),
                    "both_group": tie_credit(after, row["label"], inside),
                    "soft_all": tie_credit(soft, row["label"]),
                    "soft_group": tie_credit(soft, row["label"], inside),
                    "sentence_only_group": 1.0 if sentence_only == row["label"] else 0.0,
                    "p_true_before": base[row["label"]], "p_true_after": after[row["label"]],
                })
    return trials


def summarise(trials):
    """Metrics for one list of trials, with intervals over items and over trials."""
    n = len(trials)
    if not n:
        return {"n": 0}
    items = [t["item"] for t in trials]
    contexts = [t["context"] for t in trials]
    out = {"n_trials": n, "n_items": len(set(items)), "n_contexts": len(set(contexts)),
           "unsure": round(float(np.mean([t["status"] == "unsure" for t in trials])), 4)}
    for key in ("option_all", "option_group", "both_all", "both_group", "soft_all", "soft_group", "sentence_only_group"):
        values = np.array([t[key] for t in trials])
        out[key] = round(float(values.mean()), 4)
        out[key + "_ci_items"] = cluster_bootstrap(values, items)
        out[key + "_ci_contexts"] = cluster_bootstrap(values, contexts)
        out[key + "_ci_trials"] = row_bootstrap(values)
    for a, b, name in (("both_group", "option_group", "gain_group"), ("both_all", "option_all", "gain_all")):
        diff = np.array([t[a] - t[b] for t in trials])
        out[name] = round(float(diff.mean()), 4)
        out[name + "_ci_items"] = cluster_bootstrap(diff, items)
        out[name + "_ci_contexts"] = cluster_bootstrap(diff, contexts)
        out[name + "_ci_trials"] = row_bootstrap(diff)
    answered = [t for t in trials if t["status"] != "unsure"]
    out["both_group_when_answered"] = round(float(np.mean([t["both_group"] for t in answered])), 4) if answered else None
    out["n_answered"] = len(answered)
    out["p_true_before"] = round(float(np.mean([t["p_true_before"] for t in trials])), 4)
    out["p_true_after"] = round(float(np.mean([t["p_true_after"] for t in trials])), 4)
    return out


def by_pair(trials):
    out = {}
    for pair in sorted({t["group"] for t in trials}):
        sub = [t for t in trials if t["group"] == pair]
        row = summarise(sub)
        row["items"] = sorted({t["item"] for t in sub})
        row["per_class"] = {}
        for cls in pair:
            mine = [t for t in sub if t["truth"] == cls]
            row["per_class"][cls] = {"n_trials": len(mine), "option_group": round(float(np.mean([t["option_group"] for t in mine])), 4),
                                     "both_group": round(float(np.mean([t["both_group"] for t in mine])), 4)}
        out["/".join(pair)] = row
    return out


# ---------------------------------------------------------------- card

def build_card(readers, items, pools, log=lambda message: None):
    groups = collision_groups(items)
    summary, pairs, all_trials = [], {}, {}
    for name, reader in readers.items():
        if reader is None:
            summary.append({"reader": name, "loaded": False})
            continue
        trials = run_trials(reader, groups, pools)
        all_trials[name] = trials
        log(f"{name}: {len(trials)} trials")
        for pool_name in pools:
            sub = [t for t in trials if t["pool"] == pool_name]
            summary.append({"reader": name, "pool": pool_name, "loaded": True, "threshold": round(float(reader.threshold), 4),
                            **summarise(sub)})
        pairs[name] = {pool_name: by_pair([t for t in trials if t["pool"] == pool_name]) for pool_name in pools}

    head = next((r for r in summary if r.get("reader") == HEADLINE_READER and r.get("pool") == HEADLINE_POOL and r.get("loaded")), None)
    if head is None:
        head = next((r for r in summary if r.get("loaded")), {})
    n_items = len({item["item_id"] for item, _, _ in groups})
    return {
        "id": "E17",
        "title": 'Does asking "why?" add anything? (collision items only)',
        "slice": f"{n_items} collision items from quiz_items.json; sentences from the {HEADLINE_POOL} pool (LLM-written)",
        "n": head.get("n_trials", 0),
        "metrics": {
            "reader": head.get("reader"), "pool": head.get("pool"),
            "option_alone_group": head.get("option_group"), "option_plus_sentence_group": head.get("both_group"),
            "gain_group": head.get("gain_group"), "gain_group_ci95_contexts": head.get("gain_group_ci_contexts"),
            "gain_group_ci95_items": head.get("gain_group_ci_items"),
            "option_alone_all19": head.get("option_all"), "option_plus_sentence_all19": head.get("both_all"),
        },
        "per_class": {pair: {cls: v for cls, v in row["per_class"].items()}
                      for pair, row in pairs.get(head.get("reader"), {}).get(head.get("pool"), {}).items()},
        "plots": [],
        "generated_at": ec.today(),
        "caveat": LLM_CAVEAT + " The sentences were not written for these items (see not_measured).",
        "n_items": n_items,
        "n_groups": len(groups),
        "protocol": {
            "units": "a trial = (collision item, shared wrong answer, mistake c in the group, one sentence written for c)",
            "option_alone": "ml.bayes.prior (uniform model prior, population learner prior) then ml.bayes.update with the shared wrong answer",
            "option_plus_sentence": "then ml.bayes.apply_text with the reader's probabilities (CORRECT_REASON counted as CORRECT), skipped when the reader says unsure",
            "masking": "serve.allowed_labels(item code) then serve.mask_probs, with the item's own classes never removed (the keep= argument of read())",
            "score": "top label equals c; a tie of k labels counts 1/k. 'group' = among the mistakes that share the answer, 'all19' = over all 19 labels",
            "intervals": (f"95% percentile, {ec.N_BOOT} resamples. _ci_contexts resamples whole sentence contexts "
                          "(the one to quote: the sentences of one context are about one question, and the test pool has one context "
                          f"per mistake); _ci_items resamples the {n_items} items, which differ little within a pair of mistakes, so it is "
                          "narrow; _ci_trials resamples single trials and is narrowest"),
            "pools": {name: POOLS[name][2] + f" (n sentences used: {sum(1 for r in rows if r['label'] in {c for _, _, g in groups for c in g})})"
                      for name, rows in pools.items()},
        },
        "not_measured": [
            "Sentences written for these items: none exist. Trials use sentences about other questions, written for the same mistake. A sentence about the item itself could be easier or harder to read.",
            "Real students: the learner never exists here. The 'learner' is a sentence drawn from a pool of LLM-written text.",
            "The probability that a learner with mistake c picks the shared answer (p_b): the option update is taken from the table as written (03 §6.2), not simulated.",
            "Whether the app asks the follow-up on every collision item: ask_reason is true on all of these by construction (collision_answers).",
        ],
        "summary": summary,
        "by_pair": pairs,
    }


def to_markdown(card):
    out = ["# E17: does asking \"why?\" add anything?", "",
           f"Generated {card['generated_at']} by `python -m ml.eval.e17_why`.", "", f"**Caveat.** {card['caveat']}", "",
           f"Collision items: {card['n_items']} items, {card['n_groups']} shared wrong answers. Each trial is one sentence "
           "written for one of the mistakes that share the answer. **The sentences are about other questions**, not these items.", ""]
    out += ["## Headline (naming the right mistake among those that share the answer)", ""]
    rows = []
    for r in card["summary"]:
        if not r.get("loaded"):
            rows.append([r["reader"], "not loaded", "", "", "", "", "", ""])
            continue
        rows.append([r["reader"], r["pool"], r["n_trials"], f"{r['option_group']:.2f}",
                     f"{r['both_group']:.2f}", f"{r['gain_group']:+.2f}", ci_text(r["gain_group_ci_contexts"]),
                     f"{r['unsure']:.0%}"])
    out += md_table(["Reader", "Sentences", "Trials", "Option alone", "Option + sentence", "Gain", "Gain 95% (resample contexts)", "Reader unsure"], rows)
    out += ["Option alone is a tie between the mistakes in the group, so it scores 1/size (0.50 for every pair here). "
            "\"Reader unsure\" trials leave the posterior unchanged.", ""]
    out += ["## Same, over all 19 labels and the sentence-only ablation", ""]
    rows = [[r["reader"], r["pool"], f"{r['option_all']:.2f}", f"{r['both_all']:.2f}", f"{r['sentence_only_group']:.2f}",
             f"{r['soft_group']:.2f}", ci_text(r["soft_group_ci_contexts"]),
             "-" if r["both_group_when_answered"] is None else f"{r['both_group_when_answered']:.2f}", r["n_answered"]]
            for r in card["summary"] if r.get("loaded")]
    out += md_table(["Reader", "Sentences", "Option alone (19 labels)", "Option + sentence (19 labels)",
                     "Sentence alone (group)", "Variant: sentence always used (group)", "95% (contexts)",
                     "Option + sentence, answered only", "Answered"], rows)
    out += ["How to read this. The 05 §4 rule skips the sentence when the reader is unsure, and on these items the "
            "reader is unsure for roughly half of the sentences; those trials stay at a coin flip, which is what pulls "
            "\"option + sentence\" below \"sentence alone\". \"Sentence alone\" and the variant never skip, so they do not pay "
            "for the threshold; the variant is **not** the rule in 05 §4, it is shown to say what the threshold costs "
            "on these pairs. When the reader does answer, the pair is right almost every time (\"answered only\"). "
            "\"Sentence alone\" is scored inside the group only, so it is not comparable with the 19-label columns.", ""]
    out += ["## Intervals for the headline rows", ""]
    rows = [[r["reader"], r["pool"], r["n_contexts"], f"{r['both_group']:.2f}", ci_text(r["both_group_ci_contexts"]),
             ci_text(r["both_group_ci_items"]), ci_text(r["both_group_ci_trials"]),
             f"{r['gain_group']:+.2f}", ci_text(r["gain_group_ci_contexts"]), ci_text(r["gain_group_ci_items"]),
             ci_text(r["gain_group_ci_trials"])] for r in card["summary"] if r.get("loaded")]
    out += md_table(["Reader", "Sentences", "Contexts", "Option + sentence", "95% (contexts)", "95% (items)", "95% (trials)",
                     "Gain", "Gain 95% (contexts)", "Gain 95% (items)", "Gain 95% (trials)"], rows)
    out += ["## By pair of mistakes (reader " + str(card["metrics"]["reader"]) + ", " + str(card["metrics"]["pool"]) + " sentences)", ""]
    pairs = card["by_pair"].get(card["metrics"]["reader"], {}).get(card["metrics"]["pool"], {})
    rows = []
    for pair, r in pairs.items():
        per = ", ".join(f"{c}: {v['option_group']:.2f}→{v['both_group']:.2f}" for c, v in r["per_class"].items())
        rows.append([pair, len(r["items"]), r["n_trials"], f"{r['option_group']:.2f}", f"{r['both_group']:.2f}", f"{r['gain_group']:+.2f}", per])
    out += md_table(["Pair", "Items", "Trials", "Option alone", "Option + sentence", "Gain", "Per mistake (alone→with sentence)"], rows)
    out += ["## Not measured", ""] + [f"- {line}" for line in card["not_measured"]] + [""]
    out += ["## Protocol", ""] + [f"- **{k}**: {v}" for k, v in card["protocol"].items() if isinstance(v, str)] + [""]
    return "\n".join(out)


def run(log=print):
    ec.limit_threads()
    items = json.loads((ec.ROOT / "ml" / "data" / "quiz_items.json").read_text(encoding="utf-8"))
    pools = {}
    for name, (path, keep, _) in POOLS.items():
        pools[name] = [row for row in read_jsonl(path) if keep(row)]
    readers = {name: serve.load_reader(name) for name in serve.READER_ORDER}
    for name, reader in readers.items():
        log(f"{name}: {'loaded' if reader is not None else 'not loaded, skipped'}")
    return build_card(readers, items, pools, log=log)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    card = run()
    path = write_card(card, "e17", to_markdown(card))
    print(f"wrote {path}")
    for r in card["summary"]:
        if r.get("loaded"):
            print(f"{r['reader']:10} {r['pool']:8} alone {r['option_group']:.2f}  with sentence {r['both_group']:.2f}  "
                  f"gain {r['gain_group']:+.2f} {r['gain_group_ci_contexts']} (contexts)  unsure {r['unsure']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
