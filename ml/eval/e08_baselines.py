"""E8 — baselines (03 §5.7, §9.2).

Same splits as E1, E2 and E4 for: majority, rules, TF-IDF + logistic regression, TF-IDF + LightGBM, and a
DeepSeek zero-shot row on R. The zero-shot row is called ONCE (52 programs, temperature 0) through
ml/text/llm_client.py; every answer is cached under ml/data/llm_raw/ea_e08_zero_shot/, so a rerun costs nothing.
The key is read inside the client and never printed. If it is missing the row says "not run".

    .venv\\Scripts\\python -m ml.eval.e08_baselines [--no-llm]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ml.contracts.classes import CLASS_INFO, LABELS
from ml.eval import _ea_common as C
from ml.model import baselines as B

JOB = "ea_e08_zero_shot"
PRED_CACHE = C.CACHE / "e08_predictions.npz"


def _fit_predict_all(train, tests):
    """Baselines trained on the whole of TRAIN, predicted on each test bundle. Cached on disk."""
    names = [t.name for t in tests]
    if PRED_CACHE.exists():
        z = np.load(PRED_CACHE, allow_pickle=False)
        if z["ids_hash"].item() == _ids_hash(train, tests):
            return {f"{n}|{b}": z[f"{n}|{b}"] for n in ("majority", "rules", "tfidf_lr", "tfidf_lgbm") for b in names}
    out = {}
    for t in tests:
        out[f"majority|{t.name}"] = B.majority_predict(train.y, len(t))
        out[f"rules|{t.name}"] = B.rules_predict(t.X)
        C.log(f"  tfidf_lr -> {t.name}")
        out[f"tfidf_lr|{t.name}"] = B.tfidf_lr_predict(train.codes, train.y, t.codes)
        C.log(f"  tfidf_lgbm -> {t.name}")
        out[f"tfidf_lgbm|{t.name}"] = B.tfidf_lgbm_predict(train.codes, train.y, t.codes)
    np.savez(PRED_CACHE, ids_hash=np.array(_ids_hash(train, tests)), **out)
    return out


def _ids_hash(train, tests):
    import hashlib
    return hashlib.sha256(json.dumps([train.ids[:50], len(train)] + [t.ids for t in tests]).encode()).hexdigest()[:16]


def _e1_baselines(mb):
    cache = C.CACHE / "e08_e1.json"
    data = mb["data"]
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["data_hash"] == data.data_hash():
            return saved["scores"]
    C.log("E1 baselines on the training folds (5 folds x 2 TF-IDF models) ...")
    res = B.run_baselines(data, mb["folds"])
    scores = res["scores"]
    cache.write_text(json.dumps({"data_hash": data.data_hash(), "scores": scores}), encoding="utf-8")
    return scores


# ------------------------------------------------------------------ DeepSeek zero-shot

def _label_text():
    lines = []
    for k in LABELS:
        if k == "CORRECT":
            lines.append("CORRECT: the program solves the task correctly (even if the style is unusual).")
        elif k == "OTHER":
            lines.append("OTHER: the program is wrong or odd, but none of the 17 beliefs above explains it.")
        else:
            info = CLASS_INFO[k]
            belief = info["belief"].replace("`", "")
            lines.append(f"{k}: {info['subtitle']}. The student believes: {belief}")
    return "\n".join(lines)


def _messages(problem, code):
    system = ("You are an experienced teacher of introductory C programming. A student wrote the function below for a "
              "task. Decide which single label best describes the student's code. Labels:\n" + _label_text() +
              '\nAnswer with a JSON object {"label": "<one label id>", "reason": "<one short sentence>"} and nothing else.')
    user = (f"Task: {problem['prompt']}\nSignature: {problem['signature']}\n\nStudent code:\n```c\n{code}\n```")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def zero_shot(bundle, allow_call=True):
    """Predictions (indices into LABELS, -1 = not run) for the scored rows of `bundle`."""
    from ml.text import llm_client
    idx = [i for i in range(len(bundle)) if bundle.scored[i]]
    lists = [_messages(C.problems()[bundle.rows[i]["problem_id"]], bundle.codes[i]) for i in idx]
    if not allow_call:
        return None, {"status": "not run (--no-llm)"}
    try:
        llm_client.model_name()                       # raises if the key or model is not configured
    except Exception as exc:
        return None, {"status": f"not run: {type(exc).__name__}: key or model not configured"}
    results = llm_client.chat_json_many(JOB, lists, temperature=0.0, workers=4)
    pred = np.full(len(bundle), -1, dtype=np.int64)
    invalid = failed = 0
    raw = {}
    for i, (parsed, cached, err) in zip(idx, results):
        if parsed is None:
            failed += 1
            continue
        label = str(parsed.get("label", "")).strip().upper()
        raw[bundle.ids[i]] = {"label": label, "reason": str(parsed.get("reason", ""))[:200]}
        if label in LABELS:
            pred[i] = LABELS.index(label)
        else:
            invalid += 1
            pred[i] = LABELS.index("OTHER")
    usage = llm_client.usage_summary(JOB)
    return pred, {"status": "ran", "model": llm_client.model_name(), "calls": len(idx), "failed": failed,
                  "invalid_label_mapped_to_OTHER": invalid, "temperature": 0.0, "usage": usage,
                  "all_from_cache": all(c for _, c, _ in results), "answers": raw}


def run(use_llm=True):
    mb = C.model_bundle()
    model, data = mb["model"], mb["data"]
    hold, real = C.holdout_bundle(), C.realistic_bundle()
    rsub = real.sub(real.scored)

    e1 = _e1_baselines(mb)
    ours_pred = C.oof_proba().argmax(axis=1)
    ours_e1 = C.fold_scores(data.Y, data.y, ours_pred, mb["folds"])
    preds = _fit_predict_all(data, [hold, rsub])

    table = {}
    names = {"majority": "majority class", "rules": "rules (03 §5.7 order, on extracted features)",
             "tfidf_lr": "TF-IDF char 3-5 + logistic regression", "tfidf_lgbm": "TF-IDF char 3-5 + LightGBM"}
    for k, label in names.items():
        e2 = C.evaluate(hold.Y, hold.y, preds[f"{k}|{hold.name}"], hold.groups, hold.domain, cluster=True)
        e4 = C.evaluate(rsub.Y, rsub.y, preds[f"{k}|{rsub.name}"], None, rsub.domain, cluster=False)
        table[k] = {"name": label,
                    "E1": {"macro_f1": e1[k]["macro_f1_mean"], "std": e1[k]["macro_f1_std"], "n": e1[k]["n"]},
                    "E2": {"macro_f1": e2["macro_f1"], "ci95": e2["macro_f1_ci95"], "n": e2["n"]},
                    "E4": {"macro_f1": e4["macro_f1"], "ci95": e4["macro_f1_ci95"], "n": e4["n"]}}
    ours2 = C.evaluate(hold.Y, hold.y, model.proba(hold.X).argmax(axis=1), hold.groups, hold.domain, cluster=True)
    ours4 = C.evaluate(rsub.Y, rsub.y, model.proba(rsub.X).argmax(axis=1), None, rsub.domain, cluster=False)
    table["diagnoser"] = {"name": "diagnoser (LightGBM on A+B+R+C, calibrated, masked)",
                          "E1": {"macro_f1": float(np.mean(ours_e1)), "std": float(np.std(ours_e1)), "n": len(data)},
                          "E2": {"macro_f1": ours2["macro_f1"], "ci95": ours2["macro_f1_ci95"], "n": ours2["n"]},
                          "E4": {"macro_f1": ours4["macro_f1"], "ci95": ours4["macro_f1_ci95"], "n": ours4["n"]}}

    pred_ds, info = zero_shot(real, allow_call=use_llm)
    if pred_ds is not None:
        p = pred_ds[real.scored]
        ok = p >= 0
        e4 = C.evaluate(rsub.Y[ok], rsub.y[ok], p[ok], None, rsub.domain[ok], cluster=False)
        answers = info.pop("answers")
        table["deepseek_zero_shot"] = {
            "name": f"DeepSeek zero-shot ({info['model']}), label list + code + task text, temperature 0",
            "E1": None, "E2": None,
            "E4": {"macro_f1": e4["macro_f1"], "ci95": e4["macro_f1_ci95"], "n": e4["n"], "acc": e4["acc"]}}
        info["answers_file"] = "E08_deepseek_answers.json"
        (C.CARDS).mkdir(parents=True, exist_ok=True)
        (C.CARDS / "E08_deepseek_answers.json").write_text(json.dumps(
            {"about": "DeepSeek zero-shot label per R-llm program (an LLM's guess, not a ground truth)", "answers": answers},
            indent=1), encoding="utf-8")
    else:
        table["deepseek_zero_shot"] = {"name": "DeepSeek zero-shot", "E1": None, "E2": None, "E4": None, "status": info["status"]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["majority", "rules", "tfidf_lr", "tfidf_lgbm", "deepseek_zero_shot", "diagnoser"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.27
    for j, exp in enumerate(("E1", "E2", "E4")):
        vals = [(table[k][exp] or {}).get("macro_f1") for k in order]
        xs = np.arange(len(order)) + (j - 1) * width
        ax.bar(xs, [v if v is not None else 0 for v in vals], width, label={"E1": "E1 grouped CV", "E2": "E2 unseen problems",
                                                                         "E4": "E4 R-llm (stand-in)"}[exp])
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 0.01, "n/a", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(np.arange(len(order)), [k.replace("_", "\n") for k in order], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.legend(fontsize=7)
    ax.set_title("E8 baselines vs diagnoser", fontsize=9)
    fig.tight_layout()
    plot = C.plot_path("E08_baselines.png")
    fig.savefig(plot, dpi=130)
    plt.close(fig)

    card = {
        "id": "E8", "title": "Baselines", "slice": "E1 folds on TRAIN; E2 = HOLDOUT-P; E4 = R-llm (LLM-written stand-in, 52 scored)",
        "n": int(len(data)), "model_version": model.model_version,
        "metrics": {k: {e: v[e] for e in ("E1", "E2", "E4")} for k, v in table.items()},
        "table": table, "deepseek": {k: v for k, v in info.items()},
        "data_status": {"R-blind": C.R_BLIND_STATUS, "R-llm": C.R_LLM_NAME},
        "plots": [plot.name],
        "caveat": ("E1 is the mean of five per-fold macro-F1 values (± std); E2 uses a cluster bootstrap by problem over 6 problems; "
                   "E4 is a row bootstrap over 52 LLM-written programs. TF-IDF baselines see source text only; rules and the "
                   "diagnoser see extracted features (including the test run). The rules baseline uses the extracted features, not the "
                   "generator's predicates. The DeepSeek row sees the task text and the code but not the test results, "
                   "was run once with a prompt written by E-a (not tuned). READ THE DEEPSEEK ROW WITH CARE: the R-llm programs were "
                   "themselves written by a DeepSeek model (deepseek-v4-pro) that was told the belief and asked to act it out, the labels are "
                   "those beliefs, and the second rater is DeepSeek too. A DeepSeek model reading code from a DeepSeek model, with the belief "
                   "wording in its own label list, is the easiest possible case for it; its high score on this stand-in is not evidence "
                   "that it beats the diagnoser on real student code. R-blind (hand-written) is missing, so this cannot be checked. "
                   "Also note: the rules baseline equals the diagnoser on E1 (the rules read the same extracted features, written from the "
                   "same plan as the generator's operators) and is clearly lower on E2 and E4."),
    }
    return card


def main():
    C.write_card(run(use_llm="--no-llm" not in sys.argv), "E08")


if __name__ == "__main__":
    main()
