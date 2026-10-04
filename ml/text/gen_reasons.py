"""Generates the reason sentences the sentence reader trains on (ml_plan/05 §7).

For each (class, context, voice) DeepSeek writes a few one-line explanations a student
with that wrong belief might type. The sentences are filtered, split by context and
written to ml/data/reasons.jsonl, plus a single bundle file for the cloud training script.

Run from the repo root:
    python -m ml.text.gen_reasons --dry-run     # show the plan and one prompt, no calls
    python -m ml.text.gen_reasons               # make the calls (cached; a rerun is free)
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS, REASON_LABELS

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = Path(__file__).parent / "prompts" / "contexts.json"
OUT = ROOT / "ml" / "data" / "reasons.jsonl"
BUNDLE = ROOT / "ml" / "data" / "reasons_bundle.json"
JOB = "reasons"
PER_CALL = 5
SPLIT_BY_CONTEXT_NUMBER = {1: "train", 2: "train", 3: "train", 4: "train", 5: "val", 6: "test"}

VOICES = {
    "confident": "sure of themselves, plain English, one sentence of 8 to 18 words",
    "unsure": "hesitant (\"i think\", \"maybe\", \"not sure but\"), one sentence of 8 to 18 words",
    "very_short": "very short, 3 to 7 words, like a quick reply",
    "typos": "typed fast on a phone: all lowercase, a spelling mistake or two, little punctuation, 6 to 15 words",
    "hinglish": "Hinglish: romanised Hindi mixed with English programming words, 6 to 16 words",
    "over_explaining": "long-winded, walks through the steps one by one, 20 to 35 words",
}

SYSTEM = ("You write the short explanations that first-year programming students in India type when a C quiz "
          "asks them \"Why did you choose that answer?\". Reply with a JSON object only.")

BANNED_EVERYWHERE = ["misconception", "off-by-one", "off by one", "uninitialized", "uninitialised"]


def description(label):
    """The text a sentence is matched against: what the class is, then the wrong belief."""
    if label == "CORRECT_REASON":
        return "Correct reasoning: the explanation matches how C really behaves."
    info = CLASS_INFO[label]
    return f"{info['subtitle']}. Wrong belief: {info['belief']}"


def build_prompt(context, voice, belief, as_correct):
    answer = context["correct"] if as_correct else context["chosen"]
    thinking = (f"The student understands this correctly: {belief}" if as_correct
                else f"The student holds this wrong belief and does not know it is wrong: {belief}")
    user = f"""A student was shown this C code and question.

Code:
{context['code']}

Question: {context['question']}
The student answered: {answer}

{thinking}

Write {PER_CALL} different explanations this student might type for why they chose that answer.
Voice: {VOICES[voice]}.

Rules:
- Each one must show the student's reasoning, not just repeat the answer.
- Write as the student, in the first person or with no subject. Do not judge the answer or say it is wrong.
- Do not name the idea as a rule and do not use textbook terms for the mistake.
- Make the {PER_CALL} explanations differ in wording and in which part of the code they point at.

Return JSON: {{"sentences": ["...", "..."]}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def plan(contexts, correct_rules):
    """Every call to make: (label, context, voice, messages)."""
    by_class = defaultdict(list)
    for context in contexts:
        by_class[context["cls"]].append(context)

    calls = []
    for cls in MISCONCEPTIONS:
        for context in by_class[cls]:
            for voice in VOICES:
                calls.append((cls, context, voice, build_prompt(context, voice, CLASS_INFO[cls]["belief"], False)))

    # Correct reasoning: one context per class, a different context number each time so the
    # train / val / test contexts all get some, and three voices per context.
    voice_names = list(VOICES)
    for n, cls in enumerate(MISCONCEPTIONS):
        context = by_class[cls][n % len(by_class[cls])]
        for v in range(3):
            voice = voice_names[(n + 2 * v) % len(voice_names)]
            calls.append(("CORRECT_REASON", context, voice, build_prompt(context, voice, correct_rules[cls], True)))
    return calls


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def keep(text, label):
    """None if the sentence is usable, else the reason it is dropped."""
    tokens = words(text)
    if len(tokens) < 3:
        return "under 3 words"
    if len(tokens) > 60:
        return "too long"
    lowered = text.lower()
    banned = list(BANNED_EVERYWHERE)
    if label in CLASS_INFO:
        info = CLASS_INFO[label]
        banned += [info["name"].lower(), info["code_name"].lower(), info["code_name"].lower().replace("_", " ")]
        if jaccard(tokens, words(info["belief"])) >= 0.8:
            return "repeats the description"
    for term in banned:
        if term in lowered:
            return f"contains '{term}'"
    return None


def collect(calls, results):
    rows, dropped, failed = [], Counter(), 0
    seen = defaultdict(list)            # label -> token lists already kept
    for (label, context, voice, _), (parsed, _, error) in zip(calls, results):
        if error or not isinstance(parsed, dict) or not isinstance(parsed.get("sentences"), list):
            failed += 1
            continue
        for sentence in parsed["sentences"]:
            if not isinstance(sentence, str):
                dropped["not text"] += 1
                continue
            text = " ".join(sentence.split())
            reason = keep(text, label)
            tokens = words(text)
            if reason is None and any(jaccard(tokens, other) >= 0.85 for other in seen[label]):
                reason = "near-duplicate"
            if reason:
                dropped[reason] += 1
                continue
            seen[label].append(tokens)
            number = int(context["id"].rsplit("_c", 1)[1])
            rows.append({
                "id": f"r-{label}-{len(seen[label]):04d}", "label": label, "context_id": context["id"],
                "code": context["code"],
                "chosen": context["correct"] if label == "CORRECT_REASON" else context["chosen"],
                "text": text, "voice": voice, "source": "deepseek", "split": SPLIT_BY_CONTEXT_NUMBER[number],
            })
    return rows, dropped, failed


def report(rows, dropped, failed):
    print(f"kept {len(rows)} sentences; calls that failed: {failed}; dropped: {dict(dropped)}")
    per_label = defaultdict(Counter)
    contexts = defaultdict(set)
    for row in rows:
        per_label[row["label"]][row["split"]] += 1
        contexts[row["label"]].add(row["context_id"])
    print(f"{'label':16} {'train':>6} {'val':>5} {'test':>5} {'contexts':>9}")
    for label in REASON_LABELS:
        c = per_label[label]
        print(f"{label:16} {c['train']:6} {c['val']:5} {c['test']:5} {len(contexts[label]):9}")
    print("voices:", dict(Counter(r["voice"] for r in rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan and one prompt; make no calls")
    parser.add_argument("--limit", type=int, help="only make the first N calls (for a trial run)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(CONTEXTS.read_text(encoding="utf-8"))
    calls = plan(data["contexts"], data["correct_rules"])
    per_class = Counter(c["cls"] for c in data["contexts"])
    assert set(per_class) == set(MISCONCEPTIONS) and set(per_class.values()) == {6}, "need 6 contexts per class"
    print(f"{len(calls)} calls planned, {PER_CALL} sentences each")
    if args.dry_run:
        print(calls[0][3][1]["content"])
        return
    if args.limit:
        calls = calls[:args.limit]

    from ml.text import llm_client
    results = llm_client.chat_json_many(JOB, [c[3] for c in calls], temperature=1.1, workers=args.workers)
    print(f"from cache: {sum(r[1] for r in results)}; new calls: {sum(not r[1] and r[2] is None for r in results)}")
    for error in sorted({r[2] for r in results if r[2]})[:3]:
        print("error:", error[:300])

    rows, dropped, failed = collect(calls, results)
    report(rows, dropped, failed)
    print("tokens used so far:", llm_client.usage_summary(JOB))
    if args.limit:
        for row in rows[:12]:
            print(f"  [{row['label']} / {row['voice']}] {row['text']}")
        return

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    BUNDLE.write_text(json.dumps({
        "labels": REASON_LABELS,
        "descriptions": {label: description(label) for label in REASON_LABELS},
        "rows": [{k: r[k] for k in ("id", "label", "context_id", "text", "voice", "split")} for r in rows],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {BUNDLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
