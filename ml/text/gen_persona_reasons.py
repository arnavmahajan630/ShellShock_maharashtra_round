"""A second, differently made test set of reason sentences: a stand-in for classmates' answers.

ml_plan/05 §7 asks for 50 or more sentences typed by real classmates, as the honest test of the
sentence reader. The team could not collect them, so this builds the nearest thing an LLM can
give: a different model (deepseek-v4-pro, not the deepseek-flash that wrote the training
sentences) role-plays one student at a time. The student is given a character and how C works
in their head, sees a question with two options, picks one and types why. Answers are kept
only when the student picked the option their belief predicts.

About half the time the model picked the correct option in spite of the belief it was given.
Those calls are repeated with the option named; such rows have a voice ending in "_told".

It uses only the val and test contexts (numbers 5 and 6 of each class), which the training
sentences never saw. Rows are source "deepseek" with voice "persona_<name>"; they are
LLM-written and must not be reported as real student sentences.

Run from the repo root:
    python -m ml.text.gen_persona_reasons --dry-run
    python -m ml.text.gen_persona_reasons
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ml.contracts import schemas as S
from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS, REASON_LABELS
from ml.text.gen_reasons import CONTEXTS, keep

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "reasons_persona.jsonl"
JOB = "reasons_persona"
MODEL = "deepseek-v4-pro"

PERSONAS = {
    "hostel_2am": "half asleep in the hostel at 2 am, typing with one thumb, short and sloppy, no capitals",
    "topper": "the class topper: fluent, a little smug, uses proper terms but in their own words",
    "friend": "explaining to a friend on WhatsApp in Hinglish (romanised Hindi with English code words)",
    "nervous": "nervous first-timer who hedges, apologises and is never sure",
    "crammer": "studied from notes the night before: refers to what 'sir said' or what the notes show, half-remembered",
    "tracer": "worked it out on paper: mentions the actual values step by step, a bit long",
}

SYSTEM = ("You role-play one first-semester engineering student in India answering a C quiz on their phone. "
          "Stay fully in character: think and type as that student would. Reply with a JSON object only.")


def options_for(context):
    """The belief answer and the correct answer, in an order fixed by the context id."""
    pair = [context["chosen"], context["correct"]]
    if int(hashlib.sha1(context["id"].encode()).hexdigest(), 16) % 2:
        pair.reverse()
    return pair


def build_prompt(context, persona, mind, told=None):
    """`told` names the option when the first try picked the other one (the model "knew better")."""
    first, second = options_for(context)
    pick = (f'Your character picks "{told}", because of how C works in their head.' if told
            else "Pick the option your character would pick.")
    user = f"""Your character: {PERSONAS[persona]}.

How C works in your character's head (they are sure of it and nobody has corrected them):
{mind}

The quiz shows:
{context['code']}

{context['question']}
Options: "{first}" or "{second}"

{pick} Then type what they would type in the "Why?" box:
one message, at most 30 words, the way this character really types. No lecture, no rule names.

Return JSON: {{"answer": "<the option, copied exactly>", "why": "<the message>"}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def plan(data):
    """Calls for contexts 5 and 6 of every class: all personas with the wrong belief, two with the right rule."""
    names = list(PERSONAS)
    calls = []
    contexts = [c for c in data["contexts"] if c["id"].endswith(("_c5", "_c6"))]
    for n, context in enumerate(contexts):
        cls = context["cls"]
        for persona in names:
            calls.append((cls, context, persona, context["chosen"], build_prompt(context, persona, CLASS_INFO[cls]["belief"]),
                          CLASS_INFO[cls]["belief"]))
        for k in range(2):
            persona = names[(n + 3 * k) % len(names)]
            calls.append(("CORRECT_REASON", context, persona, context["correct"],
                          build_prompt(context, persona, data["correct_rules"][cls]), data["correct_rules"][cls]))
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data = json.loads(CONTEXTS.read_text(encoding="utf-8"))
    calls = plan(data)
    print(f"{len(calls)} calls planned, one sentence each")
    if args.dry_run:
        print(calls[0][4][1]["content"])
        return
    if args.limit:
        calls = calls[:args.limit]

    from ml.text import llm_client
    results = llm_client.chat_json_many(JOB, [c[4] for c in calls], temperature=1.2, workers=8, model=MODEL)

    def picked(parsed, expected):
        return str(parsed.get("answer", "")).strip().strip('"').lower() == expected.lower()

    # Second try for the students who "knew better": the same call, but told which option they pick.
    retry = [i for i, (call, (parsed, _, error)) in enumerate(zip(calls, results))
             if not error and isinstance(parsed, dict) and not picked(parsed, call[3])]
    second = llm_client.chat_json_many(
        JOB, [build_prompt(calls[i][1], calls[i][2], calls[i][5], told=calls[i][3]) for i in retry],
        temperature=1.2, workers=8, model=MODEL)
    told = set(retry)
    for i, result in zip(retry, second):
        results[i] = result

    rows, dropped, seen = [], Counter(), defaultdict(int)
    for i, ((label, context, persona, expected, _, _), (parsed, _, error)) in enumerate(zip(calls, results)):
        if error or not isinstance(parsed, dict) or not isinstance(parsed.get("why"), str):
            dropped["call failed"] += 1
            continue
        if not picked(parsed, expected):
            dropped["picked the other option twice"] += 1
            continue
        text = " ".join(parsed["why"].split())
        reason = keep(text, label)
        if reason:
            dropped[reason] += 1
            continue
        seen[label] += 1
        rows.append({"id": f"p-{label}-{seen[label]:03d}", "label": label, "context_id": context["id"],
                     "code": context["code"], "chosen": expected, "text": text,
                     "voice": f"persona_{persona}" + ("_told" if i in told else ""),
                     "source": "deepseek", "split": "test"})
    for row in rows:
        S.ReasonRow.model_validate(row)

    per_label = Counter(r["label"] for r in rows)
    print(f"kept {len(rows)} of {len(calls)}; dropped: {dict(dropped)}; "
          f"told which option to pick on a second try: {sum(r['voice'].endswith('_told') for r in rows)}")
    print("per label:", {label: per_label[label] for label in REASON_LABELS})
    print("tokens:", llm_client.usage_summary(JOB))
    for row in rows[:: max(1, len(rows) // 14)]:
        print(f"  [{row['label']:14} {row['voice']:18}] {row['text']}")
    if args.limit:
        return
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
