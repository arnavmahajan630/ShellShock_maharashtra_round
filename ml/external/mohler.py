"""Mohler / Texas short answers (05 §8): the "should say unsure" set for the sentence reader.

Real CS1 students' short answers from the University of North Texas (Mohler, Bunescu, Mihalcea,
ACL 2011, "UNT Computer Science Short Answer Dataset" v2.0), each graded 0-5 for correctness by
two annotators. They are not tagged by mistake type and match none of our classes, so a good
sentence reader should answer "unsure" on them.

    python -m ml.external.mohler            # fetch if needed, print counts

Evaluation only; nothing is committed.
"""
import argparse
import re
import sys
from pathlib import Path

from ml.external import common

NAME = "mohler"
URL = "https://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip"
_ID_RE = re.compile(r"^\s*(\d+\.\d+)\s+(.*)$")


def fetch(target=None, max_bytes=common.MAX_BYTES, force=False):
    return common.fetch_file(NAME, URL, target or common.default_target(), max_bytes, force=force)


def _clean(text):
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>|<STOP>", " ", text)).strip()


def _by_question(path):
    table = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ID_RE.match(line)
        if match:
            table[match.group(1)] = _clean(match.group(2))
    return table


def load(root=None):
    """[{question_id, question, reference, answer, score}], score = average of the two graders (0-5)."""
    data = Path(root or common.default_target() / NAME) / "data"
    questions, references = _by_question(data / "raw" / "questions"), _by_question(data / "raw" / "answers")
    rows = []
    for path in sorted((data / "raw").iterdir(), key=lambda p: [int(x) if x.isdigit() else 0 for x in p.name.split(".")]):
        if not re.fullmatch(r"\d+\.\d+", path.name):
            continue
        answers = [m.group(2) for m in map(_ID_RE.match, path.read_text(encoding="utf-8", errors="replace").splitlines()) if m]
        scores_file = data / "scores" / path.name / "ave"
        scores = [float(s) for s in scores_file.read_text(encoding="utf-8").split()] if scores_file.exists() else []
        for index, answer in enumerate(answers):
            rows.append({"question_id": path.name, "question": questions.get(path.name, ""),
                         "reference": references.get(path.name, ""), "answer": _clean(answer),
                         "score": scores[index] if index < len(scores) else None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", type=Path, default=common.default_target())
    parser.add_argument("--max-mb", type=float, default=common.MAX_BYTES / 1024 ** 2)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_fetch:
        record = fetch(args.target, int(args.max_mb * 1024 ** 2))
        print(common.describe(record))
        if record["status"] in ("skipped", "failed"):
            return 1
    rows = load(args.target / NAME)
    scored = [r["score"] for r in rows if r["score"] is not None]
    print(f"student answers: {len(rows)}   questions: {len({r['question_id'] for r in rows})}   "
          f"with a score: {len(scored)}   mean score: {sum(scored) / max(len(scored), 1):.2f} / 5")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
