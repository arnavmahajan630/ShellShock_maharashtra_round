"""IntroClass (05 §8): buggy student C programs for six small assignments, BSD-3.

Used like Codeflaws, as a "bugs we have no class for" slice: whole programs with scanf, so only
AST features apply. github.com/ProgramRepair/IntroClass.

    python -m ml.external.introclass        # fetch if needed, print counts

Layout of the repository: <assignment>/<student hash>/<version>/<assignment>.c, with the
reference solution in <assignment>/tests/<assignment>.c. Evaluation only; nothing is committed.
"""
import argparse
import collections
import sys
from pathlib import Path

from ml.external import common

NAME = "IntroClass"
ASSIGNMENTS = ["checksum", "digits", "grade", "median", "smallest", "syllables"]


SKIP = (".log",)        # 66,543 repair-tool logs, 1.96 GB of the 1.98 GB snapshot; not needed here


def fetch(target=None, max_bytes=common.MAX_BYTES, force=False):
    return common.fetch_github_snapshot(NAME, "ProgramRepair", "IntroClass", target or common.default_target(),
                                        max_bytes, force=force, skip_suffixes=SKIP)


def load(root=None):
    """[{assignment, student, version, path}] for every student program version on disk."""
    root = Path(root or common.default_target() / NAME)
    rows = []
    for assignment in ASSIGNMENTS:
        for path in sorted((root / assignment).glob(f"*/*/{assignment}.c")):
            student, version = path.parts[-3], path.parts[-2]
            if student != "tests":
                rows.append({"assignment": assignment, "student": student, "version": version, "path": str(path)})
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
    counts = collections.Counter(r["assignment"] for r in rows)
    students = {(r["assignment"], r["student"]) for r in rows}
    print(f"program versions: {len(rows)}   (assignment, student) pairs: {len(students)}")
    for assignment in ASSIGNMENTS:
        print(f"  {assignment:<10} {counts.get(assignment, 0)}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
