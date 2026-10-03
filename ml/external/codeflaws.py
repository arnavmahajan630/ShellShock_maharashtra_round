"""Codeflaws (05 §8): 3,902 buggy / fixed C program pairs from Codeforces, with tests.

Use here: a "bugs we have no class for" slice. It is contest code with scanf, so the
interpreter rejects most of it and only AST features apply; it measures how often the model
abstains or says OTHER.

    python -m ml.external.codeflaws         # fetch if needed, print counts

The benchmark is one archive (codeflaws.tar.gz, linked from codeflaws.github.io). To spare the
disk only the C sources are unpacked by default; the archive is kept, so the tests can be
unpacked later with --all without downloading again. Evaluation only; nothing is committed.
"""
import argparse
import re
import sys
import tarfile
from pathlib import Path

from ml.external import common

NAME = "codeflaws"
URL = "http://www.comp.nus.edu.sg/~release/codeflaws/codeflaws.tar.gz"
ARCHIVE = "codeflaws.tar.gz"
# <contest>-<problem>-bug-<buggy submission>-<accepted submission>
_DEFECT_RE = re.compile(r"^(\d+)-([A-Za-z0-9]+)-bug-(\d+)-(\d+)$")


def unpack(archive, into, sources_only=True):
    """Unpack the archive; with sources_only, only the *.c files. Returns the number of files written."""
    into, count = Path(into), 0
    base = into.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            if not member.isfile() or (sources_only and not member.name.endswith(".c")):
                continue
            destination = (into / member.name).resolve()
            if not str(destination).startswith(str(base)):
                raise common.FetchError(f"archive entry escapes the target folder: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, open(destination, "wb") as out:
                out.write(source.read())
            count += 1
    return count


def fetch(target=None, max_bytes=common.MAX_BYTES, force=False, sources_only=True):
    target = Path(target or common.default_target())
    folder = target / NAME
    unpacked = any(folder.glob("*/*-bug-*")) if folder.exists() else False
    if (folder / ARCHIVE).exists() and not unpacked and not force:      # downloaded earlier, never unpacked
        record = {"source": NAME, "url": URL, "limit_bytes": max_bytes, "status": "present", "folder": str(folder),
                  "reported_bytes": (folder / ARCHIVE).stat().st_size}
    else:
        record = common.fetch_file(NAME, URL, target, max_bytes, unpack=False, force=force)
    if record["status"] not in ("fetched", "present") or (unpacked and not force):
        return record
    try:
        files = unpack(folder / ARCHIVE, folder, sources_only)
    except (tarfile.TarError, OSError, common.FetchError) as exc:
        return common.log_fetch(target, {"source": NAME, "url": URL, "status": "failed", "error": f"unpack: {exc}"})
    return common.log_fetch(target, {"source": NAME, "url": URL, "status": "fetched", "folder": str(folder),
                                     "reported_bytes": record.get("reported_bytes"), "files": files,
                                     "unpacked": "C sources only" if sources_only else "everything",
                                     "bytes_on_disk": common.folder_size(folder)})


def load(root=None):
    """[{defect, contest, problem, buggy, fixed}] with the paths of the two sources of each pair."""
    root = Path(root or common.default_target() / NAME)
    rows = []
    for folder in sorted(p for p in root.glob("*/*") if p.is_dir()):
        match = _DEFECT_RE.match(folder.name)
        if not match:
            continue
        contest, problem, buggy_id, fixed_id = match.groups()
        buggy = folder / f"{contest}-{problem}-{buggy_id}.c"
        fixed = folder / f"{contest}-{problem}-{fixed_id}.c"
        if buggy.exists() and fixed.exists():
            rows.append({"defect": folder.name, "contest": contest, "problem": problem,
                         "buggy": str(buggy), "fixed": str(fixed)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", type=Path, default=common.default_target())
    parser.add_argument("--max-mb", type=float, default=common.MAX_BYTES / 1024 ** 2)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--all", action="store_true", help="unpack the tests too, not only the C sources")
    args = parser.parse_args(argv)
    if not args.no_fetch:
        record = fetch(args.target, int(args.max_mb * 1024 ** 2), force=args.all, sources_only=not args.all)
        print(common.describe(record))
        if record["status"] in ("skipped", "failed"):
            return 1
    rows = load(args.target / NAME)
    print(f"buggy/fixed pairs: {len(rows)}   problems: {len({(r['contest'], r['problem']) for r in rows})}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
