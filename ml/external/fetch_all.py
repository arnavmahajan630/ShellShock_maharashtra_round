"""Fetch every public data source of 05 §8 and print what happened to each (package X1).

    python -m ml.external.fetch_all                       # all four, 3 GB limit per source
    python -m ml.external.fetch_all --only itsp,mohler
    python -m ml.external.fetch_all --max-mb 500 --target D:/data/external
    python -m ml.external.fetch_all --sizes               # only report sizes, download nothing

Each source's size is checked before it is downloaded; a source over the limit is skipped and
the skip is recorded in <target>/fetch_log.json. The data is for evaluation only.
"""
import argparse
import sys
from pathlib import Path

from ml.external import codeflaws, common, introclass, itsp, mohler

SOURCES = ["itsp", "codeflaws", "introclass", "mohler"]


def sizes():
    """[(source, reported bytes or None, where the figure comes from or the error)]. Downloads nothing."""
    out = []
    for name, owner, repo in [("itsp", "jyi", "ITSP"), ("introclass", "ProgramRepair", "IntroClass")]:
        try:
            out.append((name, common.github_repo_info(owner, repo)["size_bytes"], f"GitHub API, github.com/{owner}/{repo}"))
        except common.FetchError as exc:
            out.append((name, None, str(exc)))
    for name, url in [("codeflaws", codeflaws.URL), ("mohler", mohler.URL)]:
        try:
            out.append((name, common.head_size(url), f"HTTP HEAD, {url}"))
        except common.FetchError as exc:
            out.append((name, None, str(exc)))
    return sorted(out, key=lambda row: SOURCES.index(row[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", type=Path, default=common.default_target(),
                        help="folder for downloads (default: ml/data/external of the main checkout)")
    parser.add_argument("--max-mb", type=float, default=common.MAX_BYTES / 1024 ** 2,
                        help="skip a source larger than this many MB (default 3072)")
    parser.add_argument("--only", default=",".join(SOURCES), help="comma-separated: " + ", ".join(SOURCES))
    parser.add_argument("--sizes", action="store_true", help="report sizes only")
    args = parser.parse_args(argv)
    limit = int(args.max_mb * 1024 ** 2)
    wanted = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    print(f"target: {args.target}    limit per source: {common.mb(limit)}")
    for name, size, origin in sizes():
        if name in wanted:
            verdict = "unknown size" if size is None else ("over the limit: will be skipped" if size > limit else "ok")
            print(f"  {name:<11} {common.mb(size):>10}   {verdict}   ({origin})")
    if args.sizes:
        return 0

    failed = 0
    for name in wanted:
        if name == "itsp":
            code = itsp.main(["--target", str(args.target), "--max-mb", str(args.max_mb)])
        elif name == "codeflaws":
            code = codeflaws.main(["--target", str(args.target), "--max-mb", str(args.max_mb)])
        elif name == "introclass":
            code = introclass.main(["--target", str(args.target), "--max-mb", str(args.max_mb)])
        elif name == "mohler":
            code = mohler.main(["--target", str(args.target), "--max-mb", str(args.max_mb)])
        else:
            print(f"unknown source: {name}")
            code = 1
        failed += bool(code)
    print(f"log: {args.target / 'fetch_log.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
