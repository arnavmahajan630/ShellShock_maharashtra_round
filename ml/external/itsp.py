"""ITSP real-student slice (03 §3.4 "X (ITSP) protocol", steps 1-3).

    python -m ml.external.itsp                 # fetch if needed, label, write, print counts
    python -m ml.external.itsp --no-fetch      # only label what is already on disk

Step 1  fetch github.com/jyi/ITSP (a zip snapshot of the default branch; size checked first).
Step 2  pair  dataset/Lab-N/<problem>/<id>_buggy.c  with  <id>_correct.c, diff them, keep the
        pairs whose difference is ONE hunk of at most 3 lines on each side.
Step 3  map the hunk to a candidate class with the regex rules of 03 §3.4, in the order given
        there; the first rule that matches wins, otherwise OTHER.

Output: ml/data/itsp_slice.jsonl, one schemas.DatasetRow per kept pair, source "X",
`rater2_label: null`. These are CANDIDATE labels: step 4 of the protocol (a human hand-check of
40-80 items) has not been done. A review sheet with each hunk is written next to the download
(<target>/itsp_slice_review.tsv) to make that check quick.

The data is for evaluation only (05 §8). Neither the download nor the slice is committed.
"""
import argparse
import collections
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

from ml.contracts.schemas import DatasetRow
from ml.external import common

SLICE_PATH = common.REPO_ROOT / "ml" / "data" / "itsp_slice.jsonl"
MAX_HUNK_LINES = 3
AMBIGUOUS = {"M01": ["M01", "M08"]}         # 03: `<=` <-> `<` in a loop header is the M01/M08 twin

_HEADER_RE = re.compile(r"\A\s*/\*\s*numPass=(\d+)\s*,\s*numTotal=(\d+).*?\*/", re.S)
_LOOP_RE = re.compile(r"\b(for|while)\s*\(")
_COND_RE = re.compile(r"\b(if|while|for)\s*\(")
_DECL_RE = re.compile(r"^\s*\{?\s*(unsigned\s+|long\s+|short\s+)*(int|float|double|char|long)\b")
_FUNC_RE = re.compile(r"^\s*(?:unsigned\s+|long\s+|static\s+)*(?:int|void|float|double|char|long)\s+\**\s*(\w+)\s*\([^;{]*\)\s*\{?\s*$")


# ------------------------------------------------------------------ reading a pair

def split_header(text):
    """ITSP files start with a comment holding the test verdicts. Returns (code, numPass, numTotal)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _HEADER_RE.match(text)
    if not match:
        return text, None, None
    return text[match.end():].lstrip("\n"), int(match.group(1)), int(match.group(2))


def squash(line):
    """A line with all whitespace removed: students re-indent and re-space freely."""
    return re.sub(r"\s+", "", line)


def significant_lines(code):
    """[(line number, text)] for the non-blank lines."""
    return [(n, line.rstrip()) for n, line in enumerate(code.split("\n"), start=1) if line.strip()]


def single_hunk(buggy, correct, max_lines=MAX_HUNK_LINES):
    """Diff two programs ignoring whitespace and blank lines.

    A hunk is what `difflib.unified_diff` calls one (3 lines of context: changes closer than
    that to each other belong to the same hunk), so a line moved by a few lines is one hunk.

    Returns {"removed": [...], "added": [...], "between": [...], "moved_up": bool,
    "line": first buggy line of the hunk, "correct_line": ...} when the difference is exactly
    one hunk with at most `max_lines` removed and at most `max_lines` added lines; otherwise a
    string saying why the pair is dropped: "identical", "hunks>1" or "hunk>N".
    "between" holds the unchanged lines that lie between the first and the last change.
    """
    left, right = significant_lines(buggy), significant_lines(correct)
    matcher = difflib.SequenceMatcher(a=[squash(t) for _n, t in left], b=[squash(t) for _n, t in right], autojunk=False)
    groups = list(matcher.get_grouped_opcodes(3))
    if not groups:
        return "identical"
    if len(groups) > 1:
        return "hunks>1"
    changes = [op for op in groups[0] if op[0] != "equal"]
    removed = [t.strip() for _tag, i1, i2, _j1, _j2 in changes for _n, t in left[i1:i2]]
    added = [t.strip() for _tag, _i1, _i2, j1, j2 in changes for _n, t in right[j1:j2]]
    if len(removed) > max_lines or len(added) > max_lines:
        return f"hunk>{max_lines}"
    i1, j1 = changes[0][1], changes[0][3]
    return {"removed": removed, "added": added,
            "between": [t.strip() for _n, t in left[changes[0][2]:changes[-1][1]]],
            "moved_up": changes[0][0] == "insert" and changes[-1][0] == "delete",
            "line": left[i1][0] if i1 < len(left) else (left[-1][0] + 1 if left else 1),
            "correct_line": right[j1][0] if j1 < len(right) else (right[-1][0] if right else 1)}


# ------------------------------------------------------------------ the rules of 03 §3.4 step 3

def _relax_relops(text):
    return text.replace("<=", "<").replace(">=", ">")


_SINGLE_EQ_RE = re.compile(r"(?<![=!<>+\-*/%&|^])=(?!=)")
_INC_STMT_RE = re.compile(r"(?:\w+(?:\[[^\]]*\])?(?:\+\+|--)|(?:\+\+|--)\w+|\w+[-+]=1|(\w+)=\1[-+]1);")


def _eq_doubled(text):
    """Every text obtained by turning one single `=` of `text` into `==`, plus all of them at once."""
    return [text[:m.start()] + "==" + text[m.end():] for m in _SINGLE_EQ_RE.finditer(text)] + [_SINGLE_EQ_RE.sub("==", text)]


def _only_an_increment_added(removed_squashed, added_squashed):
    """True if the added text is the removed text plus exactly one statement like `i++;` or `i = i + 1;`."""
    before, after = "".join(removed_squashed), "".join(added_squashed)
    return any(after[:m.start()] + after[m.end():] == before for m in _INC_STMT_RE.finditer(after))


def _enclosing_function(code, line_number):
    """Name of the function whose body holds `line_number`, by scanning back for its header."""
    lines = code.split("\n")
    for index in range(min(line_number, len(lines)) - 1, -1, -1):
        match = _FUNC_RE.match(lines[index])
        if match:
            return match.group(1)
    return None


def _calls(text, name):
    return bool(name) and re.search(rf"\b{re.escape(name)}\s*\(", text) is not None


def _function_body(code, name):
    lines = code.split("\n")
    for index, line in enumerate(lines):
        match = _FUNC_RE.match(line)
        if match and match.group(1) == name:
            depth, body, opened = 0, [], False
            for row in lines[index:]:
                depth += row.count("{") - row.count("}")
                opened = opened or "{" in row
                body.append(row)
                if opened and depth <= 0:
                    break
            return "\n".join(body[1:])
    return ""


def classify(hunk, buggy, correct):
    """(label, rule name). Rules are tried in the order 03 §3.4 lists them."""
    removed, added = hunk["removed"], hunk["added"]
    r, a = [squash(x) for x in removed], [squash(x) for x in added]
    r_text, a_text = "\n".join(removed), "\n".join(added)
    paired = len(r) == len(a) and len(r) > 0
    function = _enclosing_function(correct, hunk["correct_line"])
    recursive = _calls(_function_body(correct, function), function)

    changed = [i for i in range(len(r)) if r[i] != a[i]] if paired else []

    # 1. `<=` <-> `<` in a loop header -> M01 / M08 (ambiguous twin)
    if changed and all(_relax_relops(r[i]) == _relax_relops(a[i]) and _LOOP_RE.search(removed[i]) for i in changed):
        return "M01", "le_lt_in_loop_header"
    # 2. `=` -> `==` in a condition -> M06   (a debug printf deleted in the same hunk is ignored)
    kept = [x for x in removed if not re.fullmatch(r"printf\(.*\);", squash(x))] if len(removed) > len(added) else removed
    if len(kept) == len(added) and kept:
        k = [squash(x) for x in kept]
        differing = [i for i in range(len(k)) if k[i] != a[i]]
        if differing and all(_COND_RE.search(kept[i]) and a[i] in _eq_doubled(k[i]) for i in differing):
            return "M06", "assign_to_equals_in_condition"
    # 3. removed `;` after `)` -> M07
    if paired and any(x != y for x, y in zip(r, a)) \
            and all(x == y or (_COND_RE.search(x) and re.sub(r"\);", ")", x, count=1) == y) for x, y in zip(r, a)):
        return "M07", "semicolon_after_header_removed"
    # 4. added `= 0` on a declaration -> M05
    if paired and any(x != y for x, y in zip(r, a)) \
            and all(x == y or (_DECL_RE.match(added[i]) and re.sub(r"=0(\.0*)?(?=[,;])", "", y) == x)
                    for i, (x, y) in enumerate(zip(r, a))):
        return "M05", "zero_initialiser_added"
    # 5. `(float)` or `.0` added -> M04
    def no_float(text):
        return re.sub(r"(\d)\.0*(?!\d)", r"\1", re.sub(r"\((float|double)\)", "", text))
    if paired and any(x != y for x, y in zip(r, a)) and all(x == no_float(y) for x, y in zip(r, a)):
        return "M04", "float_cast_or_point_zero_added"
    # 6. accumulator init moved out of the loop -> M03
    #    the same `x = 0;` line is deleted below a loop header and inserted above it
    if r and sorted(r) == sorted(a) and all(re.fullmatch(r"\w+=0(\.0*)?;", x) for x in r) \
            and hunk.get("moved_up") and any(_LOOP_RE.search(x) for x in hunk.get("between", [])):
        return "M03", "accumulator_init_moved_out_of_loop"
    # 7. added `i++` -> M02
    if _only_an_increment_added(r, a):
        return "M02", "increment_added"
    # 8. `printf` -> `return` -> M10
    if "printf" in r_text and re.search(r"\breturn\b", a_text) and "printf" not in a_text \
            and not re.search(r"\breturn\b", r_text):
        return "M10", "printf_to_return"
    # 9. temp variable introduced around a swap -> D03
    temp = re.search(r"\b(\w+)\s*=\s*\w+\s*\[[^\]]+\]\s*;", a_text)
    if temp and re.search(rf"\]\s*=\s*{re.escape(temp.group(1))}\s*;", a_text) \
            and not re.search(rf"\b{re.escape(temp.group(1))}\s*=", r_text):
        return "D03", "temp_variable_for_swap_added"
    # 10. `mid` -> `mid +- 1` -> D02
    if paired and any(re.search(r"=mid;", x) and re.search(r"=mid[-+]1;", y) for x, y in zip(r, a)):
        return "D02", "mid_to_mid_plus_minus_one"
    # 11. base case added in a recursive function -> D05
    if recursive and not removed and re.search(r"\bif\s*\(", a_text) and re.search(r"\breturn\b", a_text):
        return "D05", "base_case_added"
    # 12. `return` added before a recursive call -> D07
    if recursive and paired and any(y == "return" + x and _calls(removed[i], function) for i, (x, y) in enumerate(zip(r, a))):
        return "D07", "return_added_before_recursive_call"
    # 13. `"x"` -> `'x'` -> D08
    if paired and any(x != y for x, y in zip(r, a)) \
            and all(x == y or re.sub(r'"(\\?.)"', r"'\1'", x) == y for x, y in zip(r, a)):
        return "D08", "string_literal_to_char_literal"
    # 14. `else return` removed from a loop -> D01
    if re.search(r"\belse\b[^;]*\breturn\b", r_text.replace("\n", " ")) and not re.search(r"\belse\b", a_text):
        return "D01", "else_return_removed"
    return "OTHER", "no_rule"


# ------------------------------------------------------------------ building the slice

def find_pairs(root):
    """[(lab, problem, program id, buggy path, correct path)] under <root>/dataset."""
    pairs = []
    for buggy in sorted(Path(root, "dataset").glob("Lab-*/*/*_buggy.c")):
        correct = buggy.with_name(buggy.name.replace("_buggy.c", "_correct.c"))
        if correct.exists():
            pairs.append((buggy.parent.parent.name, buggy.parent.name, buggy.name.split("_")[0], buggy, correct))
    return pairs


def text_hash(code):
    """Stand-in for the generator's AST hash: these are whole programs with scanf, which the
    project's parser does not accept, so the hash is taken over the code without whitespace."""
    bare = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)
    return "h_" + hashlib.sha1(squash(bare).encode("utf-8")).hexdigest()[:12]


def build(root, max_lines=MAX_HUNK_LINES):
    """Returns (rows, review lines, drop counts)."""
    rows, review, dropped = [], [], collections.Counter()
    for lab, problem, program, buggy_path, correct_path in find_pairs(root):
        buggy, passed, total = split_header(buggy_path.read_text(encoding="utf-8", errors="replace"))
        correct, _p, _t = split_header(correct_path.read_text(encoding="utf-8", errors="replace"))
        hunk = single_hunk(buggy, correct, max_lines)
        if isinstance(hunk, str):
            dropped[hunk] += 1
            continue
        label, rule = classify(hunk, buggy, correct)
        members = AMBIGUOUS.get(label)
        digest = text_hash(buggy)
        row = {
            "id": f"X-{len(rows) + 1:06d}", "source": "X", "problem_id": f"ITSP-{problem}",
            "family": f"itsp_{lab.lower().replace('-', '')}", "split": "holdout_problem",
            "variant": None, "op_id": None, "op_variant": None, "aug": [], "code": buggy, "label": label,
            "soft_label": {m: 1 / len(members) for m in members} if members else None,
            "labels_all": members or [label], "is_two_bug": False,
            "ambiguous_group": digest if members else None, "ast_hash": digest,
            "verified": ({"tests_failed": total - passed, "tests_total": total, "predicate": [f"itsp_rule:{rule}"],
                          "passes_by_luck": False} if total is not None else None),
            "trace_summary": None, "author": f"itsp:{lab}/{problem}/{program}", "rater2_label": None,
        }
        DatasetRow.model_validate(row)
        rows.append(row)
        review.append("\t".join([row["id"], label, rule, row["author"], str(hunk["line"]),
                                 " ⏎ ".join(hunk["removed"]) or "(nothing)", " ⏎ ".join(hunk["added"]) or "(nothing)", ""]))
    return rows, review, dropped


def load(path=SLICE_PATH):
    """The slice as a list of validated rows (dicts)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        DatasetRow.model_validate(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", type=Path, default=common.default_target(),
                        help="folder for downloads (default: ml/data/external of the main checkout)")
    parser.add_argument("--out", type=Path, default=SLICE_PATH, help="where to write the slice")
    parser.add_argument("--max-mb", type=float, default=common.MAX_BYTES / 1024 ** 2, help="skip a source larger than this")
    parser.add_argument("--max-lines", type=int, default=MAX_HUNK_LINES, help="largest hunk kept, in lines per side")
    parser.add_argument("--no-fetch", action="store_true", help="do not download; use what is on disk")
    args = parser.parse_args(argv)

    root = args.target / "ITSP"
    if not args.no_fetch:
        record = common.fetch_github_snapshot("ITSP", "jyi", "ITSP", args.target, int(args.max_mb * 1024 ** 2))
        print(common.describe(record))
        if record["status"] in ("skipped", "failed"):
            print("ITSP is not available; no slice written.")
            return 1
    if not (root / "dataset").is_dir():
        print(f"{root / 'dataset'} does not exist; run without --no-fetch first.")
        return 1

    pairs = find_pairs(root)
    rows, review, dropped = build(root, args.max_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.out.write_text(text, encoding="utf-8")
    (args.target / "itsp_slice.jsonl").write_text(text, encoding="utf-8")      # a copy next to the download
    sheet = args.target / "itsp_slice_review.tsv"
    sheet.write_text("id\tcandidate\trule\tsource\tbuggy_line\tremoved\tadded\thuman_label\n" + "\n".join(review) + "\n",
                     encoding="utf-8")

    print(f"buggy/correct pairs found: {len(pairs)}")
    print(f"kept (one hunk of at most {args.max_lines} lines): {len(rows)}")
    for reason, count in sorted(dropped.items()):
        print(f"  dropped, {reason}: {count}")
    print("candidate labels (NOT hand-checked; rater2_label is null on every row):")
    counts = collections.Counter(row["label"] for row in rows)
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        note = "  (M01/M08 twin, soft label 0.5/0.5)" if label == "M01" else ""
        print(f"  {label:<6} {count}{note}")
    print(f"slice:        {args.out}")
    print(f"review sheet: {sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
