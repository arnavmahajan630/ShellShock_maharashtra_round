"""C3 acceptance: dataset.jsonl exists, is large enough, and only T1 labels clash."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "data" / "dataset.jsonl"
CLASSES = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10",
    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08",
    "OTHER", "CORRECT",
]


def test_dataset_contract():
    assert DATA.is_file(), "ml/data/dataset.jsonl is missing; run: .venv\\Scripts\\python -m ml.generate.build_dataset"
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 4000
    counts = Counter(row["label"] for row in rows)
    short = {label: counts[label] for label in CLASSES if counts[label] < 80}
    assert not short, short
    groups = defaultdict(set)
    for row in rows:
        groups[row["ast_hash"]].add(row["label"])
    bad = {digest: sorted(labels) for digest, labels in groups.items() if len(labels) > 1 and labels != {"M01", "M08"}}
    assert not bad, bad
    banned = [row["op_id"] for row in rows if row.get("op_id") in ("u1_chained", "u2_or_chain")]
    assert not banned
