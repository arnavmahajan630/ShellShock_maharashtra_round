"""The code is used only to remove classes the snippet makes impossible (05 §6)."""
import numpy as np
import pytest

from ml.contracts.classes import LABELS, REASON_LABELS
from ml.contracts.feature_names import GROUP_A, MASK_PRECONDITIONS
from ml.features.ast_feats import ast_features
from ml.model.mask import masked_out
from ml.text import serve
from tests.T2.conftest import check_shape

LOOP_SUM = """int total(int a[], int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += a[i];
    return s;
}"""
RECURSION = """int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}"""
BSEARCH = """int find(int a[], int n, int t) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (a[mid] == t) return mid;
        if (a[mid] < t) low = mid; else high = mid - 1;
    }
    return -1;
}"""
SWAP = """void swap_ends(int a[], int n) {
    a[0] = a[n - 1];
    a[n - 1] = a[0];
}"""
STRINGS = """int same(char s[], char t[]) {
    if (s == t) return 1;
    return 0;
}"""
ALL_CODES = [LOOP_SUM, RECURSION, BSEARCH, SWAP, STRINGS]


def removed(code):
    allowed = serve.allowed_labels(code)
    return None if allowed is None else {label for label, ok in zip(REASON_LABELS, allowed) if not ok}


class Fixed:
    """A reader that always gives the same probabilities."""
    name, threshold, version = "tfidf", 0.55, "fixed"

    def __init__(self, **by_label):
        self.row = np.zeros(serve.N)
        for label, p in by_label.items():
            self.row[REASON_LABELS.index(label)] = p

    def probs(self, texts):
        return np.tile(self.row, (len(texts), 1))


@pytest.fixture()
def fixed(monkeypatch):
    def install(**by_label):
        reader = Fixed(**by_label)
        monkeypatch.setattr(serve, "get_reader", lambda *args, **kwargs: reader)
    return install


def test_every_precondition_is_a_reader_label_and_correct_reason_has_none():
    assert set(MASK_PRECONDITIONS) <= set(REASON_LABELS) - {"CORRECT_REASON"}


def test_a_plain_loop_removes_the_classes_that_need_other_structure():
    assert removed(LOOP_SUM) == {"D02", "D03", "D04", "D05", "D06", "D07", "D08"}


def test_each_structure_keeps_its_own_classes():
    assert removed(RECURSION) == {"D02", "D03", "D04", "D08"}
    assert removed(BSEARCH) == {"D03", "D04", "D05", "D06", "D07", "D08"}
    assert removed(SWAP) == {"D02", "D05", "D06", "D07", "D08"}
    assert removed(STRINGS) == {"D02", "D03", "D04", "D05", "D06", "D07"}


@pytest.mark.parametrize("code", ALL_CODES)
def test_correct_reason_and_the_main_classes_are_never_removed(code):
    assert not removed(code) & {"CORRECT_REASON", "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10"}


@pytest.mark.parametrize("code", ALL_CODES)
def test_same_classes_removed_as_the_diagnosers_mask(code):
    """One table, one function: the reader removes exactly what ml/model/mask.py removes."""
    feats, _ = ast_features(code)
    diagnoser = set(masked_out([feats[name] for name in GROUP_A], features=GROUP_A, labels=LABELS))
    assert removed(code) == diagnoser


@pytest.mark.parametrize("code", [None, "", "  ", "not C at all {{{", "int f( {", "for (i = 0; i < n; i++) s += a[i];",
                                  "int x = 3, y = 8;\nx = y;\ny = x;", "// low = 2, high = 3\nlow = mid;", 42])
def test_no_code_or_unparsable_code_or_a_bare_fragment_is_not_masked(code):
    assert serve.allowed_labels(code) is None
    row = np.arange(1.0, serve.N + 1) / np.arange(1.0, serve.N + 1).sum()
    assert serve.mask_probs(row, serve.allowed_labels(code))[0].tolist() == pytest.approx(row.tolist())


HELPER_NOT_SHOWN = """void sort(int a[], int n) {
    for (j = 0; j < n - 1; j++)
        if (a[j] > a[j + 1]) swap(a, j, j + 1);
}"""
HELPER_SHOWN = """void swap(int a[], int i, int j) {
    int t = a[i];
    a[i] = a[j];
    a[j] = t;
}
""" + HELPER_NOT_SHOWN
PRINTS = """int add(int a, int b) {
    printf("%d", a + b);
    return 0;
}"""


def test_code_that_calls_a_helper_it_does_not_show_is_not_masked():
    """The array writes are inside `swap`; their absence here proves nothing (D04 must stay)."""
    assert not serve.self_contained(HELPER_NOT_SHOWN)
    assert serve.allowed_labels(HELPER_NOT_SHOWN) is None


def test_code_that_defines_its_helper_is_masked_and_keeps_the_sort_classes():
    assert serve.self_contained(HELPER_SHOWN)
    assert removed(HELPER_SHOWN) == {"D02", "D05", "D06", "D07", "D08"}


def test_library_calls_do_not_stop_masking():
    assert serve.self_contained(PRINTS)
    assert removed(PRINTS) == {"D02", "D03", "D04", "D05", "D06", "D07", "D08"}


def test_masking_never_removes_the_label_of_a_generated_sentence():
    """Every context of the two sentence sets: its code never rules out the label written for it."""
    import json
    rows = []
    for name in ("reasons.jsonl", "reasons_persona.jsonl"):
        rows += [json.loads(line) for line in (serve.ROOT / "ml" / "data" / name).read_text(encoding="utf-8").splitlines()]
    pairs = {(row["code"], row["label"]) for row in rows}
    masked_contexts = 0
    for code, label in pairs:
        gone = removed(code)
        masked_contexts += gone is not None
        assert gone is None or label not in gone, (label, code)
    assert masked_contexts >= 10                       # the check is not empty


def test_keep_protects_named_labels(fixed):
    fixed(D05=0.5, M02=0.3, CORRECT_REASON=0.2)
    kept = serve.read(LOOP_SUM, "it stops by itself", keep=["D05"])
    assert kept["probs"][REASON_LABELS.index("D05")] == pytest.approx(0.5)
    assert "D05" not in kept["masked"] and "D06" in kept["masked"]
    assert check_shape(serve.read(LOOP_SUM, "it stops by itself", keep="not a list"))["masked"]


def test_read_removes_an_impossible_class_and_renormalises(fixed):
    fixed(D05=0.5, M02=0.3, CORRECT_REASON=0.2)
    without = check_shape(serve.read(None, "it stops by itself"))
    assert without["status"] == "unsure" and without["masked"] == []
    assert without["probs"][REASON_LABELS.index("D05")] == pytest.approx(0.5)

    with_code = check_shape(serve.read(LOOP_SUM, "it stops by itself"))
    assert with_code["probs"][REASON_LABELS.index("D05")] == 0.0
    assert with_code["probs"][REASON_LABELS.index("M02")] == pytest.approx(0.6)
    assert with_code["probs"][REASON_LABELS.index("CORRECT_REASON")] == pytest.approx(0.4)
    assert "D05" in with_code["masked"] and "CORRECT_REASON" not in with_code["masked"]
    assert with_code["status"] == "matched"            # 0.6 clears the 0.55 threshold after masking

    recursive = serve.read(RECURSION, "it stops by itself")
    assert recursive["probs"][REASON_LABELS.index("D05")] == pytest.approx(0.5)
    assert "D05" not in recursive["masked"]


def test_correct_reason_survives_when_everything_else_is_removed(fixed):
    fixed(D05=0.7, CORRECT_REASON=0.3)
    result = serve.read(LOOP_SUM, "it stops by itself")
    assert result["probs"][serve.CORRECT_INDEX] == pytest.approx(1.0)
    assert result["status"] == "correct_reasoning"


def test_when_masking_would_remove_everything_the_row_is_left_alone(fixed):
    fixed(D05=0.6, D06=0.4)
    result = check_shape(serve.read(LOOP_SUM, "it stops by itself"))
    assert result["probs"][REASON_LABELS.index("D05")] == pytest.approx(0.6)


def test_unparsable_code_gives_the_same_answer_as_no_code(fixed):
    fixed(D05=0.5, M02=0.3, CORRECT_REASON=0.2)
    assert serve.read("int f( {", "x")["probs"] == serve.read(None, "x")["probs"]


def test_the_code_never_reaches_the_reader(monkeypatch):
    """The sentence alone is the input; the code only masks."""
    seen = []

    class Spy(Fixed):
        def probs(self, texts):
            seen.extend(texts)
            return super().probs(texts)
    reader = Spy(M01=1.0)
    monkeypatch.setattr(serve, "get_reader", lambda *args, **kwargs: reader)
    serve.read(LOOP_SUM, "  it counts 6 too  ")
    assert seen == ["it counts 6 too"]


def test_masking_with_the_real_word_count_reader():
    text = "i think it just stops when the work is done"
    plain, masked = serve.read(None, text), serve.read(LOOP_SUM, text)
    assert masked["reader"] == "tfidf" and set(masked["masked"]) == removed(LOOP_SUM)
    for label in masked["masked"]:
        assert masked["probs"][REASON_LABELS.index(label)] == 0.0
    kept = [i for i, label in enumerate(REASON_LABELS) if label not in masked["masked"]]
    share = sum(plain["probs"][i] for i in kept)
    assert [masked["probs"][i] for i in kept] == pytest.approx([plain["probs"][i] / share for i in kept])
