"""Checks the suite itself: every row of the 03 §2.2 table and every §2.5 "v3 extra unit test"
has at least one test that names it. Needs neither the interpreter nor gcc.
"""
from tests.A3 import test_semantics_table, test_strings  # noqa: F401  (importing fills helpers.COVERED)
from tests.A3.helpers import COVERED, SPEC


def test_every_spec_row_has_a_test():
    missing = [f"{spec_id}: {SPEC[spec_id]}" for spec_id in SPEC if not COVERED.get(spec_id)]
    assert not missing, missing


def test_the_table_has_21_rows_and_6_extras():
    assert len([s for s in SPEC if s.startswith("2.2/")]) == 21
    assert len([s for s in SPEC if s.startswith("2.5/")]) == 6


def test_tests_name_their_row_in_the_docstring():
    for module in (test_semantics_table, test_strings):
        for name, fn in vars(module).items():
            if name.startswith("test_") and hasattr(fn, "spec_rows"):
                assert fn.__doc__ and "03 §2." in fn.__doc__, name
