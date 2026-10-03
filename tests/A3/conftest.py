"""Package A3: interpreter tests written from the spec (03 §2.2, §2.5; ml/contracts/subset.py; notes/W0.md).

    python -m pytest tests/A3 -q                      everything
    python -m pytest tests/A3 -q -m "not strings"     without the string tests
    RELEARN_SKIP_STRINGS=1 python -m pytest tests/A3  same, by environment variable

Tests that need the interpreter are skipped while ml.runner.available("interp") is False.
Tests that only check the expected values against gcc (marker `gcc_crosscheck`) still run.
"""

MARKERS = [
    "strings: char arrays, string literals and strlen (first thing dropped if strings are cut)",
    "gcc_crosscheck: checks this suite's expected values against gcc; does not need the interpreter",
]


def pytest_configure(config):
    for line in MARKERS:
        config.addinivalue_line("markers", line)
