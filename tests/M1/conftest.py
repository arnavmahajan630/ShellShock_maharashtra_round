"""Shared fixtures for the M1 tests: the made-up matrix and one model trained on it.

The matrix (tests/fixtures/features_synth.npz) is random with one telltale column per class.
Scores on it say nothing about real accuracy; these tests only check that the code runs
and that the pieces agree with each other. Artifacts go to a pytest temp folder.
"""
from pathlib import Path

import pytest

from ml.model.data import load_npz
from ml.model.predict import Diagnoser
from ml.model.train import train

FIXTURES = Path(__file__).parent.parent / "fixtures"
SMALL_GRID = [{"num_leaves": 7}, {"num_leaves": 15, "min_data_in_leaf": 30}]


@pytest.fixture(scope="session")
def synth():
    return load_npz(FIXTURES / "features_synth.npz")


@pytest.fixture(scope="session")
def trained(synth, tmp_path_factory):
    """(summary, artifacts folder) of one short training run with a 2-config grid."""
    out = tmp_path_factory.mktemp("artifacts")
    summary = train(synth, out, grid=SMALL_GRID, max_rounds=60, log=None)
    return summary, out


@pytest.fixture(scope="session")
def model(trained):
    return Diagnoser.load(trained[0]["path"])
