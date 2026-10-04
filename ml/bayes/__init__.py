"""Bayes evidence layer (ml_plan/03 §6, ml_plan/05 §4)."""

from ml.bayes.eig import expected_information_gain, load_probes, make_best_probe, select_probe
from ml.bayes.likelihood import likelihood, map_free_answer, prediction_options
from ml.bayes.posterior import apply_text, prior, update

__all__ = [
    "apply_text",
    "expected_information_gain",
    "likelihood",
    "load_probes",
    "make_best_probe",
    "map_free_answer",
    "prediction_options",
    "prior",
    "select_probe",
    "update",
]
