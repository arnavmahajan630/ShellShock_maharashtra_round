"""Diagnosis pipeline (package S2): one learner program in, one `diagnosis` object out.

    gate (G1) -> run + trace (A1) -> features (F3) -> model (M1) -> [learner prior] ->
    [predict item] -> [probe answers] (D1) -> decide (M1) -> evidence (M1)

No FastAPI in here: `routes/attempt.py` is the thin HTTP layer. Everything is a plain function
of dicts so S1, S3 and the evaluation scripts can call it too.

Public functions
    get_problem(problem_id)            full problem dict (ml/problems/{main,dsa}) or None
    warm()                             load the model, probe bank, problems and reference traces
    run(problem, code, sample_only)    POST /run
    analyse(problem, code, ...)        gate, run, features, model: returns an `Analysis`
    diagnose(analysis, answers)        the diagnosis dict for an analysis plus probe answers
    attempt(problem, code, ...)        analyse + diagnose in one call
    code_item_rule(...)                the 05 §4 `fix_bug` rule

Numbers (thresholds, probe limits) are read from `ml.contracts.params` by the packages that
use them; nothing is hand-set here.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.contracts.classes import MISCONCEPTIONS

log = logging.getLogger("relearn.pipeline")

ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIRS = (ROOT / "ml" / "problems" / "main", ROOT / "ml" / "problems" / "dsa")
CODE_ITEMS_PATH = ROOT / "ml" / "data" / "code_items.json"

FIX_BUDGET_MS = 150.0           # the fixers (R1) are skipped once a request has used this much time
PREDICTION_EVENTS = ("predict", "prediction", "predict_answer", "predict_item")

_lock = threading.RLock()
_state = {"problems": None, "model": None, "model_error": None, "probes": None, "refs": {}, "code_items": None,
          "code_items_mtime": None}

# Hook for package S1: callable(learner_id) -> {class: P(active)} or None. Until it is set (or when
# it returns None) the learner prior step is skipped, as for the stateless /lab/diagnose.
learner_prior_fn = None


class PipelineError(ValueError):
    """The request is wrong (unknown probe, answer not an option). The route turns it into a 422."""


class ModelUnavailable(RuntimeError):
    """No diagnoser artifact. The route falls back to the older rule-based path."""


# ---------------------------------------------------------------- loading

def _problems():
    with _lock:
        if _state["problems"] is None:
            found = {}
            for folder in PROBLEM_DIRS:
                for path in sorted(folder.glob("*.json")):
                    problem = json.loads(path.read_text(encoding="utf-8"))
                    found[problem["problem_id"]] = problem
            _state["problems"] = found
        return _state["problems"]


def get_problem(problem_id):
    return _problems().get(problem_id)


def _model():
    with _lock:
        if _state["model"] is None:
            if _state["model_error"] is not None:
                raise ModelUnavailable(_state["model_error"])
            try:
                from ml.model.predict import Diagnoser
                _state["model"] = Diagnoser.latest(ROOT / "ml" / "artifacts")
            except FileNotFoundError as exc:
                _state["model_error"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return _state["model"]


def model_version():
    try:
        return _model().model_version
    except ModelUnavailable:
        return "unavailable"


def _probe_bank():
    with _lock:
        if _state["probes"] is None:
            from ml.bayes.eig import load_probes
            _state["probes"] = load_probes()
        return _state["probes"]


def _probe_by_id(probe_id):
    for probe in _probe_bank():
        if probe["probe_id"] == probe_id:
            return probe
    raise PipelineError(f"unknown probe {probe_id!r}")


def _reference(problem):
    """(reference trace, run_reference) for a problem: the first correct variant, cached."""
    pid = problem["problem_id"]
    with _lock:
        hit = _state["refs"].get(pid)
    if hit is None:
        from ml import runner
        from ml.features.relation_feats import make_run_reference
        hit = (runner.trace(problem, problem["correct_variants"][0]),
               make_run_reference(problem, runner.run_tests))
        with _lock:
            _state["refs"][pid] = hit
    return hit


def warm():
    """Load everything a first request would otherwise pay for. Safe to call from a thread."""
    try:
        _model()
        _probe_bank()
        for problem in _problems().values():
            _reference(problem)
        # One throw-away pass through the model and the extractor, so imports and caches are hot.
        problem = get_problem("P03") or next(iter(_problems().values()))
        analyse(problem, problem["correct_variants"][0])
    except ModelUnavailable:
        log.warning("no diagnoser artifact; /attempt will use the older rule-based path")
    except Exception:                                   # warming must never stop the server
        log.exception("warm-up failed")


def _code_items():
    with _lock:
        if not CODE_ITEMS_PATH.exists():
            return {}
        mtime = CODE_ITEMS_PATH.stat().st_mtime
        if _state["code_items"] is None or _state["code_items_mtime"] != mtime:
            items = json.loads(CODE_ITEMS_PATH.read_text(encoding="utf-8"))
            _state["code_items"] = {item["item_id"]: item for item in items}
            _state["code_items_mtime"] = mtime
        return _state["code_items"]


def get_code_item(item_id):
    return _code_items().get(item_id) if item_id else None


# ---------------------------------------------------------------- run (POST /run)

def _gate(problem, code):
    from server.app import gate as gate_module
    text = gate_module.normalise(code or "")
    return text, gate_module.check(problem, code or "")


def run(problem, code, sample_only=False):
    """POST /run: gate, trace and tests. No diagnosis."""
    from ml import runner
    started = time.perf_counter()
    text, gate = _gate(problem, code)
    out = {"gate": gate, "trace": None, "tests": None}
    if gate["code"] == "G0":
        out["tests"] = runner.run_tests(problem, text, sample_only=sample_only)["tests"]
        out["trace"] = runner.trace(problem, text)
    out["model_version"] = model_version()
    out["latency_ms"] = _ms(started)
    return out


def _ms(started):
    return round((time.perf_counter() - started) * 1000.0, 1)


# ---------------------------------------------------------------- analysis (code -> model probabilities)

@dataclass
class Analysis:
    """Everything about one submission that does not depend on probe answers."""
    problem: dict
    code: str
    gate: dict
    trace: dict | None = None
    tests: dict | None = None
    row: np.ndarray | None = None
    meta: dict | None = None
    p_code: dict | None = None
    base: dict | None = None                    # posterior before probe answers
    contributions: np.ndarray | None = None
    knn_dist: float = 0.0
    items: list = field(default_factory=list)   # evidence items from the Bayes layer (YOU PREDICTED)
    started: float = 0.0
    seconds_used: float = 0.0
    timings: dict = field(default_factory=dict)

    @property
    def passed(self):
        return bool(self.tests) and self.tests["total"] > 0 and self.tests["passed"] == self.tests["total"]


def _prediction_from_events(events):
    for event in events or ():
        if isinstance(event, dict) and event.get("type") in PREDICTION_EVENTS:
            payload = event.get("payload") or {}
            for key in ("answer", "value", "prediction"):
                if payload.get(key) is not None:
                    return payload[key]
    return None


def analyse(problem, code, *, learner_id=None, prediction=None, events=None):
    """Gate, run, trace, features and model for one submission."""
    from ml import runner
    from ml.bayes import posterior as bayes
    from ml.features.extract import extract

    started = time.perf_counter()
    text, gate = _gate(problem, code)
    analysis = Analysis(problem=problem, code=text, gate=gate, started=started)
    if gate["code"] != "G0":
        return analysis
    model = _model()

    t0 = time.perf_counter()
    run_result = runner.run_tests(problem, text)
    trace = runner.trace(problem, text)
    analysis.tests, analysis.trace = run_result["tests"], trace
    analysis.timings["run"] = _ms(t0)

    if trace["status"] in ("parse_error", "unsupported"):
        # gate.check() is pure pycparser syntax, no execution, and can't see a semantic
        # rejection the interpreter only finds while building the program (e.g. a parameter
        # redeclared in the function's own scope: legal-looking C, illegal C). Finding out here
        # is too late to skip feature extraction on the strength of gate.check() alone — code
        # that never actually ran has no real evidence behind it, and classifying it anyway
        # produces a diagnosis built on nothing (every test's model_answer defaulting to blank).
        # Promote it to a real gate rejection, same shape gate.check() itself returns for G3a.
        from ml.c_interp import harness
        from ml.contracts.subset import GATE_MESSAGES
        error = harness.check(text, problem)
        analysis.gate = {"code": "G3a", "message": GATE_MESSAGES["G3a"].format(n=error["line"], reason=error["reason"])}
        return analysis

    t0 = time.perf_counter()
    ref_trace, run_reference = _reference(problem)
    row, meta = extract(problem, text, trace, ref_trace, run_reference, run_result=run_result)
    analysis.row, analysis.meta = row, meta
    analysis.timings["features"] = _ms(t0)

    t0 = time.perf_counter()
    probs = model.proba(row)[0]
    analysis.p_code = model.posterior_dict(probs)
    analysis.contributions = model.contributions(row)
    analysis.knn_dist = float(model.knn_distance(row)[0])
    analysis.timings["model"] = _ms(t0)

    posterior = _normalise(analysis.p_code)
    prior = _learner_prior(learner_id)
    if prior is not None:
        posterior = bayes.prior(posterior, prior)
    prediction = prediction if prediction is not None else _prediction_from_events(events)
    posterior, analysis.items = _apply_prediction(posterior, problem, prediction, bayes)
    analysis.base = posterior
    analysis.seconds_used = time.perf_counter() - started
    return analysis


def _normalise(d):
    total = sum(max(v, 0.0) for v in d.values())
    if total <= 0:
        raise PipelineError("the model returned no probability mass")
    return {k: max(v, 0.0) / total for k, v in d.items()}


def _learner_prior(learner_id):
    if learner_id and learner_prior_fn is not None:
        try:
            return learner_prior_fn(learner_id)
        except Exception:
            log.exception("learner prior lookup failed; continuing without it")
    return None


def _apply_prediction(posterior, problem, prediction, bayes):
    """The mission's predict item (03 §6.5): scored with the same likelihood table."""
    item = problem.get("predict_item")
    if prediction is None or not item:
        return posterior, []
    try:
        posterior = bayes.update(posterior, prediction, correct=item["correct"], belief=item.get("belief"),
                                 options=item.get("options"))
    except ValueError:
        try:                                            # not one of the options: treat as a free answer
            posterior = bayes.update(posterior, prediction, correct=item["correct"], belief=item.get("belief"))
        except (ValueError, TypeError):
            return posterior, []
    text = f"You predicted {prediction} for \"{item['question']}\"; the run says {item['correct']}."
    return posterior, [{"type": "YOU PREDICTED", "text": text}]


# ---------------------------------------------------------------- diagnosis

def diagnose(analysis, answers=()):
    """The `diagnosis` object (03 §11.1) for an analysis plus probe answers applied in order.

    answers: [{"probe_id", "answer"}]. A probe id that appears twice counts once (the first answer).
    """
    from ml.bayes import posterior as bayes
    from ml.bayes.eig import make_best_probe
    from ml.model import novelty as novelty_module
    from ml.model.decide import decide
    from ml.model.evidence import build_evidence, model_items

    if analysis.gate["code"] != "G0":
        return None                                     # the route answers with `diagnosis: null`, as the fixtures do
    model = _model()
    version = model.model_version

    posterior = dict(analysis.base)
    asked, probe_items = [], []
    for answer in answers:
        probe = _probe_by_id(answer["probe_id"])
        if probe["probe_id"] in asked:
            continue
        given = str(answer["answer"])
        if given not in [str(o) for o in probe["options"]]:
            raise PipelineError(f"answer {given!r} is not an option of {probe['probe_id']}")
        posterior = bayes.update(posterior, given, correct=probe["correct"], belief=probe.get("belief"),
                                 options=probe["options"])
        asked.append(probe["probe_id"])
        probe_items.append({"type": "PROBE", "text": f"You said {given} for \"{probe['prompt']}\" ({probe['code']})."})

    vector = np.array([posterior.get(label, 0.0) for label in model.labels])
    novelty = novelty_module.assess(analysis.knn_dist, vector, model.tau_d, model.tau_p, model.labels)

    fixer = _FixHelper(analysis)
    diagnosis = decide(posterior, gate_code="G0", tests_passed=analysis.passed, novelty=novelty,
                       fix_check=fixer.two_bug, best_probe=make_best_probe(asked), probes_asked=asked,
                       model_version=version)

    top = diagnosis["top"][0]["id"]
    status = diagnosis["status"]
    if status == "correct" and not diagnosis["latent"]:
        diagnosis["evidence"] = []
    else:
        explained = diagnosis["latent"]["class"] if diagnosis["latent"] else top
        fix_desc = fix_line = None
        if status in ("confident", "two_bug") and explained in MISCONCEPTIONS:
            fix_desc, fix_line = fixer.describe(explained)
        diagnosis["evidence"] = build_evidence(
            analysis.contributions, explained, analysis.meta, status=status, twin_set=diagnosis["twin_set"],
            fix_desc=fix_desc, fix_line=fix_line, extra_items=analysis.items + probe_items)
        if status == "two_bug" and len(diagnosis["top"]) > 1:
            second = model_items(analysis.contributions, diagnosis["top"][1]["id"], analysis.meta, k=1)
            diagnosis["evidence"] += [{key: v for key, v in item.items() if v is not None} for item in second]
    return diagnosis


class _FixHelper:
    """The fixers of package R1, run only for the classes that need them and only while time is left."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.begun = time.perf_counter()

    def _time_left(self):
        """Time already used by the analysis plus time since this diagnosis began."""
        used = self.analysis.seconds_used + (time.perf_counter() - self.begun)
        return used * 1000.0 < FIX_BUDGET_MS

    def describe(self, cls):
        """(rule text, first changed line) of a verified minimal fix, or (None, None)."""
        if not self._time_left():
            return None, None
        try:
            from ml.learner import fixer
            info = fixer.probe(self.analysis.problem, self.analysis.code, cls)
        except Exception:
            log.exception("fixer failed for %s", cls)
            return None, None
        fix = info["fix"]
        if info["hit"] and fix and fix.get("kind") == "minimal" and fix.get("rule"):
            lines = fix.get("changed_lines") or []
            return fix["rule"], (lines[0] if lines else None)
        return None, None

    def two_bug(self, top1, top2):
        """03 §5.5: fix(top1) alone fails, fix(top2) alone fails, fix(top1) then fix(top2) passes."""
        if not self._time_left():
            return False
        try:
            from ml.learner import fixer
            problem, code = self.analysis.problem, self.analysis.code
            if fixer.probe(problem, code, top1)["hit"] or fixer.probe(problem, code, top2)["hit"]:
                return False
            for _, partly_fixed in fixer._candidates(code, top1):
                if not self._time_left():
                    return False
                if fixer.probe(problem, partly_fixed, top2)["hit"]:
                    return True
        except Exception:
            log.exception("two-bug check failed")
        return False


def attempt(problem, code, *, learner_id=None, prediction=None, events=None, answers=()):
    """Analyse and diagnose in one call. Returns the response body without ids."""
    started = time.perf_counter()
    analysis = analyse(problem, code, learner_id=learner_id, prediction=prediction, events=events)
    diagnosis = diagnose(analysis, answers) if analysis.gate["code"] == "G0" else None
    return analysis, {
        "gate": analysis.gate, "trace": analysis.trace, "tests": analysis.tests, "diagnosis": diagnosis,
        "model_version": model_version(), "latency_ms": _ms(started),
    }


# ---------------------------------------------------------------- code items (05 §4)

def _norm_line(text):
    return "".join(text.split())


def code_item_rule(item, code, diagnosis, passed):
    """What a submission to a code item means for the knowledge model.

    complete_snippet: a normal attempt (apply_diagnosis True).
    fix_bug, planted k:
        all tests pass                                  -> response "correct", diagnosis rule not applied
        top-1 = j != k with p >= 0.5                    -> the learner wrote bug j: normal rule for j
        otherwise (planted bug still there, or an unclear failure)
                                                        -> response "wrong", diagnosis rule not applied
    """
    from ml.contracts.params import ITEM_GUESS_SLIP
    out = {"item_id": item["item_id"], "type": item["type"], "apply_diagnosis": True, "response": None}
    if item["type"] != "fix_bug":
        return out
    g, s = ITEM_GUESS_SLIP["fix_bug"]
    planted = item.get("planted")
    out.update(planted=planted, g=g, s=s, apply_diagnosis=False)
    if passed:
        out["response"] = "correct"
        return out
    top = (diagnosis or {}).get("top") or []
    top1, p1 = (top[0]["id"], top[0]["p"]) if top else (None, 0.0)
    if top1 and top1 != planted and top1 in MISCONCEPTIONS and p1 >= 0.5:
        out.update(response="other_bug", apply_diagnosis=True, other=top1)
        return out
    starter = (item.get("starter") or "").splitlines()
    lines = code.splitlines()
    unchanged = all(
        n - 1 < len(starter) and n - 1 < len(lines) and _norm_line(starter[n - 1]) == _norm_line(lines[n - 1])
        for n in item.get("bug_lines") or [])
    out.update(response="wrong", bug_lines_unchanged=unchanged)
    return out
