"""One way to run learner code against a problem's tests, whichever backend exists.

Two backends:
  "interp"  ml.c_interp.harness  (package A1): the tree-walking interpreter; gives traces.
  "gcc"     ml.oracle.gcc        (package A2): compiles with gcc; pass/fail only, no trace.

Each backend module must provide:
    run_tests(problem: dict, code: str, sample_only: bool = False) -> dict   # schemas.RunResult
and the interpreter must also provide:
    trace(problem: dict, code: str, test_index: int | None = None) -> dict   # schemas.Trace

`problem` is a problem dict (schemas.Problem). With test_index=None, trace() records steps
for the problem's display_test and aggregates counters and events over all tests (03 §2.4).

Work that only needs pass/fail can start on the gcc backend before the interpreter is merged.
"""
import importlib

BACKENDS = {"interp": "ml.c_interp.harness", "gcc": "ml.oracle.gcc"}


class BackendUnavailable(RuntimeError):
    pass


def _load(backend, needs):
    try:
        module = importlib.import_module(BACKENDS[backend])
    except ImportError as exc:
        raise BackendUnavailable(f"backend '{backend}' is not built yet ({exc})") from exc
    if not hasattr(module, needs):
        raise BackendUnavailable(f"backend '{backend}' has no {needs}()")
    return module


def available(backend):
    """True if the backend can run tests."""
    try:
        _load(backend, "run_tests")
        return True
    except BackendUnavailable:
        return False


def run_tests(problem, code, backend="auto", sample_only=False):
    """Run the problem's tests on `code`. backend: "auto" (interpreter if built, else gcc), "interp" or "gcc"."""
    if backend == "auto":
        backend = "interp" if available("interp") else "gcc"
    return _load(backend, "run_tests").run_tests(problem, code, sample_only=sample_only)


def trace(problem, code, test_index=None):
    """Full trace from the interpreter. Raises BackendUnavailable until package A1 is merged."""
    return _load("interp", "trace").trace(problem, code, test_index=test_index)
