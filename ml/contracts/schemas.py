"""Shapes of every JSON object that crosses a package or the API.

Contract: plans/03 §2.4, §3.3, §3.6, §6.3, §8.3, §9.4, §11 and plans/05 §3, §5, §7.
Field names are fixed. Where 03 and 05 leave an inner shape open, the shape chosen here
is marked "W0 decision".

Use `Model.model_validate(obj)` to check an object and `.model_dump(exclude_none=True)` to emit one.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- execution (03 §2.4)

class Event(BaseModel):
    """A trace event. Extra fields per type are listed in subset.EVENT_FIELDS."""
    model_config = ConfigDict(extra="allow")
    type: str
    line: int


class Step(Strict):
    i: int
    line: int
    vars: dict[str, Any]
    events: list[Event] = []
    effects: list[str] = []


class PerTest(Strict):
    """Counters for one test. The sums on Trace are what the frontend reads;
    per-test values are what b_iter_delta_const_pm1, b_branch_always/never and
    b_return_first_iter need (03 §2.4)."""
    status: Literal["ok", "timeout", "runtime_error", "parse_error", "unsupported"]
    returned: Any = None
    printed: str = ""
    loop_iters: dict[str, int] = {}
    branch: dict[str, dict[str, int]] = {}
    effects_count: dict[str, int] = {}
    max_depth: int = 0


class Trace(Strict):
    status: Literal["ok", "timeout", "runtime_error", "parse_error", "unsupported"]
    returned: Any = None
    printed: str = ""
    steps: list[Step] = []
    loop_iters: dict[str, int] = {}                  # "L<line>" -> iterations, summed over tests
    branch: dict[str, dict[str, int]] = {}           # "B<line>" -> {"true": n, "false": m}, summed
    events: list[Event] = []
    effects_count: dict[str, int] = {}               # summed over tests
    max_depth: int = 0                               # maximum over tests
    truncated: bool = False
    per_test: list[PerTest] = []                     # one entry per test, same order as problem.tests


class TestResult(Strict):
    args: list[Any]
    expected: dict[str, Any]
    got: dict[str, Any]
    passed: bool = Field(alias="pass")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Tests(Strict):
    passed: int
    total: int
    results: list[TestResult]


class RunResult(Strict):
    """What ml.runner.run_tests returns (W0 decision: `tests` plus how the run ended)."""
    status: Literal["ok", "timeout", "runtime_error", "parse_error", "unsupported"]
    backend: Literal["interp", "gcc"]
    tests: Tests


class Gate(Strict):
    code: Literal["G0", "G1", "G2", "G3a", "G3b", "G3c", "G4", "G4b", "G5", "G5b", "G6", "G7"]
    message: str


# ---------------------------------------------------------------- problems (03 §3.3)

class ProblemTest(Strict):
    args: list[Any]
    expect: dict[str, Any]      # keys: returned, printed, effect names, array0, max_depth_le
    adversarial: str | None = None
    sample: bool = False


class PredictItem(Strict):
    item_id: str
    code: str
    question: str
    options: list[str]
    correct: str
    belief: dict[str, str] = {}


class Problem(Strict):
    problem_id: str
    name: str
    planet: str | None = None           # main-game problems
    sector: str | None = None           # DSA problems
    world: str
    family: str
    split: Literal["train", "holdout_problem"]
    difficulty: int = 1
    prompt: str
    signature: str
    starter: str
    correct_variants: list[str]
    tests: list[ProblemTest]
    display_test: int = 0
    predict_item: PredictItem | None = None
    allowed_ops: list[str]
    markers: list[str] = []
    forbid: list[str] = []
    exposure: dict[str, float] = {}
    one_pass: str | None = None         # sorts only: reference single-pass helper for r_eq_one_pass


class PublicPredictItem(Strict):
    item_id: str
    code: str
    question: str
    options: list[str]
    correct: str


class PublicTest(Strict):
    args: list[Any]
    expect: dict[str, Any]


class PublicProblem(Strict):
    """GET /problems: no hidden tests, correct variants, exposure or belief answers."""
    problem_id: str
    name: str
    planet: str | None = None
    sector: str | None = None
    difficulty: int
    world: str
    signature: str
    starter: str
    prompt: str
    predict_item: PublicPredictItem | None = None
    markers: list[str] = []
    sample_tests: list[PublicTest] = []


# ---------------------------------------------------------------- dataset (03 §3.6)

class Verified(Strict):
    tests_failed: int
    tests_total: int
    predicate: list[str] = []
    passes_by_luck: bool = False


class TraceSummary(Strict):
    status: str
    events: list[str] = []
    loop_iters_delta: float | None = None


class DatasetRow(Strict):
    id: str
    # "R-llm": written by an LLM role-playing a student. A stand-in for the hand-written
    # R sets; never report it as hand-written or as real student code.
    source: Literal["A", "E", "AMB", "R-blind", "R-team", "R-llm", "U", "X"]
    problem_id: str
    family: str
    split: Literal["train", "holdout_problem"]
    variant: str | None = None
    op_id: str | None = None
    op_variant: str | None = None
    aug: list[str] = []
    code: str
    label: str
    soft_label: dict[str, float] | None = None
    labels_all: list[str] = []
    is_two_bug: bool = False
    ambiguous_group: str | None = None
    ast_hash: str
    verified: Verified | None = None
    trace_summary: TraceSummary | None = None
    author: str | None = None
    rater2_label: str | None = None


# ---------------------------------------------------------------- choice items (03 §6.3, §8.3, §8.5.1; 05 §3.1)

class Probe(Strict):
    probe_id: str
    twin_sets: list[str]
    code: str
    prompt: str
    options: list[str]
    correct: str
    belief: dict[str, str]


class PublicProbe(Strict):
    probe_id: str
    prompt: str
    code: str
    options: list[str]
    eig_bits: float | None = None
    for_class: str | None = None        # deferred probes in the exam report


class TrapItem(Strict):
    """ml/data/items.json (W0 decision: same fields as a probe, one class per trap)."""
    item_id: str
    type: Literal["trap"] = "trap"
    target: str
    code: str
    question: str
    options: list[str]
    correct: str
    belief: dict[str, str]


class ExamTraceItem(Strict):
    """ml/data/exam_items.json."""
    item_id: str
    kind: Literal["trace"] = "trace"
    sector: str
    difficulty: int = 1
    code: str
    question: str
    options: list[str]
    correct: str
    belief: dict[str, str]


class StateQuery(Strict):
    var: str
    after_line: int
    hit: int


class ItemVerified(Strict):
    interp: bool = False
    gcc: bool = False
    manual: bool = False


class QuizItem(Strict):
    """ml/data/quiz_items.json (05 §3.1)."""
    item_id: str
    type: Literal["mcq", "predict_output", "next_state", "reasoning"]
    concept: str
    classes: list[str]
    code: str = ""
    question: str
    options: list[str] = []
    correct: str | None = None
    belief: dict[str, str] = {}
    state_query: StateQuery | None = None
    expected: str | None = None         # reasoning items: "CORRECT_REASON"
    explain: str = ""
    difficulty: int = 1
    verified: ItemVerified = ItemVerified()


class PublicQuizItem(Strict):
    item_id: str
    type: Literal["mcq", "predict_output", "next_state", "reasoning"]
    concept: str
    code: str = ""
    question: str
    options: list[str] = []
    difficulty: int = 1


# ---------------------------------------------------------------- code items (05 §3.2)

class Hole(Strict):
    id: str
    line: int
    kind: str
    choices: list[str] = []


class CodeItem(Strict):
    """ml/data/code_items.json. Fields used depend on `type`."""
    item_id: str
    type: Literal["complete_snippet", "fix_bug", "debug_line"]
    problem_id: str
    starter: str | None = None          # complete_snippet (with ____ holes), fix_bug (buggy code)
    holes: list[Hole] = []              # complete_snippet
    exposes: list[str] = []             # complete_snippet
    planted: str | None = None          # fix_bug, debug_line (null = correct code)
    op_id: str | None = None
    code: str | None = None             # debug_line
    bug_lines: list[int] = []
    allow_no_bug: bool = True           # debug_line
    explain: str = ""
    fix: dict[str, Any] | None = None   # debug_line: {"code", "changed_lines"}


class PublicCodeItem(Strict):
    item_id: str
    type: Literal["complete_snippet", "fix_bug", "debug_line"]
    problem_id: str
    starter: str | None = None
    holes: list[Hole] = []
    code: str | None = None
    allow_no_bug: bool = True


# ---------------------------------------------------------------- diagnosis (03 §11.1)

class TopClass(Strict):
    id: str
    p: float
    name: str | None = None
    subtitle: str | None = None
    band: Literal["Likely", "Possible", "Unsure"] | None = None


class Novelty(Strict):
    knn_dist: float
    tau_d: float
    p_max: float
    tau_p: float
    abstain: bool


class EvidenceItem(Strict):
    type: Literal["CODE", "RUN", "YOU PREDICTED", "PROBE", "HISTORY", "EXAM"]
    text: str
    line: int | None = None
    feature: str | None = None
    weight: float | None = None


class Latent(BaseModel):
    """Set when every test passed but the model still names a misconception (passes-by-luck, 03 §5.5)."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    cls: str = Field(alias="class")
    p: float


class Diagnosis(Strict):
    status: Literal["confident", "ambiguous", "novel", "two_bug", "correct", "gate"]
    posterior: dict[str, float] = {}
    top: list[TopClass] = []
    twin_set: str | None = None
    two_bug: bool = False
    novelty: Novelty | None = None
    evidence: list[EvidenceItem] = []
    next_probe: PublicProbe | None = None
    probes_asked: list[str] = []
    latent: Latent | None = None
    model_version: str


# ---------------------------------------------------------------- intervention (03 §11.2)

class TimelineStep(Strict):
    step: int
    line: int
    vars: dict[str, Any] = {}
    effect: str | None = None
    flag: str | None = None


class Fix(Strict):
    kind: Literal["minimal", "reference"]
    code: str
    changed_lines: list[int] = []
    rule: str | None = None
    verified: bool = False


class Intervention(BaseModel):
    """Extra keys allowed: each modality adds its own panel (memory_strip, call_stack, window_strip, value_meter)."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    cls: str = Field(alias="class")
    modality: Literal["trace_timeline", "memory_strip", "window_strip", "value_meter", "call_stack",
                      "counterexample", "minimal_fix"]
    next_modalities: list[str] = []
    copy_lines: list[str] = Field(default=[], alias="copy")
    question: dict[str, Any] | None = None
    timeline: list[TimelineStep] = []
    counterexample: dict[str, Any] | None = None
    fix: Fix | None = None


# ---------------------------------------------------------------- learner (03 §8, §11)

class Condition(Strict):
    id: str
    label: str
    met: bool
    detail: str = ""


class KnowledgeEntry(Strict):
    """One (learner, class) record. Inner fields are a W0 decision, taken from the 02 D09 star card."""
    state: Literal["UNSEEN", "ACTIVE", "TREATING", "PROBATION", "STABLE", "MASTERED", "RELAPSED"]
    p_active: float
    times_seen: int = 0
    evidence: list[dict[str, Any]] = []         # most recent last
    interventions: list[str] = []               # modalities tried
    last_tested: str | None = None              # planet node, problem or exam item id
    recheck_queued: bool = False


class AttemptSummary(Strict):
    attempt_id: str
    problem_id: str
    ts: str
    passed: int
    total: int
    status: str
    top: str | None = None


class Learner(Strict):
    learner_id: str
    callsign: str | None = None
    misconceptions: dict[str, KnowledgeEntry]
    nodes: dict[str, dict[str, Any]] = {}       # node id -> {"status", "stars"} (W0 decision)
    attempts: list[AttemptSummary] = []


class ClassUpdate(Strict):
    cls: str = Field(alias="class")
    p_before: float
    p_after: float
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------- exam (03 §11, §11.4)

class ExamItem(Strict):
    """The `item` object served during an exam: a coding item or a trace item."""
    item_id: str
    kind: Literal["coding", "trace"]
    sector: str
    difficulty: int = 1
    prompt: str | None = None
    signature: str | None = None
    starter: str | None = None
    markers: list[str] = []
    sample_tests: list[PublicTest] = []
    code: str | None = None
    question: str | None = None
    options: list[str] = []


class Progress(Strict):
    k: int
    n: int


class SectorScore(Strict):
    sector: str
    items: list[str]
    passed: int
    rating_before: float
    rating_after: float


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    cls: str = Field(alias="class")
    status: Literal["NEW", "RELAPSED", "HELD", "UNCERTAIN", "NOT_TESTED"]
    p_active: float | None = None
    name: str | None = None
    subtitle: str | None = None
    state_after: str | None = None
    twin_set: str | None = None
    evidence: list[dict[str, Any]] = []


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    cls: str = Field(alias="class")
    problems: list[str]


class AdaptivityEntry(Strict):
    order: int
    item_id: str
    sector: str
    eig_bits: float
    reason: str
    outcome: str | None = None


class ExamReport(Strict):
    exam_id: str
    learner_id: str
    items_answered: int
    time_used_s: int
    sectors: list[SectorScore]
    findings: list[Finding]
    deferred_probes: list[PublicProbe] = []
    recommendations: list[Recommendation] = []
    adaptivity_log: list[AdaptivityEntry] = []


# ---------------------------------------------------------------- sentence reader (05 §5, §7)

class ReasonRow(Strict):
    """One line of ml/data/reasons.jsonl."""
    id: str
    label: str
    context_id: str
    code: str
    chosen: str | None = None
    text: str
    voice: str
    source: Literal["deepseek", "human", "sample"]
    split: Literal["train", "val", "test"]


# ---------------------------------------------------------------- API responses (03 §11, 05 §5)

class Envelope(Strict):
    model_version: str
    latency_ms: float


class HealthResponse(Envelope):
    ok: bool
    n_features: int
    db: str


class LearnerCreated(Envelope):
    learner_id: str


class RunResponse(Envelope):
    gate: Gate
    trace: Trace | None = None
    tests: Tests | None = None


class AttemptResponse(Envelope):
    attempt_id: str
    gate: Gate
    trace: Trace | None = None
    tests: Tests | None = None
    diagnosis: Diagnosis | None = None


class LabDiagnoseResponse(Envelope):
    gate: Gate
    trace: Trace | None = None
    tests: Tests | None = None
    diagnosis: Diagnosis | None = None


class ProbeAnswerResponse(Envelope):
    diagnosis: Diagnosis


class InterveneResponse(Intervention):
    model_version: str
    latency_ms: float


class ReassessResponse(Envelope):
    state: str
    p_active: float
    conditions: list[Condition]
    resolved_level: str | None = None
    next_item: dict[str, Any] | None = None


class LearnerResponse(Learner):
    model_version: str
    latency_ms: float


class ExamStartResponse(Envelope):
    exam_id: str
    item: ExamItem
    progress: Progress
    time_limit_s: int


class ExamAnswerResponse(Envelope):
    logged: bool
    next_item: ExamItem | None = None
    progress: Progress


class ExamReportResponse(Envelope):
    report: ExamReport


class QuizNextResponse(Envelope):
    item: PublicQuizItem | None = None
    reason: str = ""


class QuizAnswerResponse(Envelope):
    correct: bool
    correct_answer: str
    explain: str
    updates: list[ClassUpdate] = []
    ask_reason: bool = False


class DebugAnswerResponse(Envelope):
    correct: bool
    bug_lines: list[int]
    explain: str
    fix: dict[str, Any] | None = None
    updates: list[ClassUpdate] = []


class ReasonResponse(Envelope):
    status: Literal["matched", "correct_reasoning", "unsure"]
    top: list[TopClass] = []
    reader: Literal["biencoder", "frozen", "tfidf", "none"]
    updates: list[ClassUpdate] = []


# ---------------------------------------------------------------- requests (03 §11, 05 §5, 02 §8)

class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    ts: str | None = None
    payload: dict[str, Any] = {}


class LearnerCreate(Strict):
    callsign: str


class RunRequest(Strict):
    problem_id: str
    code: str
    sample_only: bool = False


class AttemptRequest(Strict):
    learner_id: str
    problem_id: str
    code: str
    events: list[EventEnvelope] = []
    code_item_id: str | None = None


class ProbeAnswerRequest(Strict):
    learner_id: str
    attempt_id: str
    probe_id: str
    answer: str


class InterveneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    learner_id: str
    attempt_id: str
    cls: str = Field(alias="class")
    modality: str | None = None


class ReassessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    learner_id: str
    cls: str = Field(alias="class")
    item_id: str
    item_type: str
    result: dict[str, Any]


class ProbeAnswerIn(Strict):
    probe_id: str
    answer: str


class LabDiagnoseRequest(Strict):
    problem_id: str
    code: str
    prediction: Any = None
    probe_answers: list[ProbeAnswerIn] = []


class ExamStartRequest(Strict):
    learner_id: str
    length: int = 10
    demo: bool = False


class ExamAnswerRequest(Strict):
    exam_id: str
    item_id: str
    code: str | None = None
    answer: str | None = None
    ms: int = 0
    sample_runs: int = 0
    skipped: bool = False


class ExamFinishRequest(Strict):
    exam_id: str


class ExamProbeRequest(Strict):
    exam_id: str
    probe_id: str
    answer: str


class QuizAnswerRequest(Strict):
    learner_id: str
    item_id: str
    answer: str
    ms: int = 0


class DebugAnswerRequest(Strict):
    learner_id: str
    item_id: str
    line: int | None = None     # null = "no bug"


class ReasonRef(Strict):
    kind: Literal["quiz", "attempt"]
    id: str


class ReasonRequest(Strict):
    learner_id: str
    ref: ReasonRef
    text: str


class LabReasonRequest(Strict):
    text: str
    code: str | None = None
