const BASE_URL = "http://localhost:8000";

export interface PredictItem {
  item_id: string;
  code: string;
  question: string;
  options: string[];
  correct: string;
}

export interface Problem {
  problem_id: string;
  name: string;
  planet: string | null;
  sector: string | null;
  difficulty: number;
  world: string;
  signature: string;
  starter: string;
  prompt: string;
  predict_item: PredictItem | null;
  markers: string[];
  sample_tests: { args: unknown[]; expect: Record<string, unknown> }[];
}

export interface Gate {
  code: string;
  message: string;
}

export interface TraceEvent {
  type: string;
  line: number;
  [key: string]: unknown;
}

export interface TraceStep {
  i: number;
  line: number;
  vars: Record<string, unknown>;
  events: TraceEvent[];
  effects: string[];
}

export interface Trace {
  status: string;
  returned: unknown;
  printed: string;
  steps: TraceStep[];
  events: TraceEvent[];
  effects_count: Record<string, number>;
  max_depth: number;
  truncated: boolean;
}

export interface TestResult {
  args: unknown[];
  expected: Record<string, unknown>;
  got: Record<string, unknown>;
  pass: boolean;
}

export interface Tests {
  passed: number;
  total: number;
  results: TestResult[];
}

export interface TopClass {
  id: string;
  p: number;
  name: string;
  subtitle: string;
  band: "Likely" | "Possible" | "Unsure";
}

export interface EvidenceItem {
  type: "CODE" | "RUN" | "YOU PREDICTED" | "PROBE" | "HISTORY" | "EXAM";
  text: string;
  line?: number;
}

export interface Diagnosis {
  status: "confident" | "ambiguous" | "novel" | "two_bug" | "correct" | "gate";
  posterior: Record<string, number>;
  top: TopClass[];
  twin_set: string | null;
  two_bug: boolean;
  novelty: { knn_dist: number; tau_d: number; p_max: number; tau_p: number; abstain: boolean };
  evidence: EvidenceItem[];
  next_probe: { probe_id: string; prompt: string; code: string; options: string[]; eig_bits: number } | null;
  model_version: string;
}

export interface AttemptResponse {
  attempt_id: string;
  gate: Gate;
  trace: Trace | null;
  tests: Tests | null;
  diagnosis: Diagnosis | null;
}

export interface Intervention {
  class: string;
  modality: "trace_timeline" | "memory_strip" | "value_meter";
  copy: string[];
  timeline: { step: number; line: number; vars: Record<string, unknown>; effect: string; flag: string }[];
  memory_strip: { array: string; values: unknown[]; reads: number[] } | null;
  value_meter: { exact: unknown; shown: unknown } | null;
  counterexample: { input: unknown; intended: Record<string, unknown>; yours: Record<string, unknown>; effect_diff: string } | null;
  fix: { kind: string; code: string; changed_lines: number[]; rule: string; verified: boolean } | null;
}

export interface ReassessResponse {
  state: string;
  p_active: number;
  conditions: { id: string; label: string; met: boolean; detail?: string }[];
  resolved_level: string;
  next_item: { item_id: string; item_type: string } | null;
}

export interface TrapItem {
  class: string;
  prompt: string;
  code: string;
  options: string[];
  correct: number;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export interface DsaExample {
  input: string;
  output: string;
  explanation: string;
}

export interface DsaPredictItem {
  item_id: string;
  code: string;
  question: string;
  call_stack: string[];
  stdout: string[];
  options: string[];
  correct: number;
}

export interface DsaTrial {
  problem_id: string;
  name: string;
  title: string;
  sector: string;
  difficulty: number;
  difficulty_label: string;
  prompt: string;
  signature: string;
  starter: string;
  examples: DsaExample[];
  constraints: string[];
  sample_tests: { args: unknown[]; expect: Record<string, unknown> }[];
  predict_item: DsaPredictItem;
  hints: string[];
}

export interface ExamStartResponse {
  exam_id: string;
  total_trials: number;
  current_index: number;
  item: DsaTrial;
  trials: DsaTrial[];
  time_limit_s: number;
}

export interface ExamRunResponse {
  tests: {
    passed: number;
    total: number;
    results: TestResult[];
  };
  trace: Trace;
  diagnosis: Record<string, unknown>;
}

export interface ExamAnswerResponse {
  logged: boolean;
  next_item: DsaTrial | null;
  finished: boolean;
  progress: { k: number; n: number };
  diagnosis: Record<string, unknown>;
}

export interface DebriefSector {
  sector: string;
  name: string;
  items: string[];
  passed: number;
  rating_before: number;
  rating_after: number;
}

export interface DebriefFinding {
  class: string;
  status: string;
  p_active: number;
  name: string;
  subtitle: string;
  belief: string;
  item_id: string;
  trial_title: string;
  evidence: { type: string; text: string }[];
}

export interface DebriefRecommendation {
  sector: string;
  title: string;
  description: string;
  route: string;
}

export interface DebriefReport {
  exam_id: string;
  items_total: number;
  items_passed: number;
  score_pct: number;
  sectors: DebriefSector[];
  findings: DebriefFinding[];
  recommendations: DebriefRecommendation[];
}

export const api = {
  getProblem: (problemId: string) => get<Problem>(`/problems/${problemId}`),
  attempt: (payload: { learner_id: string; problem_id: string; code: string; events?: unknown[] }) =>
    post<AttemptResponse>("/attempt", { events: [], ...payload }),
  intervene: (payload: { problem_id: string; code: string; class: string }) =>
    post<Intervention>("/intervene", payload),
  reassess: (payload: { learner_id: string; class: string; item_id: string; item_type: "trap" | "transfer_code"; result: unknown }) =>
    post<ReassessResponse>("/reassess", payload),
  trapItem: (classId: string) => get<TrapItem>(`/trap-items/${classId}`),
  probeAnswer: (payload: { learner_id: string; attempt_id: string; probe_id: string; answer: string; problem_id: string; code: string }) =>
    post<{ diagnosis: Diagnosis }>("/probe/answer", payload),
  startExam: (learnerId = "pilot") => post<ExamStartResponse>("/exam/start", { learner_id: learnerId }),
  runTrial: (payload: { item_id: string; code: string }) => post<ExamRunResponse>("/exam/run", payload),
  answerTrial: (payload: { exam_id: string; item_id: string; code: string; predict_answer?: number }) =>
    post<ExamAnswerResponse>("/exam/answer", payload),
  finishExam: (examId: string) => post<{ report: DebriefReport }>("/exam/finish", { exam_id: examId }),
  getExamReport: (examId: string) => get<{ report: DebriefReport }>(`/exam/${examId}/report`),
  getTrials: () => get<DsaTrial[]>("/exam/trials"),
};
