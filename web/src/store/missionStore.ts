import { create } from "zustand";
import type { AttemptResponse, Diagnosis, Intervention, Problem, ReassessResponse } from "../lib/api";

const LEARNER_ID = "demo-learner";
const PASSED_KEY = "relearn:planets:passed";

function loadPassed(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(PASSED_KEY) ?? "{}");
  } catch {
    return {};
  }
}

interface MissionState {
  learnerId: string;
  passedProblems: Record<string, boolean>;
  markPassed: (problemId: string) => void;

  problem: Problem | null;
  code: string;
  attempt: AttemptResponse | null;
  intervention: Intervention | null;
  trapPassed: boolean | null;
  transferAttempt: AttemptResponse | null;
  verdict: ReassessResponse | null;

  startMission: (problem: Problem) => void;
  setCode: (code: string) => void;
  setAttempt: (attempt: AttemptResponse) => void;
  updateDiagnosis: (diagnosis: Diagnosis) => void;
  setIntervention: (intervention: Intervention) => void;
  setTrapPassed: (passed: boolean) => void;
  setTransferAttempt: (attempt: AttemptResponse) => void;
  setVerdict: (verdict: ReassessResponse) => void;
  reset: () => void;
}

export const useMissionStore = create<MissionState>((set) => ({
  learnerId: LEARNER_ID,
  passedProblems: loadPassed(),
  markPassed: (problemId) =>
    set((state) => {
      const next = { ...state.passedProblems, [problemId]: true };
      localStorage.setItem(PASSED_KEY, JSON.stringify(next));
      return { passedProblems: next };
    }),

  problem: null,
  code: "",
  attempt: null,
  intervention: null,
  trapPassed: null,
  transferAttempt: null,
  verdict: null,

  startMission: (problem) =>
    set({
      problem,
      code: problem.starter,
      attempt: null,
      intervention: null,
      trapPassed: null,
      transferAttempt: null,
      verdict: null,
    }),
  setCode: (code) => set({ code }),
  setAttempt: (attempt) => set({ attempt }),
  updateDiagnosis: (diagnosis) =>
    set((state) => (state.attempt ? { attempt: { ...state.attempt, diagnosis } } : state)),
  setIntervention: (intervention) => set({ intervention }),
  setTrapPassed: (trapPassed) => set({ trapPassed }),
  setTransferAttempt: (transferAttempt) => set({ transferAttempt }),
  setVerdict: (verdict) => set({ verdict }),
  reset: () => set({ problem: null, code: "", attempt: null, intervention: null, trapPassed: null, transferAttempt: null, verdict: null }),
}));
