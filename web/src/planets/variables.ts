import type { PlanetConfig } from "./types";

export const variables: PlanetConfig = {
  slug: "variables",
  name: "Planet Neo Core",
  artSrc: "/images/landing_planetd.png",
  artWidth: 1886,
  artHeight: 834,
  backButton: { top: 1.2, left: 1.1, width: 14.7, height: 7.0 },
  nodes: [
    { id: "BRIEFING", kind: "briefing", label: "System State · Briefing", position: { top: 58.4, left: 25.7 } },
    { id: "WARMUP", kind: "warmup", label: "Synchronization · Predict Warm-up", position: { top: 63.5, left: 34.3 } },
    { id: "P13", kind: "mission", label: "Distributed Coordination · sync_ratio", position: { top: 68.2, left: 43.8 }, problemId: "P13" },
    { id: "P14", kind: "mission", label: "Consensus Protocols · signal_diff", position: { top: 68.2, left: 53.9 }, problemId: "P14" },
    { id: "P18", kind: "mission", label: "Fault Tolerance · state_balance", position: { top: 68.3, left: 63.7 }, problemId: "P18" },
    { id: "P19", kind: "ghost", label: "Scaling Coordination · coordinate_nodes", position: { top: 64.6, left: 73.3 }, problemId: "P19" },
  ],
  briefingText: [
    "Variables in C must be given an initial value before being read. An uninitialized local variable holds garbage memory, not 0 (Static Signal).",
    "Dividing two integers in C discards any fractional part before storing the result (Fraction Shear). Cast at least one side to (float): `(float)a / b`.",
    "Single `=` assigns a value; `==` compares. Writing `if (state = 1)` always evaluates to true and overwrites state (Sensor Overwrite).",
    "`printf` displays output to the console but does not hand a value back to the caller. A function must use `return` to pass state back.",
  ],
  warmup: {
    code: "int active;\nint step = 5;\nint total = active + step;",
    question: "What is stored in total after this runs?",
    options: ["5", "Unpredictable garbage value", "0"],
    correct: "Unpredictable garbage value",
    explanation:
      "active was declared without an initial value. In C, local variables do not default to 0; they hold whatever raw bytes were already in that memory slot.",
  },
  nextProblem: { P13: "P14", P14: "P18", P18: "P19", P19: "P13" },
};
