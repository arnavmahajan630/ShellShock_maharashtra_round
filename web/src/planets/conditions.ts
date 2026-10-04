import type { PlanetConfig } from "./types";

export const conditions: PlanetConfig = {
  slug: "conditions",
  name: "Aegis Grid",
  artSrc: "/images/new_planeta.png",
  artWidth: 1779,
  artHeight: 884,
  backButton: { top: 0.9, left: 0.8, width: 14.3, height: 5.7 },
  nodes: [
    { id: "BRIEFING", kind: "briefing", label: "Briefing", position: { top: 51.3, left: 20.5 } },
    { id: "WARMUP", kind: "warmup", label: "Predict Warm-up", position: { top: 54.7, left: 30.9 } },
    { id: "MMA-01", kind: "mission", label: "shield_trace", position: { top: 59.8, left: 41.6 }, problemId: "MMA-01" },
    { id: "P12", kind: "mission", label: "shield_mode", position: { top: 62.1, left: 52.0 }, problemId: "P12" },
    { id: "P16", kind: "mission", label: "in_range", position: { top: 66.0, left: 62.4 }, problemId: "P16" },
    { id: "P17", kind: "ghost", label: "max_of_three", position: { top: 67.7, left: 73.9 }, problemId: "P17" },
  ],
  briefingText: [
    "An `if` checks a condition and runs its block only when that condition is true.",
    "`==` compares two values. `=` stores a value — using it inside an `if` is a trap: `if (x = 1)` always runs.",
    "A `;` right after `if (...)` ends the statement there. `if (x < 5);` means the block below always runs, condition or not.",
  ],
  warmup: {
    code: "int code = 7;\nint open = 0;\nif (code = 42) {\n    open = 1;\n}",
    question: "What is open after this runs?",
    options: ["0", "1", "42"],
    correct: "1",
    explanation: "`if (code = 42)` assigns 42 to code — it doesn't compare. The block always runs.",
  },
  nextProblem: { "MMA-01": "P12", P11: "P12", P12: "P16", P16: "P17", P17: "MMA-01" },
};
