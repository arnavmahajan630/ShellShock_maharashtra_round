import type { PlanetConfig } from "./types";

export const loops: PlanetConfig = {
  slug: "loops",
  name: "Miner's Belt",
  artSrc: "/images/new_planetb.png",
  artWidth: 1779,
  artHeight: 884,
  backButton: { top: 0.9, left: 0.8, width: 14.3, height: 5.7 },
  nodes: [
    { id: "BRIEFING", kind: "briefing", label: "Briefing", position: { top: 48.5, left: 19.4 } },
    { id: "WARMUP", kind: "warmup", label: "Predict Warm-up", position: { top: 51.3, left: 29.2 } },
    { id: "MMB-01", kind: "mission", label: "astronaut_search", position: { top: 55.8, left: 39.6 }, problemId: "MMB-01" },
    { id: "P03", kind: "mission", label: "total_energy", position: { top: 58.7, left: 51.4 }, problemId: "P03" },
    { id: "P05", kind: "mission", label: "charge_steps", position: { top: 60.9, left: 63.0 }, problemId: "P05" },
    { id: "P06", kind: "ghost", label: "power_up", position: { top: 62.1, left: 74.2 }, problemId: "P06" },
  ],
  briefingText: [
    "A loop's stopping condition decides exactly how many times it runs — `i <= n` and `i < n` (from 0) differ by one pass.",
    "A loop only ends when something inside it moves the loop variable toward the exit. If nothing updates it, the loop never stops.",
    "An accumulator (a running total) must grow with `+=`. Plain `=` inside the loop replaces the total instead of adding to it.",
  ],
  warmup: {
    code: "int cells[3] = {2, 4, 6};\nint total = 0;\nfor (int i = 0; i <= 3; i++) {\n    total += cells[i];\n}",
    question: "How many cells are read?",
    options: ["3", "4", "2"],
    correct: "4",
    explanation: "The loop condition uses `<=` over an array of 3 cells, so it reads index 3 too — one cell past the end. You'll meet this exact ambiguity for real in total_energy: code alone can't tell whether the loop bound is wrong, or whether cells are believed to start at index 1.",
  },
  nextProblem: { "MMB-01": "P03", P01: "P03", P03: "P05", P05: "P06", P06: "MMB-01" },
};
