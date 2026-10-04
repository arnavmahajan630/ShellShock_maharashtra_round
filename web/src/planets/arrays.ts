import type { PlanetConfig } from "./types";

export const arrays: PlanetConfig = {
  slug: "arrays",
  name: "Lost Fleet",
  artSrc: "/images/landing_planetc.png",
  artWidth: 1918,
  artHeight: 820,
  backButton: { top: 1.5, left: 0.9, width: 14.7, height: 5.7 },
  nodes: [
    { id: "BRIEFING", kind: "briefing", label: "Fleet Overview · Briefing", position: { top: 70.7, left: 26.6 } },
    { id: "WARMUP", kind: "warmup", label: "Array Reconstruction · Predict Warm-up", position: { top: 75.6, left: 35.2 } },
    { id: "P08", kind: "mission", label: "Signal Routing · last_beacon", position: { top: 79.3, left: 44.1 }, problemId: "P08" },
    { id: "P07", kind: "mission", label: "Redundancy & Failover · avg_fuel", position: { top: 79.3, left: 53.5 }, problemId: "P07" },
    { id: "P10", kind: "mission", label: "Fleet Synchronization · sum_first_k", position: { top: 79.3, left: 63.5 }, problemId: "P10" },
    { id: "P09", kind: "ghost", label: "Full Fleet Restoration · max_shield", position: { top: 73.2, left: 73.1 }, problemId: "P09" },
  ],
  briefingText: [
    "Arrays in C are 0-indexed: an array of n items has valid indices from 0 up to n - 1.",
    "Accessing index n reads or writes past the end of the array (Index Origin Fault), causing silent memory corruption or garbage values.",
    "Dividing two integers in C (e.g. sum / n) performs integer division, discarding decimals. Cast at least one operand to (float) when computing averages.",
  ],
  warmup: {
    code: "int ids[4] = {10, 20, 30, 40};\nint target = ids[4];",
    question: "What is stored in target after this runs?",
    options: ["40", "Out-of-bounds memory / garbage", "30"],
    correct: "Out-of-bounds memory / garbage",
    explanation:
      "An array of 4 elements only has indices 0, 1, 2, and 3. Accessing ids[4] reads past the end of the array into unallocated memory.",
  },
  nextProblem: { P08: "P07", P07: "P10", P10: "P09", P09: "P08" },
};
