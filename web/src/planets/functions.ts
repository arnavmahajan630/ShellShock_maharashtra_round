import type { PlanetConfig } from "./types";

export const functions: PlanetConfig = {
  slug: "functions",
  name: "Module Deck",
  artSrc: "/images/landing_planete.png",
  artWidth: 1870,
  artHeight: 841,
  backButton: { top: 1.2, left: 1.1, width: 12.3, height: 5.9 },
  nodes: [
    { id: "P20", kind: "mission", label: "Modular Thinking · clamp_range", position: { top: 67.0, left: 28.3 }, problemId: "P20" },
    { id: "P21", kind: "mission", label: "Interfaces & Contracts · boost_shield", position: { top: 70.0, left: 36.8 }, problemId: "P21" },
    { id: "P22", kind: "mission", label: "Composition · compose_pipeline", position: { top: 73.0, left: 45.8 }, problemId: "P22" },
    { id: "WARMUP", kind: "warmup", label: "Dependency Management · Protocol Quiz", position: { top: 73.0, left: 55.0 } },
    { id: "BRIEFING", kind: "briefing", label: "Modular Debugging · Architecture Briefing", position: { top: 70.0, left: 63.8 } },
    { id: "P23", kind: "ghost", label: "Real-world Modular Design · integrate_subsystem", position: { top: 67.0, left: 73.5 }, problemId: "P23" },
  ],
  briefingText: [
    "Functions in C receive scalar arguments by value — they get private copies, not the caller's variables (Copy Module / Pass by Value).",
    "To hand computed results back to the caller, a function must use `return`. `printf` displays text on screen but leaves the caller with garbage (Silent Messenger).",
    "Modular software combines small, well-tested helper functions into larger pipelines. Keep each helper focused on a single responsibility.",
    "Verify return values across every execution branch. Omitting a return statement on certain conditions causes undefined behavior.",
  ],
  warmup: {
    code: "void add_power(int shield, int boost) {\n    shield += boost;\n}\nint main() {\n    int s = 40;\n    add_power(s, 30);\n    return s;\n}",
    question: "What does this program return?",
    options: ["40 — C parameters are passed by value, so s is unchanged", "70", "30"],
    correct: "40 — C parameters are passed by value, so s is unchanged",
    explanation:
      "add_power receives a copy of s. Mutating that copy inside the function has no effect on the original variable s in main(). To change the value, the function should return the new total.",
  },
  nextProblem: { P20: "P21", P21: "P22", P22: "P23", P23: "P20" },
};
