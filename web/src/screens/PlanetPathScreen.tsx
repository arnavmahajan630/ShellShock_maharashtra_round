import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ArtStage from "../components/ArtStage";
import { useMissionStore } from "../store/missionStore";

type NodeKind = "briefing" | "warmup" | "mission" | "ghost";

interface PathNode {
  id: string;
  kind: NodeKind;
  label: string;
  position: { top: number; left: number };
  problemId?: string;
}

const NODES: PathNode[] = [
  { id: "BRIEFING", kind: "briefing", label: "Briefing", position: { top: 51.3, left: 20.5 } },
  { id: "WARMUP", kind: "warmup", label: "Predict Warm-up", position: { top: 54.7, left: 30.9 } },
  { id: "P11", kind: "mission", label: "door_open", position: { top: 59.8, left: 41.6 }, problemId: "P11" },
  { id: "P12", kind: "mission", label: "shield_mode", position: { top: 62.1, left: 52.0 }, problemId: "P12" },
  { id: "P16", kind: "mission", label: "in_range", position: { top: 66.0, left: 62.4 }, problemId: "P16" },
  { id: "P17", kind: "ghost", label: "max_of_three", position: { top: 67.7, left: 73.9 }, problemId: "P17" },
];

const BRIEFING_TEXT = [
  "An `if` checks a condition and runs its block only when that condition is true.",
  "`==` compares two values. `=` stores a value — using it inside an `if` is a trap: `if (x = 1)` always runs.",
  "A `;` right after `if (...)` ends the statement there. `if (x < 5);` means the block below always runs, condition or not.",
];

export default function PlanetPathScreen() {
  const navigate = useNavigate();
  const passed = useMissionStore((s) => s.passedProblems);
  const markPassed = useMissionStore((s) => s.markPassed);
  const [modal, setModal] = useState<"briefing" | "warmup" | null>(null);
  const [warmupAnswer, setWarmupAnswer] = useState<string | null>(null);

  function stateOf(node: PathNode, index: number): "done" | "current" | "available" | "locked" {
    if (passed[node.id]) return "done";
    const prevDone = index === 0 || passed[NODES[index - 1].id];
    return prevDone ? (index === 0 ? "current" : "available") : "locked";
  }

  function openNode(node: PathNode, state: string) {
    if (state === "locked") return;
    if (node.kind === "briefing") setModal("briefing");
    else if (node.kind === "warmup") {
      setWarmupAnswer(null);
      setModal("warmup");
    } else if (node.problemId) {
      navigate(`/planet/conditions/mission/${node.problemId}`);
    }
  }

  const warmup = {
    code: "int code = 7;\nint open = 0;\nif (code = 42) {\n    open = 1;\n}",
    question: "What is open after this runs?",
    options: ["0", "1", "42"],
    correct: "1",
  };

  return (
    <ArtStage src="/images/landing_planetA.png" width={1779} height={884} alt="Aegis Grid — planet path">
      <motion.button
        type="button"
        onClick={() => navigate("/map")}
        whileHover={{ scale: 1.04, boxShadow: "0 0 20px 4px var(--color-warp-cyan)" }}
        whileTap={{ scale: 0.96 }}
        className="absolute cursor-pointer rounded outline-none focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
        style={{ top: "0.9%", left: "0.8%", width: "14.3%", height: "5.7%" }}
        aria-label="Back to Galaxy"
      />
      {NODES.map((node, index) => {
        const state = stateOf(node, index);
        return (
          <motion.button
            key={node.id}
            type="button"
            disabled={state === "locked"}
            onClick={() => openNode(node, state)}
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 + index * 0.08, duration: 0.35 }}
            whileHover={state !== "locked" ? { scale: 1.1, boxShadow: "0 0 24px 6px var(--color-warp-cyan)" } : undefined}
            whileTap={state !== "locked" ? { scale: 0.92 } : undefined}
            className={
              "absolute -translate-x-1/2 -translate-y-1/2 rounded-full outline-none" +
              (state === "locked" ? " cursor-not-allowed" : " cursor-pointer focus-visible:ring-4 focus-visible:ring-warp-cyan/70")
            }
            style={{ top: `${node.position.top}%`, left: `${node.position.left}%`, width: "8%", aspectRatio: "1 / 1" }}
            aria-label={`${node.label}${state === "locked" ? " (locked)" : ""}`}
          />
        );
      })}

      <AnimatePresence>
        {modal === "briefing" && (
          <Modal onClose={() => setModal(null)}>
            <h2 className="font-display text-sm text-starlight mb-4">Briefing</h2>
            <ul className="font-ui text-xl space-y-3 text-light">
              {BRIEFING_TEXT.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
            <button
              type="button"
              onClick={() => {
                markPassed("BRIEFING");
                setModal(null);
              }}
              className="mt-6 font-ui text-xl px-6 py-2 rounded bg-starlight text-deep-space hover:brightness-110 active:scale-95"
            >
              Got it
            </button>
          </Modal>
        )}

        {modal === "warmup" && (
          <Modal onClose={() => setModal(null)}>
            <h2 className="font-display text-sm text-warp-cyan mb-4">Predict Warm-up</h2>
            <p className="font-ui text-lg text-slate mb-2">This code has a bug. Predict what it actually does — not what it was meant to do.</p>
            <pre className="font-mono text-base bg-deep-space/60 border border-void-blue rounded p-3 text-light whitespace-pre-wrap">{warmup.code}</pre>
            <p className="font-ui text-xl mt-4 text-light">{warmup.question}</p>
            <div className="flex gap-3 mt-3">
              {warmup.options.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  disabled={warmupAnswer !== null}
                  onClick={() => setWarmupAnswer(opt)}
                  className={
                    "font-ui text-xl px-4 py-2 rounded border " +
                    (warmupAnswer === null
                      ? "border-slate text-light hover:border-warp-cyan"
                      : opt === warmup.correct
                        ? "border-mint-success text-mint-success"
                        : opt === warmupAnswer
                          ? "border-alert-red text-alert-red"
                          : "border-slate text-slate")
                  }
                >
                  {opt}
                </button>
              ))}
            </div>
            {warmupAnswer !== null && (
              <div className="mt-4">
                <p className="font-ui text-lg text-light">
                  {warmupAnswer === warmup.correct ? "Right." : "Not quite."} `if (code = 42)` assigns 42 to code — it doesn't compare. The block always runs.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    markPassed("WARMUP");
                    setModal(null);
                  }}
                  className="mt-4 font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
                >
                  Continue
                </button>
              </div>
            )}
          </Modal>
        )}
      </AnimatePresence>
    </ArtStage>
  );
}

function Modal({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-deep-space/80 p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="max-w-lg w-full bg-void-blue border border-warp-cyan/40 rounded-lg p-6"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 16 }}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}
