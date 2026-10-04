import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useParams } from "react-router-dom";
import ArtStage from "../components/ArtStage";
import { useMissionStore } from "../store/missionStore";
import { getPlanetConfig } from "../planets";
import type { PathNode } from "../planets";
import DeadEnd from "../components/DeadEnd";

export default function PlanetPathScreen() {
  const { planet: planetSlug } = useParams<{ planet: string }>();
  const navigate = useNavigate();
  const passed = useMissionStore((s) => s.passedProblems);
  const markPassed = useMissionStore((s) => s.markPassed);
  const [modal, setModal] = useState<"briefing" | "warmup" | null>(null);
  const [warmupAnswer, setWarmupAnswer] = useState<string | null>(null);

  const planet = getPlanetConfig(planetSlug);
  if (!planet) {
    return <DeadEnd message={`Unknown planet "${planetSlug}".`} />;
  }

  const { nodes, warmup } = planet;
  // Mission node ids (problem_ids) are globally unique; BRIEFING/WARMUP are not, so they're
  // namespaced per planet to avoid Loops reading Conditions' progress (or vice versa).
  const progressKey = (node: PathNode) => (node.problemId ? node.id : `${planetSlug}:${node.id}`);

  // Every node is open from the start — no node gates the next. "done" just reflects
  // this session's progress (checkmark), it never blocks access.
  function stateOf(node: PathNode): "done" | "available" {
    return passed[progressKey(node)] ? "done" : "available";
  }

  function openNode(node: PathNode) {
    if (node.kind === "briefing") setModal("briefing");
    else if (node.kind === "warmup") {
      setWarmupAnswer(null);
      setModal("warmup");
    } else if (node.problemId) {
      navigate(`/planet/${planetSlug}/mission/${node.problemId}`);
    }
  }

  return (
    <ArtStage src={planet.artSrc} width={planet.artWidth} height={planet.artHeight} alt={`${planet.name} — planet path`}>
      <motion.button
        type="button"
        onClick={() => navigate("/map")}
        whileHover={{ scale: 1.04, boxShadow: "0 0 20px 4px var(--color-warp-cyan)" }}
        whileTap={{ scale: 0.96 }}
        className="absolute cursor-pointer rounded outline-none focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
        style={{
          top: `${planet.backButton.top}%`,
          left: `${planet.backButton.left}%`,
          width: `${planet.backButton.width}%`,
          height: `${planet.backButton.height}%`,
        }}
        aria-label="Back to Galaxy"
      />
      {nodes.map((node, index) => {
        const state = stateOf(node);
        return (
          <motion.button
            key={node.id}
            type="button"
            onClick={() => openNode(node)}
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 + index * 0.08, duration: 0.35 }}
            whileHover={{ scale: 1.1, boxShadow: "0 0 24px 6px var(--color-warp-cyan)" }}
            whileTap={{ scale: 0.92 }}
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full outline-none cursor-pointer focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
            style={{ top: `${node.position.top}%`, left: `${node.position.left}%`, width: "8%", aspectRatio: "1 / 1" }}
            aria-label={`${node.label}${state === "done" ? " (done)" : ""}`}
          />
        );
      })}

      <AnimatePresence>
        {modal === "briefing" && (
          <Modal onClose={() => setModal(null)}>
            <h2 className="font-display text-sm text-starlight mb-4">Briefing</h2>
            <ul className="font-ui text-xl space-y-3 text-light">
              {planet.briefingText.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
            <button
              type="button"
              onClick={() => {
                markPassed(`${planetSlug}:BRIEFING`);
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
                  {warmupAnswer === warmup.correct ? "Right." : "Not quite."} {warmup.explanation}
                </p>
                <button
                  type="button"
                  onClick={() => {
                    markPassed(`${planetSlug}:WARMUP`);
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
