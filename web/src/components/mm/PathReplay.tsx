import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import type { FlowchartNode } from "./FlowchartView";

interface PathReplayProps {
  learnerPath: string[];
  correctPath: string[];
  divergenceNode?: string | null;
  divergenceReason?: string | null;
  nodes: FlowchartNode[];
  onClose?: () => void;
}

export default function PathReplay({
  learnerPath,
  correctPath,
  divergenceNode,
  divergenceReason,
  nodes,
  onClose,
}: PathReplayProps) {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const maxSteps = Math.max(learnerPath.length, correctPath.length);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      setActiveStep((prev) => {
        if (prev + 1 >= maxSteps) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1200);
    return () => clearInterval(timer);
  }, [isPlaying, maxSteps]);

  function restart() {
    setActiveStep(0);
    setIsPlaying(true);
  }

  return (
    <div className="rounded-lg border border-void-blue bg-void-blue/40 p-5">
      <div className="flex items-center justify-between border-b border-void-blue pb-3 mb-4">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-starlight animate-pulse" />
          <h3 className="font-display text-xs text-starlight uppercase tracking-wider">
            Execution Path Analysis & Divergence Replay
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={restart}
            className="px-4 py-1 font-ui text-xl rounded bg-deep-space border border-void-blue text-slate hover:border-starlight hover:text-starlight transition-colors cursor-pointer"
          >
            {isPlaying ? "Replaying..." : "Replay"}
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="px-2 py-1 font-ui text-xl text-slate hover:text-light cursor-pointer"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Divergence Alert Banner */}
      {divergenceNode && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-5 rounded border border-starlight/40 bg-starlight/10 p-3.5 font-ui text-xl text-starlight"
        >
          <div className="flex items-center gap-2 font-bold text-starlight mb-1">
            <span>⚠ PATH DIVERGENCE DETECTED AT NODE:</span>
            <span className="px-2 py-0.5 rounded bg-starlight/20 text-starlight border border-starlight/40">
              {divergenceNode}
            </span>
          </div>
          <p className="text-light">
            {divergenceReason ||
              "Your execution trace branched away from canonical execution here."}
          </p>
        </motion.div>
      )}

      {/* Side-by-side Path Step Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Learner Trace */}
        <div className="rounded border border-void-blue bg-deep-space/70 p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="font-display text-[10px] uppercase tracking-wider text-starlight">
              Your Traced Path
            </span>
            <span className="font-ui text-lg text-slate">
              {learnerPath.length} steps
            </span>
          </div>

          <div className="space-y-2">
            {learnerPath.map((nodeId, idx) => {
              const node = nodeMap.get(nodeId);
              const isDivergent = nodeId === divergenceNode;
              const isCurrent = idx === activeStep;
              const isPast = idx <= activeStep;

              return (
                <motion.div
                  key={`learner-${nodeId}-${idx}`}
                  animate={{
                    scale: isCurrent ? 1.02 : 1,
                    opacity: isPast ? 1 : 0.4,
                  }}
                  className={`flex items-start gap-3 rounded border p-2.5 font-ui text-xl transition-colors ${
                    isDivergent
                      ? "border-alert-red/60 bg-alert-red/15 text-alert-red"
                      : isCurrent
                      ? "border-starlight bg-starlight/15 text-starlight shadow-[0_0_10px_rgba(252,177,67,0.25)]"
                      : "border-void-blue bg-deep-space/60 text-slate"
                  }`}
                >
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                      isDivergent
                        ? "bg-alert-red text-deep-space"
                        : "bg-void-blue text-light"
                    }`}
                  >
                    {idx + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">{nodeId}</span>
                      <span className="text-sm uppercase text-slate">
                        ({node?.kind})
                      </span>
                      {isDivergent && (
                        <span className="ml-auto rounded bg-alert-red/20 px-2 py-0.5 text-sm text-alert-red font-bold border border-alert-red/40">
                          Diverged
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 line-clamp-1 text-base text-slate">
                      {node?.label.replace("\n", " ")}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Canonical / Expected Trace */}
        <div className="rounded border border-void-blue bg-deep-space/70 p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="font-display text-[10px] uppercase tracking-wider text-warp-cyan">
              Canonical Path
            </span>
            <span className="font-ui text-lg text-slate">
              {correctPath.length} steps
            </span>
          </div>

          <div className="space-y-2">
            {correctPath.map((nodeId, idx) => {
              const node = nodeMap.get(nodeId);
              const isCurrent = idx === activeStep;
              const isPast = idx <= activeStep;

              return (
                <motion.div
                  key={`correct-${nodeId}-${idx}`}
                  animate={{
                    scale: isCurrent ? 1.02 : 1,
                    opacity: isPast ? 1 : 0.4,
                  }}
                  className={`flex items-start gap-3 rounded border p-2.5 font-ui text-xl transition-colors ${
                    isCurrent
                      ? "border-warp-cyan bg-warp-cyan/15 text-warp-cyan shadow-[0_0_10px_rgba(45,231,252,0.25)]"
                      : "border-void-blue bg-deep-space/60 text-slate"
                  }`}
                >
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-warp-cyan/20 text-sm font-bold text-warp-cyan border border-warp-cyan/40">
                    {idx + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-warp-cyan">{nodeId}</span>
                      <span className="text-sm uppercase text-slate">
                        ({node?.kind})
                      </span>
                    </div>
                    <p className="mt-0.5 line-clamp-1 text-base text-slate">
                      {node?.label.replace("\n", " ")}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
