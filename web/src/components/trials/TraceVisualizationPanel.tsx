import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { DsaTrial } from "../../lib/api";

interface TraceVisualizationPanelProps {
  trial: DsaTrial;
  selectedOption: number | null;
  onSelectOption: (idx: number) => void;
  diagnosticFinding?: { class: string; text: string } | null;
}

export default function TraceVisualizationPanel({
  trial,
  selectedOption,
  onSelectOption,
  diagnosticFinding,
}: TraceVisualizationPanelProps) {
  const [activeTab, setActiveTab] = useState<"code" | "trace" | "visualization" | "hints">("trace");
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [revealedHints, setRevealedHints] = useState<number[]>([0]);

  const predict = trial.predict_item;
  const callStack = predict?.call_stack || ["main()", `${trial.name}()`];
  const stdoutLines = predict?.stdout || [];
  const maxSteps = Math.max(callStack.length, stdoutLines.length);

  // Playback timer
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setStepIndex((prev) => {
        if (prev >= maxSteps - 1) {
          setIsPlaying(false);
          return maxSteps - 1;
        }
        return prev + 1;
      });
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, maxSteps, playbackSpeed]);

  function handlePlayToggle() {
    if (stepIndex >= maxSteps - 1) {
      setStepIndex(0);
    }
    setIsPlaying(!isPlaying);
  }

  return (
    <aside className="flex flex-col h-full bg-void/80 border border-warp-cyan/20 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Tab Bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-black/60 border-b border-warp-cyan/20 select-none">
        <div className="flex items-center gap-1.5">
          {(["trace", "code", "visualization", "hints"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 rounded text-xs font-mono font-bold capitalize transition-all cursor-pointer ${
                activeTab === tab
                  ? "bg-gradient-to-r from-warp-cyan/30 to-blue-600/30 border border-warp-cyan text-warp-cyan shadow-[0_0_12px_rgba(0,240,255,0.4)]"
                  : "text-white/50 hover:text-white hover:bg-white/5 border border-transparent"
              }`}
            >
              {tab === "hints" && diagnosticFinding ? "⚡ Hints & Diagnosis" : tab}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Body Content */}
      <div className="flex-1 flex flex-col p-4 overflow-y-auto scrollbar-thin scrollbar-thumb-warp-cyan/30">
        {/* TRACE TAB (inspiration.png) */}
        {activeTab === "trace" && predict && (
          <div className="flex flex-col h-full space-y-4">
            {/* Header / Subtitle */}
            <div>
              <h3 className="font-display text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-warp-cyan shadow-[0_0_8px_#00f0ff]" />
                Trace the Output
              </h3>
              <p className="text-xs font-ui text-warp-cyan/90 mt-0.5">{predict.question}</p>
            </div>

            {/* Split Visualization: Code Snippet & Call Stack / Output */}
            <div className="grid grid-cols-2 gap-3 bg-black/50 p-3 rounded-xl border border-white/10">
              {/* Code Snippet */}
              <div className="flex flex-col space-y-1">
                <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">
                  Snippet
                </span>
                <pre className="p-2.5 rounded bg-black/70 border border-white/5 text-[11px] font-mono text-white/90 overflow-x-auto leading-relaxed">
                  {predict.code}
                </pre>
              </div>

              {/* Call Stack & Stdout */}
              <div className="flex flex-col space-y-2">
                {/* Call Stack Frames */}
                <div>
                  <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">
                    Call Stack
                  </span>
                  <div className="space-y-1 mt-1">
                    {callStack.slice(0, stepIndex + 1).map((frame, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="px-2 py-1 rounded bg-gradient-to-r from-blue-900/60 to-cyan-950/60 border border-warp-cyan/40 text-[11px] font-mono text-warp-cyan flex items-center justify-between"
                      >
                        <span>{frame}</span>
                        <span className="text-[9px] text-white/40">d={idx}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Stdout Feed */}
                <div>
                  <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">
                    Output (stdout)
                  </span>
                  <div className="p-2 rounded bg-black/70 border border-white/5 text-[11px] font-mono text-mint-success min-h-[50px] space-y-0.5">
                    {stdoutLines.slice(0, stepIndex + 1).map((line, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-1.5"
                      >
                        <span className="text-white/30 text-[9px]">›</span>
                        <span>{line}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Playback Controls Bar */}
            <div className="flex items-center justify-between px-3 py-2 bg-black/60 border border-white/10 rounded-lg">
              <div className="flex items-center gap-2">
                {/* Step Back */}
                <button
                  type="button"
                  onClick={() => setStepIndex((p) => Math.max(0, p - 1))}
                  disabled={stepIndex === 0}
                  className="w-7 h-7 rounded bg-white/5 hover:bg-white/10 disabled:opacity-30 text-white text-xs flex items-center justify-center cursor-pointer"
                  title="Step Back"
                >
                  ⏮
                </button>

                {/* Play / Pause */}
                <button
                  type="button"
                  onClick={handlePlayToggle}
                  className="px-3 py-1 rounded bg-warp-cyan/20 border border-warp-cyan text-warp-cyan hover:bg-warp-cyan hover:text-black font-ui text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_10px_rgba(0,240,255,0.3)]"
                >
                  <span>{isPlaying ? "⏸ Pause" : "▶ Play"}</span>
                </button>

                {/* Step Forward */}
                <button
                  type="button"
                  onClick={() => setStepIndex((p) => Math.min(maxSteps - 1, p + 1))}
                  disabled={stepIndex >= maxSteps - 1}
                  className="w-7 h-7 rounded bg-white/5 hover:bg-white/10 disabled:opacity-30 text-white text-xs flex items-center justify-center cursor-pointer"
                  title="Step Forward"
                >
                  ⏭
                </button>
              </div>

              {/* Speed Selector */}
              <div className="flex items-center gap-1 text-[11px] font-mono text-white/60">
                <span>Speed:</span>
                <select
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                  className="bg-black border border-white/20 rounded px-1.5 py-0.5 text-warp-cyan text-xs outline-none cursor-pointer"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={2}>2x</option>
                </select>
              </div>
            </div>

            {/* Multiple Choice Prediction Options (inspiration.png layout) */}
            <div className="space-y-2 mt-auto">
              <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">
                Select Predicted Behavior:
              </span>
              <div className="grid grid-cols-2 gap-2">
                {predict.options.map((opt, idx) => {
                  const isSelected = selectedOption === idx;
                  const isCorrect = idx === predict.correct;
                  const showFeedback = selectedOption !== null;

                  return (
                    <motion.button
                      key={idx}
                      type="button"
                      onClick={() => onSelectOption(idx)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`p-2.5 rounded-lg border text-left text-xs font-mono transition-all cursor-pointer flex items-start gap-2 ${
                        showFeedback
                          ? isSelected
                            ? isCorrect
                              ? "bg-emerald-950/40 border-emerald-500 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.4)]"
                              : "bg-rose-950/40 border-rose-500 text-rose-300 shadow-[0_0_15px_rgba(244,63,94,0.4)]"
                            : isCorrect
                            ? "bg-emerald-950/20 border-emerald-500/50 text-emerald-400"
                            : "bg-black/30 border-white/5 text-white/40"
                          : isSelected
                          ? "bg-warp-cyan/20 border-warp-cyan text-warp-cyan shadow-[0_0_12px_rgba(0,240,255,0.4)]"
                          : "bg-black/50 border-white/10 text-white/80 hover:border-warp-cyan/40 hover:bg-white/5"
                      }`}
                    >
                      <span className="font-bold text-warp-cyan">{String.fromCharCode(65 + idx)}</span>
                      <span className="line-clamp-2">{opt}</span>
                    </motion.button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* CODE & REASONING TAB */}
        {activeTab === "code" && (
          <div className="space-y-4 text-xs font-ui text-white/80 leading-relaxed">
            <h3 className="font-display text-sm font-bold text-white tracking-wide">
              {trial.title} · Signature & Contracts
            </h3>
            <div className="p-3 rounded-lg bg-black/60 border border-white/10 font-mono text-warp-cyan">
              {trial.signature}
            </div>
            <p>
              In C, arrays are passed by reference as pointers to the first element. Scalar arguments
              are passed by value.
            </p>
            <div className="p-3 rounded-lg bg-deep-space border border-warp-cyan/20 space-y-1.5">
              <span className="text-warp-cyan font-bold block">Invariants to Maintain:</span>
              <ul className="list-disc list-inside space-y-1 text-white/70 font-mono">
                <li>Valid array indices: 0 to n - 1</li>
                <li>Avoid mutating inputs unless in-place modification is specified</li>
                <li>Guarantee that all control paths return an explicit value</li>
              </ul>
            </div>
          </div>
        )}

        {/* VISUALIZATION TAB */}
        {activeTab === "visualization" && (
          <div className="space-y-4 text-xs font-mono text-white/80">
            <h3 className="font-display text-sm font-bold text-white tracking-wide">
              {trial.sector.toUpperCase()} Visualizer
            </h3>
            <div className="p-4 rounded-xl bg-black/60 border border-warp-cyan/30 flex flex-col items-center justify-center min-h-[180px] space-y-3">
              <span className="text-white/40 text-[11px]">Memory Diagram</span>
              <div className="flex items-center gap-1.5">
                {[0, 1, 2, 3].map((cell) => (
                  <div
                    key={cell}
                    className="w-12 h-12 rounded border border-warp-cyan/60 bg-warp-cyan/10 flex flex-col items-center justify-center text-xs"
                  >
                    <span className="text-white font-bold">{cell * 2 + 1}</span>
                    <span className="text-[9px] text-white/40">[{cell}]</span>
                  </div>
                ))}
              </div>
              <span className="text-[10px] text-warp-cyan tracking-wider">
                0-indexed sequential buffer
              </span>
            </div>
          </div>
        )}

        {/* HINTS TAB */}
        {activeTab === "hints" && (
          <div className="space-y-4 text-xs font-ui">
            {/* Diagnostic Alert if active */}
            {diagnosticFinding && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/60 text-rose-200 space-y-1.5 shadow-[0_0_15px_rgba(244,63,94,0.3)]"
              >
                <div className="font-bold font-mono text-rose-300 flex items-center gap-1.5">
                  <span>⚡ Misconception Detected:</span>
                  <span className="underline">{diagnosticFinding.class}</span>
                </div>
                <p className="text-xs leading-relaxed">{diagnosticFinding.text}</p>
              </motion.div>
            )}

            {/* Progressive Hints */}
            <div className="space-y-2.5">
              <h3 className="font-display text-xs font-bold text-white uppercase tracking-wider">
                Progressive Guidance
              </h3>
              {trial.hints.map((hint, idx) => {
                const isUnlocked = revealedHints.includes(idx);
                return (
                  <div
                    key={idx}
                    className="p-3 rounded-lg border border-white/10 bg-black/40 text-xs font-mono transition-all"
                  >
                    {isUnlocked ? (
                      <div>
                        <span className="text-warp-cyan font-bold">Hint {idx + 1}: </span>
                        <span className="text-white/90">{hint}</span>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setRevealedHints((prev) => [...prev, idx])}
                        className="text-white/50 hover:text-warp-cyan flex items-center gap-1.5 cursor-pointer"
                      >
                        <span>🔒 Unlock Hint {idx + 1}</span>
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
