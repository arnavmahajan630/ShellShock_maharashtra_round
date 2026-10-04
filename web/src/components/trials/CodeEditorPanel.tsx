import { useState } from "react";
import { motion } from "framer-motion";
import CodeMirror from "@uiw/react-codemirror";
import { cpp } from "@codemirror/lang-cpp";
import type { DsaTrial, TestResult } from "../../lib/api";

interface CodeEditorPanelProps {
  trial: DsaTrial;
  code: string;
  onCodeChange: (code: string) => void;
  onRunSamples: () => void;
  onSubmit: () => void;
  onSkipClick: () => void;
  isRunning: boolean;
  isSubmitting: boolean;
  testResults: TestResult[] | null;
  canFinishEarly: boolean;
  onFinishEarly: () => void;
}

export default function CodeEditorPanel({
  trial,
  code,
  onCodeChange,
  onRunSamples,
  onSubmit,
  onSkipClick,
  isRunning,
  isSubmitting,
  testResults,
  canFinishEarly,
  onFinishEarly,
}: CodeEditorPanelProps) {
  const [activeTab] = useState("main.c");
  const [isExpanded, setIsExpanded] = useState(false);

  function resetCode() {
    onCodeChange(trial.starter);
  }

  return (
    <div className="flex flex-col h-full bg-void/80 border border-warp-cyan/20 rounded-xl overflow-hidden shadow-2xl">
      {/* Editor Tab Bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-black/60 border-b border-warp-cyan/20 select-none">
        <div className="flex items-center gap-1">
          <div className="flex items-center gap-2 px-3 py-1 rounded-t-md bg-deep-space border-t-2 border-warp-cyan text-xs font-mono text-white shadow-[0_0_10px_rgba(0,240,255,0.2)]">
            <span className="text-warp-cyan font-bold">C</span>
            <span>{activeTab}</span>
            <span className="text-white/40 text-[10px]">✕</span>
          </div>
          <button
            type="button"
            className="px-2 py-1 text-white/40 hover:text-white text-xs font-mono rounded hover:bg-white/5"
            title="New Tab (Disabled)"
          >
            +
          </button>
        </div>

        <div className="flex items-center gap-2">
          {/* Reset Code */}
          <button
            type="button"
            onClick={resetCode}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-white/5 hover:bg-white/10 text-white/70 hover:text-white text-xs font-mono transition-colors cursor-pointer"
            title="Reset code to template"
          >
            <span>↺</span>
            <span>Reset</span>
          </button>

          {/* Expand Toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 text-white/50 hover:text-white text-xs rounded hover:bg-white/5 cursor-pointer"
            title={isExpanded ? "Collapse Editor" : "Expand Editor"}
          >
            ⛶
          </button>
        </div>
      </div>

      {/* CodeMirror Editor Area */}
      <div className={`overflow-auto border-b border-warp-cyan/20 ${isExpanded ? "h-[450px]" : "h-[290px]"}`}>
        <CodeMirror
          value={code}
          height="100%"
          theme="dark"
          extensions={[cpp()]}
          onChange={onCodeChange}
          basicSetup={{
            lineNumbers: true,
            foldGutter: true,
            highlightActiveLineGutter: true,
            bracketMatching: true,
            indentOnInput: true,
          }}
          className="text-xs font-mono"
        />
      </div>

      {/* Sample Test Cases Section */}
      <div className="flex-1 flex flex-col p-3 overflow-y-auto bg-black/40">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-white/70 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-warp-cyan" />
            Sample Test Cases
          </span>
          {testResults && (
            <span className="text-[11px] font-mono text-warp-cyan">
              {testResults.filter((r) => r.pass).length} / {testResults.length} Passed
            </span>
          )}
        </div>

        {/* Test Cases Cards */}
        <div className="space-y-2 flex-1">
          {trial.sample_tests.map((test, idx) => {
            const result = testResults && testResults[idx];
            const hasRun = !!result;
            const isPassed = result?.pass;

            return (
              <div
                key={idx}
                className={`p-2.5 rounded-lg border text-xs font-mono flex items-center justify-between transition-colors ${
                  hasRun
                    ? isPassed
                      ? "bg-emerald-950/20 border-emerald-500/40 text-emerald-300"
                      : "bg-rose-950/20 border-rose-500/40 text-rose-300"
                    : "bg-white/5 border-white/10 text-white/80"
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded bg-black/50 text-[10px] text-white/50 font-bold">
                      Case {idx + 1}
                    </span>
                    <span className="text-white/60">Args:</span>
                    <span className="text-white truncate max-w-[200px]">
                      {JSON.stringify(test.args)}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 text-[11px]">
                    <div>
                      <span className="text-white/40">Expected: </span>
                      <span className="text-mint-success font-semibold">
                        {JSON.stringify(test.expect.returned ?? test.expect.array0 ?? test.expect)}
                      </span>
                    </div>

                    {hasRun && (
                      <div>
                        <span className="text-white/40">Got: </span>
                        <span className={isPassed ? "text-mint-success" : "text-alert-red font-bold"}>
                          {JSON.stringify(result.got.returned ?? result.got.array0 ?? result.got)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Status Icon */}
                <div>
                  {hasRun ? (
                    isPassed ? (
                      <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-400 flex items-center justify-center text-xs">
                        ✓
                      </span>
                    ) : (
                      <span className="w-5 h-5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-400 flex items-center justify-center text-xs">
                        ✕
                      </span>
                    )
                  ) : (
                    <span className="w-5 h-5 rounded-full bg-white/5 text-white/30 border border-white/20 flex items-center justify-center text-xs">
                      •
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Action Controls Footer */}
        <div className="flex items-center justify-between pt-3 mt-2 border-t border-white/10">
          {/* Skip / Finish Early */}
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={onSkipClick}
              className="text-xs font-ui text-white/50 hover:text-white underline underline-offset-4 cursor-pointer"
            >
              Skip this trial
            </button>
            {canFinishEarly && (
              <button
                type="button"
                onClick={onFinishEarly}
                disabled={isRunning || isSubmitting}
                title="Compile a report for the trial(s) you've already submitted"
                className="text-xs font-ui text-mint-success hover:text-white underline underline-offset-4 cursor-pointer disabled:opacity-50"
              >
                Finish &amp; view report →
              </button>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            {/* Run Samples */}
            <motion.button
              type="button"
              onClick={onRunSamples}
              disabled={isRunning || isSubmitting}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-warp-cyan bg-deep-space/80 hover:bg-warp-cyan/20 text-warp-cyan font-ui text-xs font-bold tracking-wider transition-all disabled:opacity-50 cursor-pointer shadow-[0_0_15px_rgba(0,240,255,0.2)]"
            >
              <span>▶</span>
              <span>{isRunning ? "Running..." : "Run Samples"}</span>
            </motion.button>

            {/* Submit Answer */}
            <motion.button
              type="button"
              onClick={onSubmit}
              disabled={isRunning || isSubmitting}
              whileHover={{ scale: 1.03, boxShadow: "0 0 20px 4px rgba(0, 240, 255, 0.6)" }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center gap-1.5 px-5 py-2 rounded-lg bg-gradient-to-r from-warp-cyan to-starlight text-deep-space font-ui text-xs font-bold tracking-wider hover:brightness-110 transition-all disabled:opacity-50 cursor-pointer shadow-[0_0_20px_rgba(0,240,255,0.4)]"
            >
              <span>✈</span>
              <span>{isSubmitting ? "Submitting..." : "Submit Answer"}</span>
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  );
}
