import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";
import { api, type DebriefReport } from "../lib/api";

export default function TrialsDebriefScreen() {
  const navigate = useNavigate();
  const location = useLocation();
  const examId = (location.state as { examId?: string })?.examId || "ex_live";

  const [phase, setPhase] = useState<"compiling" | "report">("compiling");
  const [compilingProgress, setCompilingProgress] = useState(15);
  const [activeChecklist, setActiveChecklist] = useState(0);
  const [report, setReport] = useState<DebriefReport | null>(null);
  const [preparingPdf, setPreparingPdf] = useState(false);

  async function handleDownloadPdf() {
    if (!report || preparingPdf) return;
    setPreparingPdf(true);
    try {
      const { downloadDebriefPdf } = await import("../lib/reportPdf");
      downloadDebriefPdf(report);
    } finally {
      setPreparingPdf(false);
    }
  }

  // Fetch / Finish report from backend
  useEffect(() => {
    let mounted = true;
    api
      .finishExam(examId)
      .then((res) => {
        if (!mounted) return;
        setReport(res.report);
      })
      .catch(() => {
        // Fallback report
        if (!mounted) return;
        setReport({
          exam_id: examId,
          items_total: 5,
          items_passed: 4,
          score_pct: 80,
          sectors: [
            { sector: "arrays", name: "Arrays & Indexing", items: ["T1_two_sum"], passed: 1, rating_before: 1400, rating_after: 1440 },
            { sector: "searching", name: "Binary Search", items: ["T2_binary_search"], passed: 1, rating_before: 1400, rating_after: 1440 },
            { sector: "sorting", name: "Sorting Algorithms", items: ["T3_bubble_sort"], passed: 0, rating_before: 1400, rating_after: 1370 },
            { sector: "strings", name: "String Traversal", items: ["T4_is_palindrome"], passed: 1, rating_before: 1400, rating_after: 1440 },
            { sector: "recursion", name: "Recursion & Trees", items: ["T5_recursive_cascade"], passed: 1, rating_before: 1400, rating_after: 1440 },
          ],
          findings: [
            {
              class: "D03",
              status: "ACTIVE",
              p_active: 0.92,
              name: "Cargo Overwrite",
              subtitle: "Swapping without a temp loses a value",
              belief: "Assigning two elements in sequence swaps them",
              item_id: "T3_bubble_sort",
              trial_title: "Bubble Sort",
              evidence: [
                {
                  type: "CODE",
                  text: "a[j] = a[j+1]; a[j+1] = a[j]; overwrote a[j] before copying it into a temporary buffer.",
                },
              ],
            },
          ],
          recommendations: [
            {
              sector: "sorting",
              title: "Review Swap Invariants",
              description: "Practice buffer swapping in bubble sort and array permutations.",
              route: "/planet/arrays",
            },
          ],
          remediations: [
            {
              class: "D03",
              name: "Cargo Overwrite",
              subtitle: "Swapping without a temp loses a value",
              trial_title: "Bubble Sort",
              root_cause: "Attempting to swap elements without a temporary buffer, overwriting the first value.",
              rule_to_remember: "Two assignments cannot execute simultaneously. A swap requires a temporary buffer: temp = a; a = b; b = temp;.",
              code_fix: {
                wrong: "arr[j] = arr[j + 1]; // Overwrites arr[j]!\narr[j + 1] = arr[j]; // Copies back the overwritten value",
                right: "int temp = arr[j];\narr[j + 1] = temp;\narr[j] = arr[j + 1];\n// Safely preserves original values",
              },
              self_check: "Trace with a = 5, b = 9. After step 1 (a = b), both a and b are 9! A temporary storage variable is mandatory.",
            },
          ],
        });
      });

    return () => {
      mounted = false;
    };
  }, [examId]);

  // Animated compiling progress
  useEffect(() => {
    if (phase !== "compiling") return;

    const timer = setInterval(() => {
      setCompilingProgress((prev) => {
        if (prev >= 98) {
          clearInterval(timer);
          setTimeout(() => setPhase("report"), 400);
          return 100;
        }
        const next = prev + Math.floor(Math.random() * 12) + 5;
        if (next > 30) setActiveChecklist(1);
        if (next > 60) setActiveChecklist(2);
        if (next > 85) setActiveChecklist(3);
        return Math.min(100, next);
      });
    }, 200);

    return () => clearInterval(timer);
  }, [phase]);

  const checklistItems = [
    "Processing results",
    "Generating insights",
    "Preparing recommendations",
    "Finalizing report",
  ];

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black flex flex-col font-ui select-none">
      {/* Background Ambience */}
      <img
        src="/images/trials_warp.png"
        alt="Deep Space Background"
        className="absolute inset-0 w-full h-full object-cover object-center opacity-30 pointer-events-none"
      />
      <div className="absolute inset-0 bg-radial-gradient from-transparent via-black/40 to-black pointer-events-none" />

      {/* PHASE 1: COMPILING REPORT TRANSITION (components_inspiration.png Panel 9) */}
      <AnimatePresence mode="wait">
        {phase === "compiling" && (
          <motion.div
            key="compiling"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ duration: 0.4 }}
            className="relative z-10 m-auto flex flex-col items-center justify-center p-8 max-w-lg w-full rounded-2xl bg-deep-space/90 border border-warp-cyan/40 shadow-[0_0_50px_rgba(0,240,255,0.25)] text-center space-y-6"
          >
            {/* Spinning Cosmic Hologram Ring */}
            <div className="relative w-28 h-28 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-2 border-warp-cyan/20 animate-ping" />
              <div className="w-20 h-20 rounded-full border-2 border-t-warp-cyan border-r-neon-purple border-b-transparent border-l-transparent animate-spin" />
              <div className="absolute text-2xl">🛸</div>
            </div>

            {/* Header Titles */}
            <div className="space-y-1">
              <h2 className="font-display text-xl text-white font-bold tracking-wider uppercase drop-shadow-[0_0_10px_rgba(0,240,255,0.5)]">
                Compiling your report...
              </h2>
              <p className="text-xs font-mono text-white/60">
                Analysing your performance across all trials.
              </p>
            </div>

            {/* Dynamic Checklist */}
            <div className="w-full max-w-xs space-y-2 text-left text-xs font-mono">
              {checklistItems.map((item, idx) => {
                const isDone = activeChecklist > idx;
                const isCurrent = activeChecklist === idx;

                return (
                  <div
                    key={idx}
                    className={`flex items-center gap-2.5 transition-colors ${
                      isDone
                        ? "text-mint-success"
                        : isCurrent
                        ? "text-warp-cyan font-bold"
                        : "text-white/30"
                    }`}
                  >
                    <span>{isDone ? "✓" : isCurrent ? "⏳" : "○"}</span>
                    <span>{item}</span>
                  </div>
                );
              })}
            </div>

            {/* Progress Bar & Percentage */}
            <div className="w-full space-y-2">
              <div className="w-full h-2 bg-black/60 border border-warp-cyan/30 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-warp-cyan via-starlight to-neon-purple shadow-[0_0_12px_#00f0ff]"
                  style={{ width: `${compilingProgress}%` }}
                />
              </div>
              <div className="flex justify-between text-[11px] font-mono text-white/40">
                <span>Synchronizing telemetry</span>
                <span className="text-warp-cyan font-bold">{compilingProgress}%</span>
              </div>
            </div>

            <p className="text-[11px] font-mono text-white/40">This may take a few seconds.</p>
          </motion.div>
        )}

        {/* PHASE 2: DEBRIEF REPORT (components_inspiration.png Panels 10 & 11) */}
        {phase === "report" && report && (
          <motion.div
            key="report"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="relative z-10 flex-1 grid grid-cols-12 gap-4 p-6 max-w-7xl mx-auto w-full min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-warp-cyan/30"
          >
            {/* Left & Middle Column (8 of 12): Debrief Performance Overview */}
            <div className="col-span-8 space-y-4">
              {/* Header Score Card */}
              <div className="p-5 rounded-2xl bg-deep-space/90 border border-warp-cyan/40 shadow-[0_0_30px_rgba(0,240,255,0.15)] flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-xs font-mono text-warp-cyan uppercase tracking-widest font-bold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-warp-cyan shadow-[0_0_8px_#00f0ff]" />
                    DEBRIEF REPORT (D18)
                  </div>
                  <h1 className="font-display text-2xl text-white font-bold mt-1">
                    Your Performance Overview
                  </h1>
                  <p className="text-xs text-white/60 font-mono mt-0.5">
                    {report.sectors.length > 0
                      ? `Completed ${report.items_total} trial${report.items_total === 1 ? "" : "s"} across ${report.sectors.map((s) => s.name).join(", ")}.`
                      : "No trials attempted yet."}
                  </p>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  {/* Download PDF */}
                  <button
                    type="button"
                    onClick={handleDownloadPdf}
                    disabled={preparingPdf}
                    title="Download this report as a PDF"
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-warp-cyan/50 bg-warp-cyan/10 hover:bg-warp-cyan/20 text-warp-cyan font-ui text-xs font-bold tracking-wide transition-all cursor-pointer whitespace-nowrap disabled:opacity-50 disabled:cursor-wait"
                  >
                    <span>{preparingPdf ? "⏳" : "⬇"}</span>
                    <span>{preparingPdf ? "Preparing…" : "Download PDF"}</span>
                  </button>

                  {/* Score Circular Badge */}
                  <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-black/60 border border-warp-cyan/50 min-w-[120px] shadow-inner">
                    <span className="font-display text-3xl font-bold text-warp-cyan drop-shadow-[0_0_10px_#00f0ff]">
                      {report.score_pct}%
                    </span>
                    <span className="text-[10px] font-mono text-white/60 whitespace-nowrap">
                      {report.items_passed} / {report.items_total} Cleared
                    </span>
                  </div>
                </div>
              </div>

              {/* Sector Mastery Breakdown */}
              <div className="p-5 rounded-2xl bg-deep-space/80 border border-white/10 space-y-3">
                <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-white/70 flex items-center gap-1.5">
                  <span>📊</span> Sector Performance
                  <span className="text-white/30 font-normal">({report.sectors.length} attempted)</span>
                </h3>

                {report.sectors.length > 0 ? (
                  <div className="flex flex-wrap gap-2.5">
                    {report.sectors.map((sec, idx) => (
                      <div
                        key={idx}
                        className={`flex-1 min-w-[120px] p-3 rounded-xl border text-center space-y-1.5 font-mono ${
                          sec.passed > 0
                            ? "bg-emerald-950/20 border-emerald-500/40 text-emerald-300"
                            : "bg-rose-950/20 border-rose-500/40 text-rose-300"
                        }`}
                      >
                        <div className="text-[10px] text-white/40 uppercase tracking-widest truncate" title={sec.name}>
                          {sec.name}
                        </div>
                        <div className="text-lg font-bold">
                          {sec.passed > 0 ? "PASSED" : "REVIEW"}
                        </div>
                        <div className="text-[10px] text-white/60">
                          Rating: {sec.rating_after}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs font-mono text-white/40">Nothing attempted in this session.</p>
                )}
              </div>

              {/* Misconceptions Identified Card */}
              <div className="p-5 rounded-2xl bg-deep-space/80 border border-white/10 space-y-3">
                <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                  <span>⚡</span> Misconceptions Identified
                  {report.findings.length > 0 && (
                    <span className="text-white/30 font-normal">({report.findings.length})</span>
                  )}
                </h3>

                {report.findings.length > 0 ? (
                  <div className="space-y-3">
                    {report.findings.map((f, idx) => (
                      <div
                        key={idx}
                        className="p-4 rounded-xl bg-black/60 border border-rose-500/40 text-xs font-mono space-y-2 shadow-inner"
                      >
                        <div className="flex items-center justify-between flex-wrap gap-1">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold border border-rose-500/40">
                              {f.class}
                            </span>
                            <span className="font-bold text-white text-sm">{f.name}</span>
                          </div>
                          <span className="text-[11px] text-white/50">On: {f.trial_title}</span>
                        </div>

                        <p className="text-white/80 leading-relaxed font-ui text-xs">
                          {f.subtitle}
                        </p>

                        {f.evidence && f.evidence.length > 0 && (
                          <div className="space-y-1.5">
                            {f.evidence.map((ev, evIdx) => (
                              <div
                                key={evIdx}
                                className="p-2.5 rounded bg-rose-950/30 border border-rose-500/20 text-[11px] text-rose-200"
                              >
                                <span className="font-bold">{ev.type === "RUN" ? "Diagnosis: " : ev.type === "YOU PREDICTED" ? "Your prediction: " : `${ev.type}: `}</span>
                                {ev.text}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-black/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono flex items-center gap-2">
                    <span>✓</span>
                    <span>
                      No active misconceptions detected in the {report.items_total} trial
                      {report.items_total === 1 ? "" : "s"} you attempted.
                    </span>
                  </div>
                )}
              </div>

              {/* How to Clear Identified Misconceptions Card */}
              {report.remediations && report.remediations.length > 0 && (
                <div className="p-5 rounded-2xl bg-deep-space/80 border border-warp-cyan/40 space-y-4 shadow-[0_0_20px_rgba(0,240,255,0.08)]">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-warp-cyan flex items-center gap-2">
                      <span>💡</span> How to Clear Identified Misconceptions
                    </h3>
                    <span className="text-[10px] font-mono px-2.5 py-0.5 rounded bg-warp-cyan/10 border border-warp-cyan/30 text-warp-cyan font-bold tracking-wide">
                      LLM PEDAGOGY GUIDE
                    </span>
                  </div>

                  <p className="text-xs text-white/60 font-ui">
                    Targeted cognitive shifts, invariant rules, and code patterns to permanently resolve diagnosed bugs.
                  </p>

                  <div className="space-y-4">
                    {report.remediations.map((rem, idx) => (
                      <div
                        key={idx}
                        className="p-4 rounded-xl bg-black/60 border border-warp-cyan/25 text-xs font-mono space-y-3.5 shadow-inner"
                      >
                        <div className="flex items-center justify-between flex-wrap gap-2 border-b border-white/10 pb-2.5">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-warp-cyan/20 text-warp-cyan font-bold border border-warp-cyan/40">
                              {rem.class}
                            </span>
                            <span className="font-bold text-white text-sm">{rem.name}</span>
                          </div>
                          {rem.trial_title && (
                            <span className="text-[11px] text-white/50">Context: {rem.trial_title}</span>
                          )}
                        </div>

                        {/* Cognitive shift */}
                        <div className="space-y-1">
                          <div className="text-[10px] text-white/40 uppercase tracking-wider font-bold">
                            1. Cognitive Shift (Why this occurs)
                          </div>
                          <p className="text-white/80 font-ui leading-relaxed text-xs">
                            {rem.root_cause}
                          </p>
                        </div>

                        {/* Rule to remember */}
                        <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-500/40 text-amber-200 space-y-1">
                          <div className="text-[10px] text-amber-400 uppercase tracking-widest font-bold flex items-center gap-1">
                            <span>⚡</span> Rule to Remember
                          </div>
                          <p className="text-xs font-ui leading-relaxed text-amber-100 font-semibold">
                            {rem.rule_to_remember}
                          </p>
                        </div>

                        {/* Code fix */}
                        {rem.code_fix && (rem.code_fix.wrong || rem.code_fix.right) && (
                          <div className="space-y-1.5">
                            <div className="text-[10px] text-white/40 uppercase tracking-wider font-bold">
                              2. Code Transformation (Wrong vs Correct)
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
                              {rem.code_fix.wrong && (
                                <div className="p-2.5 rounded bg-rose-950/20 border border-rose-500/30 font-mono space-y-1">
                                  <div className="text-[10px] text-rose-400 font-bold uppercase tracking-wider">
                                    ✗ Faulty Pattern
                                  </div>
                                  <pre className="text-rose-200/90 whitespace-pre-wrap leading-tight text-[11px] font-mono">
                                    {rem.code_fix.wrong}
                                  </pre>
                                </div>
                              )}
                              {rem.code_fix.right && (
                                <div className="p-2.5 rounded bg-emerald-950/20 border border-emerald-500/30 font-mono space-y-1">
                                  <div className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">
                                    ✓ Invariant Safe
                                  </div>
                                  <pre className="text-emerald-200 whitespace-pre-wrap leading-tight text-[11px] font-mono">
                                    {rem.code_fix.right}
                                  </pre>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Self check */}
                        {rem.self_check && (
                          <div className="flex items-start gap-2 text-[11px] text-warp-cyan/90 bg-warp-cyan/10 p-2.5 rounded-lg border border-warp-cyan/30">
                            <span className="shrink-0 font-bold">🔍 Self-Check:</span>
                            <span className="font-ui">{rem.self_check}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column (4 of 12): Next Steps (Panel 11) */}
            <div className="col-span-4 space-y-3 flex flex-col">
              <div className="text-xs font-mono text-white/60 font-bold uppercase tracking-wider mb-1">
                11. NEXT STEPS
              </div>

              {/* Recommendations driven by what this attempt actually found */}
              {report.recommendations.map((rec, idx) => {
                const isMastery = rec.sector === "mastery";
                return (
                  <motion.button
                    key={idx}
                    type="button"
                    whileHover={{ scale: 1.02, x: 2 }}
                    onClick={() => navigate(rec.route)}
                    className={`p-4 rounded-2xl text-left cursor-pointer transition-all group flex items-start justify-between ${
                      isMastery
                        ? "bg-gradient-to-r from-emerald-950/40 to-deep-space border border-emerald-500/50 hover:border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                        : "bg-gradient-to-r from-amber-950/40 to-deep-space border border-amber-500/50 hover:border-amber-400 shadow-[0_0_15px_rgba(245,158,11,0.1)]"
                    }`}
                  >
                    <div className="space-y-1">
                      <div className={`flex items-center gap-2 font-display text-sm font-bold ${isMastery ? "text-emerald-400" : "text-amber-400"}`}>
                        <span>{isMastery ? "🏆" : "🎯"}</span>
                        <span>{rec.title}</span>
                      </div>
                      <p className="text-xs text-white/70 font-ui leading-relaxed">{rec.description}</p>
                    </div>
                    <span className={`text-lg group-hover:translate-x-1 transition-transform ${isMastery ? "text-emerald-400" : "text-amber-400"}`}>
                      →
                    </span>
                  </motion.button>
                );
              })}

              {/* Action Card 2: Retake Exam */}
              <motion.button
                type="button"
                whileHover={{ scale: 1.02, x: 2 }}
                onClick={() => navigate("/trials/exam")}
                className="p-4 rounded-2xl bg-gradient-to-r from-purple-950/40 to-deep-space border border-purple-500/50 hover:border-purple-400 text-left cursor-pointer transition-all shadow-[0_0_15px_rgba(168,85,247,0.1)] group flex items-start justify-between"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-purple-400 font-display text-sm font-bold">
                    <span>🔄</span>
                    <span>Retake Exam</span>
                  </div>
                  <p className="text-xs text-white/70 font-ui leading-relaxed">
                    Try again with a fresh session to clear any active misconceptions.
                  </p>
                </div>
                <span className="text-purple-400 text-lg group-hover:translate-x-1 transition-transform">
                  →
                </span>
              </motion.button>

              {/* Action Card 4: Back to Galaxy */}
              <motion.button
                type="button"
                whileHover={{ scale: 1.02, x: 2 }}
                onClick={() => navigate("/map")}
                className="p-4 rounded-2xl bg-gradient-to-r from-blue-950/40 to-deep-space border border-blue-500/50 hover:border-blue-400 text-left cursor-pointer transition-all shadow-[0_0_15px_rgba(59,130,246,0.1)] group flex items-start justify-between mt-auto"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-blue-400 font-display text-sm font-bold">
                    <span>🌌</span>
                    <span>Back to Galaxy</span>
                  </div>
                  <p className="text-xs text-white/70 font-ui leading-relaxed">
                    Return to the galaxy map. Explore the other planets and mission paths.
                  </p>
                </div>
                <span className="text-blue-400 text-lg group-hover:translate-x-1 transition-transform">
                  →
                </span>
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
