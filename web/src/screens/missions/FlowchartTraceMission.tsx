import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, type Problem, type MmProblem } from "../../lib/api";
import { useMissionStore } from "../../store/missionStore";
import FlowchartView, { type FlowchartNode, type FlowchartEdge } from "../../components/mm/FlowchartView";
import PathReplay from "../../components/mm/PathReplay";
import SignalGateWorld from "../../components/SignalGateWorld";
import { countHedges } from "../../lib/speech";

interface FlowchartTraceMissionProps {
  problem: Problem;
}

export default function FlowchartTraceMission({ problem }: FlowchartTraceMissionProps) {
  const { planet } = useParams<{ planet: string }>();
  const navigate = useNavigate();
  const learnerId = useMissionStore((s) => s.learnerId);
  const setAttempt = useMissionStore((s) => s.setAttempt);
  const markPassed = useMissionStore((s) => s.markPassed);

  const [mmProblem, setMmProblem] = useState<MmProblem | null>(null);
  const [selectedPath, setSelectedPath] = useState<string[]>([]);
  const [predictedOutput, setPredictedOutput] = useState<string>("");
  const [explanation, setExplanation] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [attemptResult, setAttemptResult] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [startTime] = useState<number>(Date.now());
  const [showReplay, setShowReplay] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    api
      .getMmProblem(problem.problem_id)
      .then((data) => {
        if (isMounted) setMmProblem(data);
      })
      .catch(() => {
        // Fallback: If mm endpoint failed, use problem metadata from problem.json
        if (isMounted && problem.graph) {
          setMmProblem({
            ...problem,
            type: "flowchart_trace",
            story: problem.prompt,
            inputs: problem.inputs || {},
            code_equivalent: problem.starter,
            graph: problem.graph as any,
            correct_path: ["n1", "n2", "n3", "n4"],
            correct_output: "1",
            rubric: [],
            canonical_explanation: "",
          });
        }
      });
    return () => {
      isMounted = false;
    };
  }, [problem.problem_id]);

  function handleNodeClick(nodeId: string) {
    if (selectedPath[selectedPath.length - 1] === nodeId) return;
    setSelectedPath((prev) => [...prev, nodeId]);
  }

  function handleUndo() {
    setSelectedPath((prev) => prev.slice(0, -1));
  }

  function handleReset() {
    setSelectedPath([]);
  }

  async function handleSubmit() {
    if (selectedPath.length === 0) {
      setErrorMsg("⚠️ Please trace the execution path by tapping nodes on the diagram or clicking the step buttons above.");
      return;
    }
    if (!predictedOutput.trim()) {
      setErrorMsg("⚠️ Please choose or enter your predicted terminal output value (0 or 1 below).");
      return;
    }

    setSubmitting(true);
    setErrorMsg(null);

    const durationMs = Date.now() - startTime;
    const hedges = countHedges(explanation);

    try {
      const res = await api.mmAttempt({
        learner_id: learnerId,
        problem_id: problem.problem_id,
        modality: "image",
        response: {
          path: selectedPath,
          predicted_output: predictedOutput.trim(),
          explanation_text: explanation.trim() || undefined,
        },
        signals: {
          response_ms: durationMs,
          hedge_count: hedges,
        },
      });

      setAttempt(res);
      setAttemptResult(res);

      if (res.gate.code === "G0" && res.tests && res.tests.passed === res.tests.total) {
        markPassed(problem.problem_id);
      } else {
        setShowReplay(true);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to analyze multimodal attempt.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleContinue() {
    if (!attemptResult) return;
    if (attemptResult.tests && attemptResult.tests.passed === attemptResult.tests.total) {
      navigate(`/planet/${planet}`);
    } else {
      navigate(`/planet/${planet}/diagnosis`);
    }
  }

  const nodes: FlowchartNode[] = (mmProblem?.graph?.nodes as FlowchartNode[]) || [
    { id: "n1", kind: "start", label: "Start" },
    { id: "n2", kind: "decision", label: "Condition" },
    { id: "n3", kind: "process", label: "Process" },
    { id: "n4", kind: "end", label: "End" },
  ];

  const edges: FlowchartEdge[] = (mmProblem?.graph?.edges as FlowchartEdge[]) || [];

  const divergenceNode = attemptResult?.diagnosis?.evidence?.find(
    (e: any) => e.chip === "YOU PREDICTED" && e.text.includes("diverged at")
  )?.text?.match(/diverged at (n\d)/)?.[1] || (attemptResult && attemptResult.tests?.passed < attemptResult.tests?.total ? "n2" : null);

  const readyToSubmit = selectedPath.length > 0 && Boolean(predictedOutput.trim());

  return (
    <div className="min-h-screen bg-deep-space px-6 py-8 text-light">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* Mission Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-void-blue pb-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-1 rounded font-display text-[10px] bg-void-blue/80 text-starlight border border-starlight/50 uppercase tracking-wider">
                Multimodal Level A · Flowchart Trace
              </span>
              <span className="font-ui text-lg text-slate">{problem.problem_id}</span>
            </div>
            <h1 className="font-display text-sm md:text-base text-starlight tracking-wide">{problem.name}</h1>
          </div>
          <button
            type="button"
            onClick={() => navigate(`/planet/${planet}`)}
            className="self-start md:self-auto px-4 py-1.5 rounded font-ui text-xl border border-void-blue bg-void-blue/40 text-slate hover:text-light hover:border-warp-cyan transition-colors"
          >
            ← Return to Orbit
          </button>
        </div>

        {/* Narrative / Context Briefing */}
        <div className="rounded-lg border border-void-blue bg-void-blue/40 p-5 backdrop-blur-sm">
          <p className="font-ui text-xl leading-relaxed text-light">
            {mmProblem?.story || problem.prompt}
          </p>
          {mmProblem?.inputs && Object.keys(mmProblem.inputs).length > 0 && (
            <div className="mt-3 flex items-center gap-2 font-ui text-xl text-starlight flex-wrap">
              <span className="text-slate">TELEMETRY INPUTS:</span>
              {Object.entries(mmProblem.inputs).map(([k, v]) => (
                <span key={k} className="px-3 py-0.5 rounded bg-deep-space/80 border border-void-blue font-bold text-starlight">
                  {k} = {String(v)}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Interactive Flowchart Diagram */}
        <FlowchartView
          nodes={nodes}
          edges={edges}
          selectedPath={selectedPath}
          onNodeClick={handleNodeClick}
          onUndo={handleUndo}
          onReset={handleReset}
        />

        {/* Learner Predictions and Explanations */}
        {!attemptResult && (
          <div className="rounded-lg border border-void-blue bg-void-blue/40 p-6 space-y-5">
            <div>
              <label className="block font-display text-xs text-warp-cyan mb-3">
                1. Predict Terminal Output Value
              </label>
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      setPredictedOutput("0");
                      setErrorMsg(null);
                    }}
                    className={`font-ui text-2xl px-6 py-2 rounded border transition-all cursor-pointer ${
                      predictedOutput === "0"
                        ? "border-starlight bg-starlight/20 text-starlight shadow-[0_0_15px_rgba(252,177,67,0.4)]"
                        : "border-void-blue bg-deep-space/70 text-light hover:border-starlight hover:text-starlight"
                    }`}
                  >
                    0 (shield off / 0)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPredictedOutput("1");
                      setErrorMsg(null);
                    }}
                    className={`font-ui text-2xl px-6 py-2 rounded border transition-all cursor-pointer ${
                      predictedOutput === "1"
                        ? "border-starlight bg-starlight/20 text-starlight shadow-[0_0_15px_rgba(252,177,67,0.4)]"
                        : "border-void-blue bg-deep-space/70 text-light hover:border-starlight hover:text-starlight"
                    }`}
                  >
                    1 (shield active / 1)
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-ui text-xl text-slate">or custom:</span>
                  <input
                    type="text"
                    value={predictedOutput}
                    onChange={(e) => {
                      setPredictedOutput(e.target.value);
                      setErrorMsg(null);
                    }}
                    placeholder="0 or 1"
                    className="w-24 rounded border border-void-blue bg-deep-space px-3 py-1 font-ui text-xl text-light placeholder-slate/40 focus:border-starlight focus:outline-none"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block font-display text-xs text-warp-cyan mb-3">
                2. Explain Your Reasoning (Optional Rationale)
              </label>
              <textarea
                value={explanation}
                onChange={(e) => setExplanation(e.target.value)}
                rows={2}
                placeholder="Why did you trace this branch? (e.g., 'power is 10 which is >= 5, so condition is true...')"
                className="w-full rounded border border-void-blue bg-deep-space p-3 font-ui text-xl text-light placeholder-slate/40 focus:border-starlight focus:outline-none"
              />
            </div>

            {/* Mission Readiness Checklist */}
            <div className="flex items-center gap-4 font-ui text-xl py-2 px-4 rounded bg-deep-space/70 border border-void-blue">
              <span className={selectedPath.length > 0 ? "text-mint-success font-bold flex items-center gap-1" : "text-starlight flex items-center gap-1"}>
                {selectedPath.length > 0 ? `✓ Path (${selectedPath.join(" → ")})` : "○ 1. Trace flowchart above"}
              </span>
              <span className={predictedOutput.trim() ? "text-mint-success font-bold flex items-center gap-1" : "text-starlight flex items-center gap-1"}>
                {predictedOutput.trim() ? `✓ Output: ${predictedOutput}` : "○ 2. Choose output (0 or 1)"}
              </span>
            </div>

            {errorMsg && (
              <div className="rounded border border-alert-red bg-alert-red/10 p-3 font-ui text-xl text-alert-red">
                {errorMsg}
              </div>
            )}

            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className={`w-full rounded py-3 font-ui text-2xl font-bold uppercase tracking-wider transition-all cursor-pointer ${
                readyToSubmit
                  ? "bg-starlight hover:brightness-110 text-deep-space shadow-[0_0_20px_rgba(252,177,67,0.4)] active:scale-95"
                  : "bg-void-blue text-slate border border-void-blue opacity-50"
              }`}
            >
              {submitting ? "Analyzing Execution Trace..." : "Transmit Path Evaluation →"}
            </button>
          </div>
        )}

        {/* Results & Divergence Inspection */}
        {attemptResult && (
          <div className="space-y-6">
            <SignalGateWorld
              label={problem.name}
              status={attemptResult.tests?.passed === attemptResult.tests?.total ? "pass" : "fail"}
              passed={attemptResult.tests?.passed ?? 0}
              total={attemptResult.tests?.total ?? 1}
            />

            {showReplay && attemptResult.tests?.passed < attemptResult.tests?.total && (
              <PathReplay
                learnerPath={selectedPath}
                correctPath={mmProblem?.correct_path || ["n1", "n2", "n3", "n4"]}
                divergenceNode={divergenceNode}
                divergenceReason={
                  divergenceNode === "n2"
                    ? "With power = 10, the condition power >= 5 is true (10 >= 5), so the active shield branch (n3: shield = 1) is followed instead of skipping directly to n4."
                    : undefined
                }
                nodes={nodes}
                onClose={() => setShowReplay(false)}
              />
            )}

            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={handleContinue}
                className="font-ui text-2xl px-8 py-2.5 rounded bg-warp-cyan text-deep-space font-bold shadow-[0_0_15px_rgba(45,231,252,0.4)] hover:brightness-110 active:scale-95 transition-all"
              >
                Continue to Diagnostic Telemetry →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
