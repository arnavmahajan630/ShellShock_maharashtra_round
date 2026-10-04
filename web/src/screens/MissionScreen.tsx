import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { cpp } from "@codemirror/lang-cpp";
import CodeMirror from "@uiw/react-codemirror";
import { api } from "../lib/api";
import { useMissionStore } from "../store/missionStore";
import SignalGateWorld from "../components/SignalGateWorld";

type Phase = "loading" | "predict" | "code" | "run";

export default function MissionScreen() {
  const { planet, problemId } = useParams<{ planet: string; problemId: string }>();
  const navigate = useNavigate();
  const learnerId = useMissionStore((s) => s.learnerId);
  const problem = useMissionStore((s) => s.problem);
  const code = useMissionStore((s) => s.code);
  const attempt = useMissionStore((s) => s.attempt);
  const startMission = useMissionStore((s) => s.startMission);
  const setCode = useMissionStore((s) => s.setCode);
  const setAttempt = useMissionStore((s) => s.setAttempt);
  const markPassed = useMissionStore((s) => s.markPassed);

  const [phase, setPhase] = useState<Phase>("loading");
  const [predictAnswer, setPredictAnswer] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  function load() {
    if (!problemId) return;
    setPhase("loading");
    setLoadError(null);
    api
      .getProblem(problemId)
      .then((p) => {
        startMission(p);
        setPhase(p.predict_item ? "predict" : "code");
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : "Couldn't reach the backend."));
  }

  useEffect(load, [problemId]);

  if (loadError) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-deep-space">
        <p className="font-ui text-2xl text-alert-red">{loadError}</p>
        <p className="font-ui text-lg text-slate">Is the backend running? (uvicorn server.app.main:app --port 8000)</p>
        <button type="button" onClick={load} className="font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110">
          Retry
        </button>
      </div>
    );
  }

  if (phase === "loading" || !problem) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-slate">Loading mission…</p>
      </div>
    );
  }

  async function runCode() {
    setRunning(true);
    setPhase("run");
    setRunError(null);
    try {
      const result = await api.attempt({
        learner_id: learnerId,
        problem_id: problem!.problem_id,
        code,
        prediction: predictAnswer ?? undefined,
      });
      setAttempt(result);
      if (result.gate.code === "G0" && result.tests && result.tests.passed === result.tests.total) {
        markPassed(problem!.problem_id);
      }
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Couldn't reach the backend.");
    } finally {
      setRunning(false);
    }
  }

  function afterRun() {
    if (!attempt) return;
    if (attempt.gate.code !== "G0") {
      setPhase("code");
      return;
    }
    if (attempt.tests && attempt.tests.passed === attempt.tests.total) {
      navigate(`/planet/${planet}`);
    } else {
      navigate(`/planet/${planet}/diagnosis`);
    }
  }

  return (
    <div className="min-h-screen bg-deep-space px-6 py-8 text-light">
      <div className="mx-auto max-w-3xl">
        <p className="font-display text-xs text-starlight mb-1">{problem.name}</p>
        <p className="font-ui text-xl text-slate mb-6">{problem.prompt}</p>

        {phase === "predict" && problem.predict_item && (
          <div className="rounded-lg border border-void-blue bg-void-blue/40 p-6">
            <h2 className="font-display text-xs text-warp-cyan mb-4">Predict</h2>
            <pre className="font-mono text-base bg-deep-space/60 border border-void-blue rounded p-3 whitespace-pre-wrap">{problem.predict_item.code}</pre>
            <p className="font-ui text-xl mt-4">{problem.predict_item.question}</p>
            <div className="flex gap-3 mt-3">
              {problem.predict_item.options.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => setPredictAnswer(opt)}
                  className={
                    "font-ui text-xl px-4 py-2 rounded border " +
                    (predictAnswer === opt ? "border-warp-cyan text-warp-cyan" : "border-slate text-light hover:border-warp-cyan")
                  }
                >
                  {opt}
                </button>
              ))}
            </div>
            <button
              type="button"
              disabled={predictAnswer === null}
              onClick={() => setPhase("code")}
              className="mt-6 font-ui text-xl px-6 py-2 rounded bg-starlight text-deep-space disabled:opacity-40 hover:brightness-110 active:scale-95"
            >
              Lock in → Code
            </button>
          </div>
        )}

        {(phase === "code" || phase === "run") && (
          <div className="rounded-lg border border-void-blue overflow-hidden">
            <CodeMirror value={code} height="280px" theme="dark" extensions={[cpp()]} onChange={(value) => setCode(value)} />
          </div>
        )}

        {phase === "code" && (
          <button
            type="button"
            onClick={runCode}
            className="mt-4 font-ui text-xl px-6 py-2 rounded bg-mint-success text-deep-space hover:brightness-110 active:scale-95"
          >
            ▶ Run
          </button>
        )}

        {phase === "run" && (
          <div className="mt-6">
            {runError ? (
              <div className="rounded-lg border border-alert-red bg-alert-red/10 p-6">
                <p className="font-ui text-xl text-alert-red">{runError}</p>
                <p className="font-ui text-lg text-slate mt-1">Is the backend running? (uvicorn server.app.main:app --port 8000)</p>
                <button
                  type="button"
                  onClick={() => setPhase("code")}
                  className="mt-4 font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
                >
                  Back to code
                </button>
              </div>
            ) : (
              <>
                <SignalGateWorld
                  label={problem.name}
                  status={running ? "running" : attempt?.gate.code !== "G0" ? "fail" : attempt?.tests && attempt.tests.passed === attempt.tests.total ? "pass" : "fail"}
                  passed={attempt?.tests?.passed ?? 0}
                  total={attempt?.tests?.total ?? 0}
                />
                {!running && attempt && attempt.gate.code !== "G0" && (
                  <p className="font-ui text-xl text-alert-red mt-4">{attempt.gate.message}</p>
                )}
                {!running && attempt && (
                  <button
                    type="button"
                    onClick={afterRun}
                    className="mt-6 font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
                  >
                    Continue
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
