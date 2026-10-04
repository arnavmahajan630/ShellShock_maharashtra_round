import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useMissionStore } from "../store/missionStore";
import ValueMeter from "../components/ValueMeter";
import MemoryStrip from "../components/MemoryStrip";
import DeadEnd from "../components/DeadEnd";

export default function InterventionScreen() {
  const navigate = useNavigate();
  const { planet } = useParams<{ planet: string }>();
  const problem = useMissionStore((s) => s.problem);
  const code = useMissionStore((s) => s.code);
  const attempt = useMissionStore((s) => s.attempt);
  const intervention = useMissionStore((s) => s.intervention);
  const setIntervention = useMissionStore((s) => s.setIntervention);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!problem || !attempt?.diagnosis?.top[0]) return;
    setLoadError(null);
    api
      .intervene({ problem_id: problem.problem_id, code, class: attempt.diagnosis.top[0].id })
      .then(setIntervention)
      .catch((err) => setLoadError(err instanceof Error ? err.message : "Couldn't reach the backend."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [problem?.problem_id]);

  const [showFix, setShowFix] = useState(false);

  if (!problem || !attempt?.diagnosis?.top[0]) {
    return <DeadEnd message="No diagnosis to intervene on." to={planet ? `/planet/${planet}` : undefined} />;
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-alert-red">{loadError}</p>
      </div>
    );
  }

  if (!intervention) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-slate">Preparing intervention…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-deep-space px-6 py-10 text-light">
      <div className="mx-auto max-w-2xl rounded-lg border border-void-blue bg-void-blue/30 p-8">
        <h1 className="font-display text-sm text-starlight mb-4">{attempt.diagnosis.top[0].name}</h1>
        <div className="font-ui text-xl space-y-2 mb-6">
          {intervention.copy.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>

        <div className="mb-6">
          {intervention.modality === "memory_strip" && intervention.memory_strip && (
            <MemoryStrip {...intervention.memory_strip} />
          )}
          {intervention.modality === "value_meter" && intervention.value_meter && (
            <ValueMeter {...intervention.value_meter} />
          )}
          {intervention.modality === "trace_timeline" && (
            <>
              <h2 className="font-display text-xs text-warp-cyan mb-3">What happened, step by step</h2>
              <ul className="space-y-1 font-mono text-sm">
                {intervention.timeline.map((step) => (
                  <li key={step.step} className="rounded border border-void-blue bg-deep-space/50 px-3 py-2">
                    line {step.line} · <span className="text-pulsar-magenta">{step.flag}</span> · {JSON.stringify(step.vars)}
                  </li>
                ))}
                {intervention.timeline.length === 0 && <li className="font-ui text-lg text-slate">No flagged steps captured.</li>}
              </ul>
            </>
          )}
        </div>

        {intervention.counterexample && (
          <div className="mb-6">
            <h2 className="font-display text-xs text-warp-cyan mb-2">Try this input</h2>
            <p className="font-ui text-lg">
              Input: <code>{JSON.stringify(intervention.counterexample.input)}</code> · Intended:{" "}
              <code className="text-mint-success">{JSON.stringify(intervention.counterexample.intended)}</code> · Yours:{" "}
              <code className="text-alert-red">{JSON.stringify(intervention.counterexample.yours)}</code>
            </p>
          </div>
        )}

        {intervention.fix && (
          <div className="mb-6">
            <button
              type="button"
              onClick={() => setShowFix((v) => !v)}
              className="font-ui text-lg text-warp-cyan underline"
            >
              {showFix ? "Hide" : "Show"} reference solution {intervention.fix.verified && "(verified ✓)"}
            </button>
            {showFix && (
              <pre className="font-mono text-base bg-deep-space/60 border border-mint-success/40 rounded p-3 mt-2 whitespace-pre-wrap">
                {intervention.fix.code}
              </pre>
            )}
          </div>
        )}

        <button
          type="button"
          onClick={() => navigate(`/planet/${planet}/trap`)}
          className="font-ui text-xl px-6 py-2 rounded bg-mint-success text-deep-space hover:brightness-110 active:scale-95"
        >
          I understand it now →
        </button>
      </div>
    </div>
  );
}
