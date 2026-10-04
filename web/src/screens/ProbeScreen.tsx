import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useMissionStore } from "../store/missionStore";
import DeadEnd from "../components/DeadEnd";

/**
 * Conditions misconceptions (M06/M07) are code-separable, so its diagnoser never actually
 * emits status "ambiguous" — this screen only does real work for the Loops P03 hard twin
 * (M01 Boundary Drift vs M08 Index Origin Fault). POST /probe/answer is live for that case.
 */
export default function ProbeScreen() {
  const navigate = useNavigate();
  const { planet } = useParams<{ planet: string }>();
  const learnerId = useMissionStore((s) => s.learnerId);
  const problem = useMissionStore((s) => s.problem);
  const code = useMissionStore((s) => s.code);
  const attempt = useMissionStore((s) => s.attempt);
  const updateDiagnosis = useMissionStore((s) => s.updateDiagnosis);
  const [answered, setAnswered] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Captured once: answering mutates the store's diagnosis (next_probe -> null), which must
  // not yank the question out from under the "Logged." confirmation still being shown.
  const [probe] = useState(() => attempt?.diagnosis?.next_probe ?? null);
  if (!probe || !problem || !attempt) {
    return <DeadEnd message="No probe pending." to={planet ? `/planet/${planet}` : undefined} />;
  }

  async function submit(option: string) {
    setSubmitting(true);
    setError(null);
    try {
      const { diagnosis } = await api.probeAnswer({
        learner_id: learnerId,
        attempt_id: attempt!.attempt_id,
        probe_id: probe!.probe_id,
        answer: option,
        problem_id: problem!.problem_id,
        code,
      });
      updateDiagnosis(diagnosis);
      setAnswered(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't reach the backend.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-deep-space px-6 text-light">
      <div className="max-w-lg w-full rounded-lg border border-pulsar-magenta/50 bg-void-blue/30 p-8">
        <h1 className="font-display text-sm text-pulsar-magenta mb-4">Let me check one thing</h1>
        <pre className="font-mono text-base bg-deep-space/60 border border-void-blue rounded p-3 whitespace-pre-wrap">{probe.code}</pre>
        <p className="font-ui text-xl mt-4">{probe.prompt}</p>
        <div className="flex gap-3 mt-4">
          {probe.options.map((opt) => (
            <button
              key={opt}
              type="button"
              disabled={answered || submitting}
              onClick={() => submit(opt)}
              className="font-ui text-xl px-4 py-2 rounded border border-slate text-light hover:border-pulsar-magenta disabled:opacity-50"
            >
              {opt}
            </button>
          ))}
        </div>
        {error && <p className="font-ui text-lg text-alert-red mt-4">{error}</p>}
        {answered && (
          <div className="mt-6">
            <p className="font-ui text-lg text-slate">Logged.</p>
            <button
              type="button"
              onClick={() => navigate(`/planet/${planet}/intervention`)}
              className="mt-3 font-ui text-xl px-6 py-2 rounded bg-pulsar-magenta text-deep-space hover:brightness-110 active:scale-95"
            >
              Continue →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
