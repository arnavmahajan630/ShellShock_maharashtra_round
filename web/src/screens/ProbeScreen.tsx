import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMissionStore } from "../store/missionStore";

/**
 * Conditions misconceptions (M06/M07) are code-separable, so the rule-based diagnoser
 * never actually emits status "ambiguous" here — this screen exists for contract
 * completeness and for planets where twins really are code-identical (e.g. M01/M08).
 */
export default function ProbeScreen() {
  const navigate = useNavigate();
  const attempt = useMissionStore((s) => s.attempt);
  const [answered, setAnswered] = useState(false);

  const probe = attempt?.diagnosis?.next_probe;
  if (!probe) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-slate">No probe pending.</p>
      </div>
    );
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
              disabled={answered}
              onClick={() => setAnswered(true)}
              className="font-ui text-xl px-4 py-2 rounded border border-slate text-light hover:border-pulsar-magenta disabled:opacity-50"
            >
              {opt}
            </button>
          ))}
        </div>
        {answered && (
          <div className="mt-6">
            <p className="font-ui text-lg text-slate">Logged.</p>
            <button
              type="button"
              onClick={() => navigate("/planet/conditions/intervention")}
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
