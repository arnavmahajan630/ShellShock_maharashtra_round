import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useMissionStore } from "../store/missionStore";

export default function VerdictScreen() {
  const navigate = useNavigate();
  const verdict = useMissionStore((s) => s.verdict);
  const attempt = useMissionStore((s) => s.attempt);

  if (!verdict) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-slate">No verdict yet.</p>
      </div>
    );
  }

  const stable = verdict.state === "STABLE";
  const className = attempt?.diagnosis?.top[0]?.name ?? "this misconception";

  return (
    <div className="flex min-h-screen items-center justify-center bg-deep-space px-6 text-light">
      <div className="max-w-lg w-full rounded-lg border p-8 text-center" style={{ borderColor: stable ? "var(--color-mint-success)" : "var(--color-alert-red)" }}>
        {stable ? (
          <>
            <motion.div
              className="mx-auto mb-6 h-20 w-20 rounded-full"
              style={{ background: "var(--color-mint-success)" }}
              initial={{ opacity: 0.3, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1, boxShadow: "0 0 32px 10px var(--color-mint-success)" }}
              transition={{ duration: 0.8 }}
            />
            <h1 className="font-display text-sm text-mint-success mb-3">STABLE</h1>
            <p className="font-ui text-xl text-light mb-6">
              {className} looks fixed. Full lock after a later recheck (ghost return or the Trials).
            </p>
            <button
              type="button"
              onClick={() => navigate("/planet/conditions")}
              className="font-ui text-xl px-6 py-2 rounded bg-mint-success text-deep-space hover:brightness-110 active:scale-95"
            >
              Back to path
            </button>
          </>
        ) : (
          <>
            <h1 className="font-display text-sm text-alert-red mb-3">NOT YET</h1>
            <ul className="text-left font-ui text-lg space-y-2 mb-4">
              {verdict.conditions.map((c) => (
                <li key={c.id} className={c.met ? "text-mint-success" : "text-alert-red"}>
                  {c.met ? "✓" : "✗"} {c.label} {c.detail ? `— ${c.detail}` : ""}
                </li>
              ))}
            </ul>
            <p className="font-ui text-lg text-slate mb-6">p_active: {verdict.p_active.toFixed(2)} — try a different explanation.</p>
            <button
              type="button"
              onClick={() => navigate("/planet/conditions/intervention")}
              className="font-ui text-xl px-6 py-2 rounded bg-alert-red text-deep-space hover:brightness-110 active:scale-95"
            >
              Retry intervention
            </button>
          </>
        )}
      </div>
    </div>
  );
}
