import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { useMissionStore } from "../store/missionStore";
import DeadEnd from "../components/DeadEnd";

const BAND_COLOR: Record<string, string> = {
  Likely: "var(--color-mint-success)",
  Possible: "var(--color-starlight)",
  Unsure: "var(--color-slate)",
};

const CHIP_LABEL: Record<string, string> = {
  CODE: "CODE",
  RUN: "RUN",
  "YOU PREDICTED": "YOU PREDICTED",
  PROBE: "PROBE",
  HISTORY: "HISTORY",
  EXAM: "EXAM",
};

export default function DiagnosisScreen() {
  const navigate = useNavigate();
  const { planet } = useParams<{ planet: string }>();
  const attempt = useMissionStore((s) => s.attempt);
  const problem = useMissionStore((s) => s.problem);
  const [scanning, setScanning] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => setScanning(false), 700);
    return () => clearTimeout(t);
  }, []);

  if (!attempt || !attempt.diagnosis || !problem) {
    return <DeadEnd message="No attempt in progress." to={planet ? `/planet/${planet}` : undefined} />;
  }

  const diagnosis = attempt.diagnosis;

  if (scanning) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-deep-space">
        <motion.div
          className="h-16 w-16 rounded-full border-4 border-warp-cyan border-t-transparent"
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
        />
        <p className="font-ui text-2xl text-warp-cyan">Scanning…</p>
      </div>
    );
  }

  function proceed() {
    if (diagnosis!.status === "ambiguous") navigate(`/planet/${planet}/probe`);
    else navigate(`/planet/${planet}/intervention`);
  }

  const isNovel = diagnosis.status === "novel";

  return (
    <div className="min-h-screen bg-deep-space px-6 py-10 text-light">
      <div className="mx-auto max-w-2xl rounded-lg border border-void-blue bg-void-blue/30 p-8">
        <h1 className="font-display text-sm text-starlight mb-2">Diagnosis</h1>
        <p className="font-ui text-lg text-slate mb-6">{problem.name}</p>

        {isNovel ? (
          <div className="mb-6">
            <p className="font-display text-sm text-pulsar-magenta">Unknown Anomaly</p>
            <p className="font-ui text-xl text-slate mt-1">Doesn't match any known pattern</p>
          </div>
        ) : (
          <div className="mb-6 space-y-3">
            {diagnosis.top.map((card) => (
              <div key={card.id}>
                <div className="flex justify-between font-ui text-xl">
                  <span>{card.name}</span>
                  <span style={{ color: BAND_COLOR[card.band] }}>{card.band}</span>
                </div>
                <p className="font-ui text-base text-slate">{card.subtitle}</p>
                <div className="h-2 mt-1 rounded bg-deep-space overflow-hidden">
                  <motion.div
                    className="h-full"
                    style={{ background: BAND_COLOR[card.band] }}
                    initial={{ width: 0 }}
                    animate={{ width: `${card.p * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>
            ))}
            {diagnosis.status === "ambiguous" && (
              <p className="font-ui text-lg text-slate italic">Code alone can't separate these. One question will.</p>
            )}
          </div>
        )}

        <h2 className="font-display text-xs text-warp-cyan mb-3">Evidence</h2>
        <ul className="space-y-2 mb-8">
          {diagnosis.evidence.map((item, i) => (
            <li key={i} className="font-ui text-lg flex gap-2">
              <span className="shrink-0 rounded border border-slate px-2 text-sm text-slate">{CHIP_LABEL[item.type] ?? item.type}</span>
              <span>{item.text}</span>
            </li>
          ))}
          {diagnosis.evidence.length === 0 && <li className="font-ui text-lg text-slate">No execution evidence yet.</li>}
        </ul>

        <button
          type="button"
          onClick={proceed}
          className="font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
        >
          {diagnosis.status === "ambiguous" ? "Answer a probe →" : "Continue →"}
        </button>
      </div>
    </div>
  );
}
