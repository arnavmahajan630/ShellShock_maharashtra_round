import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { cpp } from "@codemirror/lang-cpp";
import CodeMirror from "@uiw/react-codemirror";
import { api, type Problem, type TrapItem } from "../lib/api";
import { useMissionStore } from "../store/missionStore";
import SignalGateWorld from "../components/SignalGateWorld";
import { getPlanetConfig } from "../planets";
import DeadEnd from "../components/DeadEnd";

export default function TransferTrapScreen() {
  const navigate = useNavigate();
  const { planet: planetSlug } = useParams<{ planet: string }>();
  const planet = getPlanetConfig(planetSlug);
  const learnerId = useMissionStore((s) => s.learnerId);
  const problem = useMissionStore((s) => s.problem);
  const attempt = useMissionStore((s) => s.attempt);
  const setTrapPassed = useMissionStore((s) => s.setTrapPassed);
  const setTransferAttempt = useMissionStore((s) => s.setTransferAttempt);
  const setVerdict = useMissionStore((s) => s.setVerdict);
  const markPassed = useMissionStore((s) => s.markPassed);

  const classId = attempt?.diagnosis?.top[0]?.id;
  const [phase, setPhase] = useState<"trap" | "transfer" | "running">("trap");
  const [trapItem, setTrapItem] = useState<TrapItem | null>(null);
  const [trapAnswer, setTrapAnswer] = useState<number | null>(null);

  const [transferProblem, setTransferProblem] = useState<Problem | null>(null);
  const [code, setCode] = useState("");
  const [running, setRunning] = useState(false);
  const [lastAttempt, setLastAttempt] = useState<Awaited<ReturnType<typeof api.attempt>> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  async function loadTransferProblem() {
    const nextId = planet!.nextProblem[problem!.problem_id];
    const next = await api.getProblem(nextId);
    setTransferProblem(next);
    setCode(next.starter);
    setPhase("transfer");
  }

  useEffect(() => {
    if (!classId) return;
    setLoadError(null);
    api
      .trapItem(classId)
      .then(setTrapItem)
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Couldn't reach the backend.";
        // The trained diagnoser can label an attempt with a class the hand-authored trap
        // bank has no question for (e.g. "OTHER", the catch-all for an unrecognized bug).
        // That's not a backend outage — skip straight to the transfer problem instead of
        // showing a hard error for something that was never going to have content.
        if (message.includes("404")) {
          loadTransferProblem().catch((err2) =>
            setLoadError(err2 instanceof Error ? err2.message : "Couldn't reach the backend.")
          );
          return;
        }
        setLoadError(message);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classId]);

  if (!problem || !classId || !planet) {
    return <DeadEnd message="No intervention in progress." to={planetSlug ? `/planet/${planetSlug}` : undefined} />;
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space">
        <p className="font-ui text-2xl text-alert-red">{loadError}</p>
      </div>
    );
  }

  async function answerTrap(index: number) {
    setTrapAnswer(index);
    const correct = index === trapItem!.correct;
    setTrapPassed(correct);
    try {
      await api.reassess({ learner_id: learnerId, class: classId!, item_id: `trap_${classId}`, item_type: "trap", result: { correct } });
      await loadTransferProblem();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't reach the backend.");
    }
  }

  async function runTransfer() {
    setRunning(true);
    setPhase("running");
    try {
      const result = await api.attempt({ learner_id: learnerId, problem_id: transferProblem!.problem_id, code });
      setLastAttempt(result);
      setTransferAttempt(result);
      if (result.gate.code === "G0" && result.tests && result.tests.passed === result.tests.total) {
        markPassed(transferProblem!.problem_id);
      }
      const verdict = await api.reassess({
        learner_id: learnerId,
        class: classId!,
        item_id: transferProblem!.problem_id,
        item_type: "transfer_code",
        result: { tests: result.tests },
      });
      setVerdict(verdict);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't reach the backend.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="min-h-screen bg-deep-space px-6 py-10 text-light">
      <div className="mx-auto max-w-2xl rounded-lg border border-void-blue bg-void-blue/30 p-8">
        {phase === "trap" && trapItem && (
          <>
            <h1 className="font-display text-sm text-starlight mb-4">Trap — no running this one</h1>
            <pre className="font-mono text-base bg-deep-space/60 border border-void-blue rounded p-3 whitespace-pre-wrap">{trapItem.code}</pre>
            <p className="font-ui text-xl mt-4">{trapItem.prompt}</p>
            <div className="flex flex-col gap-2 mt-4">
              {trapItem.options.map((opt, i) => (
                <button
                  key={opt}
                  type="button"
                  disabled={trapAnswer !== null}
                  onClick={() => answerTrap(i)}
                  className={
                    "font-ui text-xl text-left px-4 py-2 rounded border " +
                    (trapAnswer === null
                      ? "border-slate hover:border-warp-cyan"
                      : i === trapItem.correct
                        ? "border-mint-success text-mint-success"
                        : i === trapAnswer
                          ? "border-alert-red text-alert-red"
                          : "border-slate text-slate")
                  }
                >
                  {opt}
                </button>
              ))}
            </div>
          </>
        )}

        {(phase === "transfer" || phase === "running") && transferProblem && (
          <>
            <h1 className="font-display text-sm text-mint-success mb-2">Transfer — {transferProblem.name}</h1>
            <p className="font-ui text-lg text-slate mb-4">{transferProblem.prompt}</p>
            <div className="rounded-lg border border-void-blue overflow-hidden">
              <CodeMirror value={code} height="240px" theme="dark" extensions={[cpp()]} onChange={setCode} readOnly={running} />
            </div>
            {phase === "transfer" && (
              <button
                type="button"
                onClick={runTransfer}
                className="mt-4 font-ui text-xl px-6 py-2 rounded bg-mint-success text-deep-space hover:brightness-110 active:scale-95"
              >
                ▶ Run
              </button>
            )}
            {phase === "running" && (
              <div className="mt-6">
                <SignalGateWorld
                  label={transferProblem.name}
                  status={running ? "running" : lastAttempt?.tests && lastAttempt.tests.passed === lastAttempt.tests.total ? "pass" : "fail"}
                  passed={lastAttempt?.tests?.passed ?? 0}
                  total={lastAttempt?.tests?.total ?? 0}
                />
                {!running && (
                  <button
                    type="button"
                    onClick={() => navigate(`/planet/${planetSlug}/verdict`)}
                    className="mt-6 font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
                  >
                    See verdict →
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
