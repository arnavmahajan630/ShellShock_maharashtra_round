import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { cpp } from "@codemirror/lang-cpp";
import CodeMirror from "@uiw/react-codemirror";
import { api, type Problem, type MmProblem } from "../../lib/api";
import { useMissionStore } from "../../store/missionStore";
import PushToTalk, { type AudioSignals } from "../../components/mm/PushToTalk";
import TranscriptConfirm from "../../components/mm/TranscriptConfirm";
import IterationDial from "../../components/mm/IterationDial";
import SignalGateWorld from "../../components/SignalGateWorld";

interface VoiceTraceMissionProps {
  problem: Problem;
}

export default function VoiceTraceMission({ problem }: VoiceTraceMissionProps) {
  const { planet } = useParams<{ planet: string }>();
  const navigate = useNavigate();
  const learnerId = useMissionStore((s) => s.learnerId);
  const setAttempt = useMissionStore((s) => s.setAttempt);
  const markPassed = useMissionStore((s) => s.markPassed);

  const [mmProblem, setMmProblem] = useState<MmProblem | null>(null);
  const [recordedTranscript, setRecordedTranscript] = useState<string>("");
  const [audioSignals, setAudioSignals] = useState<AudioSignals>({
    response_ms: 1000,
    hedge_count: 0,
  });
  const [hasRecorded, setHasRecorded] = useState<boolean>(false);
  const [manualTextMode, setManualTextMode] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [attemptResult, setAttemptResult] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    api
      .getMmProblem(problem.problem_id)
      .then((data) => {
        if (isMounted) setMmProblem(data);
      })
      .catch(() => {
        // Fallback problem data
        if (isMounted) {
          setMmProblem({
            ...problem,
            type: "voice_trace",
            story: problem.prompt,
            inputs: { n: 3 },
            code_equivalent: problem.starter,
            correct_output: "4",
            rubric: [],
            canonical_explanation: "",
          });
        }
      });
    return () => {
      isMounted = false;
    };
  }, [problem.problem_id]);

  function handleRecorded(transcript: string, signals: AudioSignals) {
    setRecordedTranscript(transcript);
    setAudioSignals(signals);
    setHasRecorded(true);
  }

  async function handleTranscriptSubmit(
    transcript: string,
    edited: boolean,
    signals: AudioSignals
  ) {
    setSubmitting(true);
    setErrorMsg(null);

    try {
      const res = await api.mmAttempt({
        learner_id: learnerId,
        problem_id: problem.problem_id,
        modality: "voice",
        response: {
          transcript,
          transcript_edited: edited,
          explanation_text: transcript,
        },
        signals: {
          response_ms: signals.response_ms,
          hedge_count: signals.hedge_count,
          speech_rate_wps: signals.speech_rate_wps,
          pause_count: signals.pause_count,
        },
      });

      setAttempt(res);
      setAttemptResult(res);

      if (res.gate.code === "G0" && res.tests && res.tests.passed === res.tests.total) {
        markPassed(problem.problem_id);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to analyze speech attempt.");
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

  // Parse claimed vs actual iteration counts from response evidence if available
  const claimedIteration = attemptResult?.mm_debug?.evidence?.claims?.find(
    (c: any) => c.kind === "iteration_count"
  )?.value ? parseInt(attemptResult.mm_debug.evidence.claims.find((c: any) => c.kind === "iteration_count").value, 10) : null;

  return (
    <div className="min-h-screen bg-deep-space px-6 py-8 text-light">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-void-blue pb-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-1 rounded font-display text-[10px] bg-void-blue/80 text-starlight border border-starlight/50 uppercase tracking-wider">
                Multimodal Level B · Voice Loop Trace
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

        {/* Narrative & Task Prompt */}
        <div className="rounded-lg border border-void-blue bg-void-blue/40 p-5 backdrop-blur-sm">
          <p className="font-ui text-xl leading-relaxed text-light">
            {mmProblem?.story || problem.prompt}
          </p>
          <div className="mt-3 flex items-center gap-2 font-ui text-xl text-starlight flex-wrap">
            <span className="text-slate">SEARCH TARGET:</span>
            <span className="px-3 py-0.5 rounded bg-deep-space/80 border border-void-blue font-bold text-starlight">
              100 unsorted cards (random order) → explain algorithm to find Commander Rahul
            </span>
          </div>
        </div>

        {/* Code Block Viewer */}
        <div className="rounded-lg border border-void-blue overflow-hidden bg-deep-space/60 shadow-xl">
          <div className="flex items-center justify-between bg-void-blue/60 px-4 py-2 border-b border-void-blue font-ui text-xl text-slate">
            <span>C Routine Telemetry</span>
            <span className="text-starlight font-bold">Read & Trace</span>
          </div>
          <CodeMirror
            value={problem.starter}
            height="180px"
            theme="dark"
            extensions={[cpp()]}
            readOnly={true}
            basicSetup={{ lineNumbers: true, foldGutter: false }}
          />
        </div>

        {/* Voice Recording / Input Section */}
        {!attemptResult && (
          <div className="space-y-4">
            {!hasRecorded && !manualTextMode && (
              <div className="space-y-3">
                <PushToTalk onRecorded={handleRecorded} disabled={submitting} />
                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => setManualTextMode(true)}
                    className="font-ui text-xl text-slate hover:text-starlight underline transition-colors cursor-pointer"
                  >
                    Or type your explanation manually
                  </button>
                </div>
              </div>
            )}

            {(hasRecorded || manualTextMode) && (
              <TranscriptConfirm
                initialTranscript={recordedTranscript}
                signals={audioSignals}
                onSubmit={handleTranscriptSubmit}
                onRerecord={() => {
                  setHasRecorded(false);
                  setManualTextMode(false);
                  setRecordedTranscript("");
                }}
                isSubmitting={submitting}
              />
            )}

            {errorMsg && (
              <div className="rounded border border-alert-red bg-alert-red/10 p-3 font-ui text-xl text-alert-red">
                {errorMsg}
              </div>
            )}
          </div>
        )}

        {/* Attempt Evaluation & Iteration Dial */}
        {attemptResult && (
          <div className="space-y-6">
            <SignalGateWorld
              label={problem.name}
              status={attemptResult.tests?.passed === attemptResult.tests?.total ? "pass" : "fail"}
              passed={attemptResult.tests?.passed ?? 0}
              total={attemptResult.tests?.total ?? 1}
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <IterationDial
                claimedCount={claimedIteration}
                actualCount={mmProblem?.expected_iterations || 100}
                maxCount={100}
                label="Card Scan Iteration Gauge"
              />

              <div className="rounded-lg border border-void-blue bg-void-blue/40 p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="h-2 w-2 rounded-full bg-warp-cyan animate-pulse" />
                    <h4 className="font-display text-xs text-warp-cyan uppercase tracking-wider">
                      Droid Analysis Report
                    </h4>
                  </div>
                  <p className="font-ui text-xl text-light leading-relaxed">
                    {attemptResult.diagnosis?.evidence?.[0]?.text ||
                      "Evaluated execution against harness truth. The 100 astronaut cards are arranged randomly, requiring a sequential linear search from index 0 to 99 until Commander Rahul is located."}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-void-blue flex items-center justify-between font-ui text-lg text-slate">
                  <span>Modality: Voice Transcription</span>
                  <span className="text-starlight">Engine: Gemini Droid Judge</span>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={handleContinue}
                className="font-ui text-2xl px-8 py-2.5 rounded bg-starlight text-deep-space font-bold shadow-[0_0_20px_rgba(252,177,67,0.4)] hover:brightness-110 active:scale-95 transition-all"
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
