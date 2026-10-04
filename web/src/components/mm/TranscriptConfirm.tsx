import { useState, useEffect } from "react";
import { countHedges } from "../../lib/speech";
import type { AudioSignals } from "./PushToTalk";

interface TranscriptConfirmProps {
  initialTranscript: string;
  signals: AudioSignals;
  onSubmit: (transcript: string, edited: boolean, signals: AudioSignals) => void;
  onRerecord: () => void;
  isSubmitting?: boolean;
}

export default function TranscriptConfirm({
  initialTranscript,
  signals,
  onSubmit,
  onRerecord,
  isSubmitting = false,
}: TranscriptConfirmProps) {
  const [transcript, setTranscript] = useState<string>(initialTranscript);
  const [edited, setEdited] = useState<boolean>(false);

  useEffect(() => {
    setTranscript(initialTranscript);
    setEdited(false);
  }, [initialTranscript]);

  const words = transcript.trim().split(/\s+/).filter(Boolean).length;
  const currentHedges = countHedges(transcript);

  function handleChange(val: string) {
    setTranscript(val);
    setEdited(val.trim() !== initialTranscript.trim());
  }

  function handleConfirm() {
    if (!transcript.trim()) return;
    const finalSignals: AudioSignals = {
      ...signals,
      hedge_count: currentHedges,
    };
    onSubmit(transcript.trim(), edited, finalSignals);
  }

  return (
    <div className="rounded-lg border border-void-blue bg-void-blue/40 p-6">
      <div className="flex items-center justify-between border-b border-void-blue pb-3 mb-4">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-starlight animate-pulse" />
          <h3 className="font-display text-xs text-starlight uppercase tracking-wider">
            Confirm Droid Telemetry · Voice Transcript
          </h3>
        </div>
        <div className="flex items-center gap-3 font-ui text-xl text-slate">
          <span>Words: {words}</span>
          {currentHedges > 0 && (
            <span className="rounded bg-starlight/20 px-3 py-0.5 text-starlight font-bold border border-starlight/40 font-ui text-xl">
              {currentHedges} hedge {currentHedges === 1 ? "marker" : "markers"}
            </span>
          )}
        </div>
      </div>

      <p className="font-ui text-xl text-slate mb-3">
        Review your voice transcript below. You can fix any voice recognition errors before submitting to the droid analyzer:
      </p>

      <textarea
        value={transcript}
        onChange={(e) => handleChange(e.target.value)}
        rows={3}
        placeholder="Explain how you would find Commander Rahul's card..."
        disabled={isSubmitting}
        className="w-full rounded border border-void-blue bg-deep-space/80 p-3 font-ui text-2xl text-light placeholder-slate/40 focus:border-starlight focus:outline-none disabled:opacity-50"
      />

      {edited && (
        <p className="mt-1 font-ui text-lg text-warp-cyan">
          ✎ Edited manually
        </p>
      )}

      {/* Action buttons */}
      <div className="mt-4 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={onRerecord}
          disabled={isSubmitting}
          className="rounded font-ui text-xl border border-void-blue bg-deep-space px-5 py-2 text-slate hover:border-warp-cyan hover:text-light disabled:opacity-40 transition-colors cursor-pointer"
        >
          ↺ Record Again
        </button>

        <button
          type="button"
          onClick={handleConfirm}
          disabled={!transcript.trim() || isSubmitting}
          className="rounded font-ui text-2xl font-bold bg-starlight text-deep-space px-8 py-2.5 shadow-[0_0_20px_rgba(252,177,67,0.4)] hover:brightness-110 active:scale-95 disabled:opacity-40 disabled:pointer-events-none transition-all cursor-pointer"
        >
          {isSubmitting ? "Analyzing Explanation..." : "Lock In & Transmit →"}
        </button>
      </div>
    </div>
  );
}
