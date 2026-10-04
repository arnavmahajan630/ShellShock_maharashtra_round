import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { audioManager } from "../../audio/audioManager";
import {
  SpeechRecorder,
  isSpeechRecognitionSupported,
  countHedges,
} from "../../lib/speech";

export interface AudioSignals {
  response_ms: number;
  hedge_count: number;
  speech_rate_wps?: number;
  pause_count?: number;
}

interface PushToTalkProps {
  onRecorded: (transcript: string, signals: AudioSignals) => void;
  disabled?: boolean;
}

export default function PushToTalk({ onRecorded, disabled = false }: PushToTalkProps) {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [interimText, setInterimText] = useState<string>("");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [hasMicSupport] = useState<boolean>(isSpeechRecognitionSupported());
  const [audioError, setAudioError] = useState<string | null>(null);

  const recorderRef = useRef<SpeechRecorder | null>(null);
  const startTimeRef = useRef<number>(0);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (recorderRef.current) {
        recorderRef.current.stop();
      }
      audioManager.restore();
    };
  }, []);

  function startRecording() {
    if (disabled || isRecording) return;
    setAudioError(null);
    setInterimText("");
    setElapsedSeconds(0);
    startTimeRef.current = Date.now();

    // Duck background music
    audioManager.duck(0.08);

    if (hasMicSupport) {
      recorderRef.current = new SpeechRecorder({
        onTranscript: (text) => {
          setInterimText(text);
        },
        onError: (err) => {
          setAudioError(`Mic error: ${err}`);
          stopRecording();
        },
        onEnd: () => {
          setIsRecording(false);
          audioManager.restore();
        },
      });
      recorderRef.current.start();
    }

    setIsRecording(true);

    timerIntervalRef.current = setInterval(() => {
      setElapsedSeconds((s) => {
        if (s >= 29) {
          stopRecording();
          return 30;
        }
        return s + 1;
      });
    }, 1000);
  }

  function stopRecording() {
    if (!isRecording) return;
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    let finalTranscript = interimText;
    let durationMs = Date.now() - startTimeRef.current;
    let wps = 0;

    if (recorderRef.current) {
      const result = recorderRef.current.stop();
      if (result.transcript) finalTranscript = result.transcript;
      durationMs = result.durationMs;
      wps = result.wordsPerSec;
    }

    setIsRecording(false);
    audioManager.restore();

    const hedges = countHedges(finalTranscript);
    const signals: AudioSignals = {
      response_ms: Math.max(500, durationMs),
      hedge_count: hedges,
      speech_rate_wps: wps > 0 ? wps : undefined,
      pause_count: 0,
    };

    onRecorded(finalTranscript, signals);
  }

  // Keyboard Spacebar hold support (only when not typing in an input/textarea)
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (disabled || e.repeat) return;
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        startRecording();
      }
    }

    function handleKeyUp(e: KeyboardEvent) {
      if (disabled) return;
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }
      if (e.code === "Space" && isRecording) {
        e.preventDefault();
        stopRecording();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [disabled, isRecording, interimText]);

  return (
    <div className="flex flex-col items-center justify-center p-6 rounded-lg border border-void-blue bg-void-blue/40">
      {!hasMicSupport && (
        <div className="mb-4 rounded border border-starlight/40 bg-starlight/10 p-3 font-ui text-xl text-starlight">
          ℹ Web Speech API not detected in this browser. You can still type your explanation below.
        </div>
      )}

      {audioError && (
        <div className="mb-4 rounded border border-alert-red/40 bg-alert-red/10 p-2.5 font-ui text-xl text-alert-red">
          {audioError}
        </div>
      )}

      {/* Recording Waveform Visualization */}
      <div className="h-16 flex items-center justify-center gap-1.5 mb-4">
        {isRecording ? (
          Array.from({ length: 16 }).map((_, i) => (
            <motion.div
              key={i}
              className={`w-1.5 rounded-full ${i % 2 === 0 ? "bg-starlight" : "bg-warp-cyan"}`}
              animate={{
                height: [
                  12,
                  Math.max(16, ((i * 7) % 48) + Math.random() * 24),
                  8,
                ],
              }}
              transition={{
                repeat: Infinity,
                duration: 0.4 + (i % 3) * 0.15,
                ease: "easeInOut",
              }}
            />
          ))
        ) : (
          <div className="flex items-center gap-1.5 opacity-30">
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="w-1.5 h-3 rounded-full bg-slate" />
            ))}
          </div>
        )}
      </div>

      {/* Main Push to Talk Button */}
      <motion.button
        type="button"
        disabled={disabled}
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onTouchStart={startRecording}
        onTouchEnd={stopRecording}
        whileTap={{ scale: 0.95 }}
        className={`relative flex items-center justify-center h-22 w-22 rounded-full border-2 transition-all cursor-pointer ${
          isRecording
            ? "border-alert-red bg-alert-red/20 text-alert-red shadow-[0_0_35px_rgba(250,81,85,0.4)]"
            : "border-starlight bg-deep-space text-starlight hover:bg-starlight/10 hover:shadow-[0_0_25px_rgba(252,177,67,0.35)]"
        } disabled:opacity-40 disabled:pointer-events-none`}
      >
        {isRecording ? (
          <div className="flex flex-col items-center">
            <span className="h-4 w-4 rounded-sm bg-alert-red animate-pulse mb-1" />
            <span className="font-ui text-xl font-bold text-alert-red">
              {30 - elapsedSeconds}s
            </span>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <svg
              className="w-7 h-7 mb-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.8}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
            <span className="font-display text-[9px] uppercase tracking-wider font-semibold">
              Hold Talk
            </span>
          </div>
        )}
      </motion.button>

      {/* Instructions */}
      <div className="mt-4 text-center">
        <p className="font-ui text-xl text-slate">
          {isRecording ? (
            <span className="text-alert-red font-bold animate-pulse">
              ● Recording... Release when finished speaking
            </span>
          ) : (
            <span>
              Hold button or press & hold <kbd className="px-2 py-0.5 rounded bg-deep-space border border-void-blue text-starlight font-ui text-lg">SPACE</kbd> to talk
            </span>
          )}
        </p>
      </div>

      {/* Interim Transcript display */}
      <AnimatePresence>
        {isRecording && interimText && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 max-w-md text-center rounded border border-void-blue bg-deep-space/80 px-4 py-2 font-ui text-2xl text-warp-cyan italic"
          >
            "{interimText}"
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
