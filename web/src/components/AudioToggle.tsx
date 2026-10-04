import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAudio } from "../audio/AudioContext";

export default function AudioToggle() {
  const { isMuted, isPlaying, activeTrack, toggleMute } = useAudio();
  const [isHovered, setIsHovered] = useState(false);

  // Global 'M' shortcut to toggle audio, provided user is not typing in an editor/input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "m" || e.key === "M") {
        const target = e.target as HTMLElement | null;
        if (
          target &&
          (target.tagName === "INPUT" ||
            target.tagName === "TEXTAREA" ||
            target.isContentEditable ||
            target.closest(".cm-editor"))
        ) {
          return;
        }
        toggleMute();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleMute]);

  // Audio status label
  const isOff = isMuted;
  const labelText = !isMuted && isPlaying
    ? activeTrack === "blackhole"
      ? "SINGULARITY THEME"
      : "GALAXY AMBIENCE"
    : "AUDIO MUTED";

  return (
    <div
      className="fixed top-4 right-4 z-50 flex items-center select-none font-ui"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, x: 10, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 6, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="mr-2 px-2.5 py-1 rounded bg-black/85 border border-white/20 text-[11px] font-mono tracking-widest text-cyan-300 shadow-[0_0_12px_rgba(0,0,0,0.8)] backdrop-blur-md pointer-events-none whitespace-nowrap flex items-center gap-1.5"
          >
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full ${
                !isOff ? "bg-cyan-400 shadow-[0_0_6px_#22d3ee] animate-pulse" : "bg-amber-500/80"
              }`}
            />
            {labelText}
            <span className="text-white/40 text-[9px] ml-1">[M]</span>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={toggleMute}
        aria-label="Toggle Background Music"
        className={`group relative flex items-center justify-center p-2.5 rounded-full backdrop-blur-md transition-all duration-300 ${
          !isOff
            ? "bg-black/60 border border-cyan-500/40 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.25)] hover:border-cyan-400 hover:shadow-[0_0_20px_rgba(6,182,212,0.5)]"
            : "bg-black/60 border border-white/15 text-white/50 hover:text-white/80 hover:border-white/30 shadow-[0_0_10px_rgba(0,0,0,0.5)]"
        }`}
      >
        {/* Subtle Ambient Pulse Ring when Playing */}
        {!isOff && (
          <span className="absolute inset-0 rounded-full border border-cyan-400/30 animate-ping opacity-40 pointer-events-none" />
        )}

        <div className="flex items-center gap-1.5">
          {/* Speaker Icon */}
          <svg
            className={`w-4 h-4 transition-transform duration-200 group-hover:scale-110 ${
              !isOff ? "text-cyan-400" : "text-slate-400"
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            {!isOff ? (
              // Speaker with Sound Waves
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.536 8.464a5 5 0 010 7.072M18.364 5.636a9 9 0 010 12.728M11 5L6 9H2v6h4l5 4V5z"
              />
            ) : (
              // Speaker Muted with Slash
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M5.586 15H2V9h4l5-4v5.586M11 19l-2-2m-3-3L1 1m22 22L12 12m9-3.536a9 9 0 01-2.636 6.364m-3.828-3.828a5 5 0 00-1.414-2.536"
              />
            )}
          </svg>

          {/* Equalizer Visualizer Bars (Only animated when playing) */}
          {!isOff ? (
            <div className="flex items-end gap-0.5 h-3.5 w-3.5">
              <motion.span
                className="w-0.5 bg-cyan-400 rounded-full"
                animate={{ height: ["3px", "12px", "5px", "14px", "3px"] }}
                transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.span
                className="w-0.5 bg-cyan-300 rounded-full"
                animate={{ height: ["8px", "4px", "14px", "6px", "8px"] }}
                transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
              />
              <motion.span
                className="w-0.5 bg-cyan-400 rounded-full"
                animate={{ height: ["4px", "14px", "6px", "10px", "4px"] }}
                transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut", delay: 0.1 }}
              />
            </div>
          ) : (
            <span className="text-[10px] font-mono tracking-tighter text-white/40">OFF</span>
          )}
        </div>
      </button>
    </div>
  );
}
