import { motion } from "framer-motion";

interface TrialsHeaderProps {
  currentIndex: number;
  totalTrials: number;
  sector: string;
  difficultyLabel: string;
  onExitClick: () => void;
}

export default function TrialsHeader({
  currentIndex,
  totalTrials,
  sector,
  difficultyLabel,
  onExitClick,
}: TrialsHeaderProps) {
  const sectorColors: Record<string, string> = {
    arrays: "bg-cyan-950/80 text-warp-cyan border-warp-cyan/50",
    searching: "bg-blue-950/80 text-blue-400 border-blue-400/50",
    sorting: "bg-indigo-950/80 text-indigo-300 border-indigo-400/50",
    strings: "bg-purple-950/80 text-purple-300 border-purple-400/50",
    recursion: "bg-fuchsia-950/80 text-fuchsia-300 border-fuchsia-400/50",
  };

  const difficultyColors: Record<string, string> = {
    Easy: "bg-emerald-950/80 text-emerald-400 border-emerald-400/50",
    Medium: "bg-amber-950/80 text-amber-400 border-amber-400/50",
    Hard: "bg-rose-950/80 text-rose-400 border-rose-400/50",
  };

  return (
    <header className="flex flex-col gap-2 p-3 bg-deep-space/90 border-b border-warp-cyan/20 backdrop-blur-md select-none">
      {/* Top Global Status Row */}
      <div className="flex items-center justify-between">
        {/* Exit Trial Button */}
        <motion.button
          type="button"
          onClick={onExitClick}
          whileHover={{ scale: 1.04, boxShadow: "0 0 15px rgba(0, 240, 255, 0.4)" }}
          whileTap={{ scale: 0.96 }}
          className="flex items-center gap-2 px-4 py-1.5 rounded-lg border border-warp-cyan/40 bg-void/80 text-white font-ui text-xs font-semibold hover:border-warp-cyan hover:text-warp-cyan transition-colors cursor-pointer"
        >
          <span>←</span>
          <span>Exit Trial</span>
        </motion.button>

        {/* Player Stats HUD */}
        <div className="flex items-center gap-4 text-xs font-mono">
          {/* Level & XP */}
          <div className="flex items-center gap-2 px-3 py-1 rounded bg-black/40 border border-white/10">
            <span className="text-amber-400 font-bold">Lv 12</span>
            <div className="w-24 h-2 bg-white/10 rounded-full overflow-hidden">
              <div className="w-full h-full bg-gradient-to-r from-warp-cyan to-neon-purple" />
            </div>
            <span className="text-white/60 text-[10px]">1,200 / 1,200 XP</span>
          </div>

          {/* Streak */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-black/40 border border-white/10 text-amber-400">
            <span>🔥</span>
            <span className="font-bold">28</span>
            <span className="text-[10px] text-white/50">STREAK</span>
          </div>

          {/* Credits */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-black/40 border border-white/10 text-amber-300">
            <span className="w-3.5 h-3.5 rounded-full bg-amber-400/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">
              $
            </span>
            <span className="font-bold">3,560</span>
            <span className="text-[10px] text-white/50">CREDITS</span>
          </div>

          {/* Offline Replay Pill */}
          <div className="px-3 py-1 rounded bg-blue-900/30 border border-blue-500/40 text-blue-300 text-[11px] font-display flex items-center gap-1.5">
            <span>📼</span>
            <span>OFFLINE REPLAY</span>
          </div>
        </div>
      </div>

      {/* Sub-Header: Mission Banner & Stepper */}
      <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-gradient-to-r from-deep-space via-black/60 to-deep-space border border-warp-cyan/30 shadow-[0_0_20px_rgba(0,240,255,0.08)]">
        {/* Title & Trial Index */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-warp-cyan via-purple-600 to-pink-500 p-0.5 animate-spin-slow">
            <div className="w-full h-full rounded-full bg-black flex items-center justify-center">
              <div className="w-2.5 h-2.5 rounded-full bg-warp-cyan shadow-[0_0_8px_#00f0ff]" />
            </div>
          </div>
          <div>
            <h1 className="font-display text-sm tracking-wider text-white font-bold uppercase drop-shadow-[0_0_10px_rgba(0,240,255,0.5)]">
              DEEP SPACE TRIALS
            </h1>
            <div className="text-xs font-mono text-warp-cyan font-bold tracking-widest">
              TRIAL {currentIndex + 1} / {totalTrials}
            </div>
          </div>
        </div>

        {/* 5-Node Glowing Stepper */}
        <div className="flex items-center gap-3 px-6">
          {Array.from({ length: totalTrials }).map((_, idx) => {
            const isPassed = idx < currentIndex;
            const isCurrent = idx === currentIndex;

            return (
              <div key={idx} className="flex items-center">
                {/* Node Circle */}
                <motion.div
                  className={`w-7 h-7 rounded-full flex items-center justify-center font-mono text-xs transition-all ${
                    isPassed
                      ? "bg-warp-cyan text-black shadow-[0_0_12px_#00f0ff] font-bold"
                      : isCurrent
                      ? "border-2 border-warp-cyan bg-deep-space text-warp-cyan shadow-[0_0_16px_rgba(0,240,255,0.8)] font-bold animate-pulse"
                      : "border border-white/20 bg-black/60 text-white/40"
                  }`}
                  whileHover={{ scale: 1.15 }}
                >
                  {isPassed ? "★" : isCurrent ? "✦" : idx + 1}
                </motion.div>

                {/* Connecting Rail Line */}
                {idx < totalTrials - 1 && (
                  <div
                    className={`w-8 h-0.5 transition-colors ${
                      idx < currentIndex
                        ? "bg-gradient-to-r from-warp-cyan to-warp-cyan shadow-[0_0_8px_#00f0ff]"
                        : "bg-white/10"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Sector & Difficulty Pills */}
        <div className="flex items-center gap-2">
          <span
            className={`px-3 py-1 rounded-full text-xs font-mono uppercase tracking-wider border capitalize ${
              sectorColors[sector.toLowerCase()] || "bg-cyan-950/80 text-warp-cyan border-warp-cyan/50"
            }`}
          >
            {sector}
          </span>
          <span
            className={`px-3 py-1 rounded-full text-xs font-mono font-bold uppercase tracking-wider border ${
              difficultyColors[difficultyLabel] || "bg-amber-950/80 text-amber-400 border-amber-400/50"
            }`}
          >
            {difficultyLabel}
          </span>
        </div>
      </div>
    </header>
  );
}
