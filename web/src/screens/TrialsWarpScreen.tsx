import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

export default function TrialsWarpScreen() {
  const navigate = useNavigate();
  const [timeLeft, setTimeLeft] = useState(3.5);
  const totalTime = 3.5;

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 0.1) {
          clearInterval(interval);
          navigate("/trials");
          return 0;
        }
        return Math.max(0, +(prev - 0.1).toFixed(1));
      });
    }, 100);

    return () => clearInterval(interval);
  }, [navigate]);

  const progress = Math.min(100, Math.round(((totalTime - timeLeft) / totalTime) * 100));

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black flex items-center justify-center select-none font-ui">
      {/* Background Warp Image */}
      <img
        src="/images/trials_warp.png"
        alt="Warping to Deep Space Trials"
        className="absolute inset-0 w-full h-full object-cover object-center pointer-events-none"
      />

      {/* Ambient Pulsing Dark Vignette */}
      <div className="absolute inset-0 bg-radial-gradient from-transparent via-black/20 to-black/60 pointer-events-none" />

      {/* Center Cinematic Loading Overlay */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 flex flex-col items-center justify-center p-8 text-center max-w-lg"
      >
        {/* Glowing Portal HUD Ring */}
        <div className="relative w-36 h-36 flex items-center justify-center mb-6">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="44"
              className="text-white/10"
              strokeWidth="6"
              stroke="currentColor"
              fill="transparent"
            />
            <motion.circle
              cx="50"
              cy="50"
              r="44"
              className="text-warp-cyan"
              strokeWidth="6"
              strokeDasharray="276"
              strokeDashoffset={276 - (276 * progress) / 100}
              strokeLinecap="round"
              stroke="currentColor"
              fill="transparent"
              style={{
                filter: "drop-shadow(0 0 10px rgba(0, 240, 255, 0.8))",
              }}
            />
          </svg>

          {/* Digital Countdown Inside Ring */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-display text-2xl text-white font-bold tracking-wider">
              {timeLeft > 0 ? `${timeLeft.toFixed(1)}s` : "WARP"}
            </span>
            <span className="text-[10px] text-warp-cyan tracking-widest font-mono uppercase">
              {progress}%
            </span>
          </div>
        </div>

        {/* Telemetry Status Text */}
        <motion.div
          animate={{ opacity: [0.7, 1, 0.7] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="space-y-1.5"
        >
          <div className="text-xs font-mono text-warp-cyan uppercase tracking-widest flex items-center justify-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-warp-cyan animate-ping" />
            Stabilizing Singularity Vector
          </div>
          <div className="text-white/60 text-xs font-mono">
            Crossing Event Horizon · Coordinates Locked
          </div>
        </motion.div>

        {/* Linear Progress Bar */}
        <div className="w-64 h-1.5 bg-black/60 border border-warp-cyan/40 rounded-full mt-6 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-warp-cyan via-starlight to-neon-purple shadow-[0_0_12px_#00f0ff]"
            style={{ width: `${progress}%` }}
          />
        </div>
      </motion.div>

      {/* Bottom-Right Skip Button matching warp.png layout */}
      <div className="absolute bottom-6 right-8 z-20">
        <motion.button
          type="button"
          onClick={() => navigate("/trials")}
          whileHover={{ scale: 1.05, boxShadow: "0 0 20px 4px rgba(0, 240, 255, 0.6)" }}
          whileTap={{ scale: 0.95 }}
          className="flex items-center gap-2 px-6 py-2.5 rounded-lg border border-warp-cyan bg-deep-space/80 backdrop-blur-md text-warp-cyan font-display text-sm tracking-wider hover:bg-warp-cyan hover:text-black transition-all cursor-pointer shadow-[0_0_15px_rgba(0,240,255,0.3)]"
        >
          <span>SKIP</span>
          <span className="text-lg">→</span>
        </motion.button>
      </div>
    </div>
  );
}
