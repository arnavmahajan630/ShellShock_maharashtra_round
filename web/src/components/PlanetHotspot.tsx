import { motion } from "framer-motion";
import type { Planet } from "../fixtures/planets";

interface PlanetHotspotProps {
  planet: Planet;
  index: number;
  onActivate?: () => void;
}

export default function PlanetHotspot({ planet, index, onActivate }: PlanetHotspotProps) {
  const interactive = planet.status !== "locked";
  const isBlackHole = planet.status === "always-open";
  const glowShadow = isBlackHole
    ? "rgba(217, 70, 239, 0.7)"
    : "rgba(6, 182, 212, 0.7)";
  const ringColor = isBlackHole ? "border-fuchsia-400" : "border-cyan-400";

  const style: React.CSSProperties = {
    top: `${planet.position.top}%`,
    left: `${planet.position.left}%`,
    width: `${planet.size}%`,
    aspectRatio: "1 / 1",
  };

  return (
    <motion.button
      type="button"
      style={style}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.1 + index * 0.06, duration: 0.4, ease: "easeOut" }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.96 }}
      onClick={interactive ? onActivate : undefined}
      className={`group absolute -translate-x-1/2 -translate-y-1/2 rounded-full outline-none select-none transition-all duration-300 ${
        interactive ? "cursor-pointer focus-visible:ring-4 focus-visible:ring-cyan-400/70" : "cursor-not-allowed opacity-50"
      }`}
      aria-label={interactive ? planet.name : `${planet.name} (locked)`}
      title={interactive ? planet.name : `${planet.name} — locked`}
    >
      {/* Ambient Orbit Ring */}
      <div
        className={`absolute inset-0 rounded-full border border-dashed transition-all duration-300 ${
          isBlackHole
            ? "border-fuchsia-400/25 group-hover:border-fuchsia-400"
            : "border-cyan-400/20 group-hover:border-cyan-400"
        } opacity-50 group-hover:opacity-100`}
      />

      {/* Luminous Interactive Orbit Reticle on Hover */}
      <div
        style={{
          boxShadow: `0 0 25px ${glowShadow}, inset 0 0 15px ${glowShadow}`,
        }}
        className={`absolute inset-0 rounded-full transition-all duration-300 opacity-0 group-hover:opacity-100 ${ringColor} border-2 backdrop-brightness-110`}
      />

      {/* Sci-Fi Corner Target Brackets on Hover */}
      <div className="absolute -inset-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
        <span className={`absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 ${ringColor}`} />
        <span className={`absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 ${ringColor}`} />
        <span className={`absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 ${ringColor}`} />
        <span className={`absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 ${ringColor}`} />
      </div>

      {/* Floating System HUD Tooltip */}
      <div className="absolute -top-7 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-all duration-200 pointer-events-none whitespace-nowrap z-30">
        <span
          className={`px-2 py-0.5 rounded text-[10px] font-mono tracking-widest uppercase bg-black/90 border ${
            isBlackHole ? "border-fuchsia-400 text-fuchsia-300" : "border-cyan-400 text-cyan-300"
          } shadow-[0_0_10px_rgba(0,0,0,0.8)]`}
        >
          {isBlackHole ? "WARP · SINGULARITY" : `ENTER · ${planet.name}`}
        </span>
      </div>
    </motion.button>
  );
}
