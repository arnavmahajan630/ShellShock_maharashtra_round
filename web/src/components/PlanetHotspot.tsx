import { motion } from "framer-motion";
import type { Planet } from "../fixtures/planets";

const GLOW_BY_STATUS: Record<Planet["status"], string> = {
  unlocked: "var(--color-warp-cyan)",
  "always-open": "var(--color-pulsar-magenta)",
  locked: "transparent",
};

interface PlanetHotspotProps {
  planet: Planet;
  index: number;
  onActivate?: () => void;
}

export default function PlanetHotspot({ planet, index, onActivate }: PlanetHotspotProps) {
  const interactive = planet.status !== "locked";
  const glow = GLOW_BY_STATUS[planet.status];

  const style: React.CSSProperties = {
    top: `${planet.position.top}%`,
    left: `${planet.position.left}%`,
    width: `${planet.size}%`,
    aspectRatio: "1 / 1",
  };

  const common = {
    className:
      "absolute -translate-x-1/2 -translate-y-1/2 rounded-full outline-none" +
      (interactive
        ? " cursor-pointer focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
        : " cursor-not-allowed"),
    style,
    initial: { opacity: 0, scale: 0.6 },
    animate: { opacity: 1, scale: 1 },
    transition: { delay: 0.15 + index * 0.08, duration: 0.4, ease: "easeOut" as const },
    "aria-label": interactive ? planet.name : `${planet.name} (locked)`,
    title: interactive ? planet.name : `${planet.name} — locked`,
  };

  if (!interactive) {
    return <motion.div {...common} aria-disabled="true" />;
  }

  return (
    <motion.button
      {...common}
      type="button"
      whileHover={{ scale: 1.08, boxShadow: `0 0 28px 6px ${glow}` }}
      whileTap={{ scale: 0.94 }}
      onClick={onActivate ?? (() => {})}
    />
  );
}
