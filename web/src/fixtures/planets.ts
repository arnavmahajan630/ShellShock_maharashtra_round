export type PlanetStatus = "unlocked" | "locked" | "always-open";

export interface Planet {
  id: string;
  name: string;
  subtitle: string;
  status: PlanetStatus;
  /** Center position as a percentage of the map art's bounding box. */
  position: { top: number; left: number };
  /** Diameter as a percentage of the map art's width. */
  size: number;
}

export const PLANETS: Planet[] = [
  {
    id: "aegis-grid",
    name: "Aegis Grid",
    subtitle: "Conditions",
    status: "unlocked",
    position: { top: 31.05, left: 27.21 },
    size: 15.62,
  },
  {
    id: "miners-belt",
    name: "Miner's Belt",
    subtitle: "Loops",
    status: "unlocked",
    position: { top: 31.05, left: 50.0 },
    size: 15.62,
  },
  {
    id: "lost-fleet",
    name: "Lost Fleet",
    subtitle: "Arrays",
    status: "unlocked",
    position: { top: 31.05, left: 72.79 },
    size: 15.62,
  },
  {
    id: "nav-core",
    name: "Nav Core",
    subtitle: "Variables",
    status: "unlocked",
    position: { top: 63.48, left: 37.63 },
    size: 13.67,
  },
  {
    id: "module-deck",
    name: "Module Deck",
    subtitle: "Functions",
    status: "unlocked",
    position: { top: 63.48, left: 59.38 },
    size: 13.67,
  },
  {
    id: "deep-space-trials",
    name: "Deep Space Trials",
    subtitle: "Adaptive DSA Exam",
    status: "always-open",
    position: { top: 68.36, left: 89.71 },
    size: 16.93,
  },
];
