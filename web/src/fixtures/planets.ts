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
    position: { top: 28.6, left: 29.8 },
    size: 9,
  },
  {
    id: "miners-belt",
    name: "Miner's Belt",
    subtitle: "Loops",
    status: "unlocked",
    position: { top: 27.1, left: 52.1 },
    size: 9,
  },
  {
    id: "lost-fleet",
    name: "Lost Fleet",
    subtitle: "Arrays",
    status: "unlocked",
    position: { top: 29.0, left: 74.2 },
    size: 9,
  },
  {
    id: "nav-core",
    name: "Nav Core",
    subtitle: "Variables",
    status: "locked",
    position: { top: 63.1, left: 39.3 },
    size: 8,
  },
  {
    id: "module-deck",
    name: "Module Deck",
    subtitle: "Functions",
    status: "locked",
    position: { top: 64.5, left: 61.4 },
    size: 8,
  },
  {
    id: "deep-space-trials",
    name: "Deep Space Trials",
    subtitle: "Adaptive DSA Exam",
    status: "always-open",
    position: { top: 70.4, left: 90.5 },
    size: 11,
  },
];
