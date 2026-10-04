import { conditions } from "./conditions";
import { loops } from "./loops";
import type { PlanetConfig } from "./types";

export const PLANETS: Record<string, PlanetConfig> = { conditions, loops };

export function getPlanetConfig(slug: string | undefined): PlanetConfig | null {
  return slug ? (PLANETS[slug] ?? null) : null;
}

export type { PlanetConfig, PathNode, WarmupConfig } from "./types";
