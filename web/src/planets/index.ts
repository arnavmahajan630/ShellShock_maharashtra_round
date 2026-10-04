import { arrays } from "./arrays";
import { conditions } from "./conditions";
import { functions } from "./functions";
import { loops } from "./loops";
import { variables } from "./variables";
import type { PlanetConfig } from "./types";

export const PLANETS: Record<string, PlanetConfig> = { conditions, loops, arrays, variables, functions };

export function getPlanetConfig(slug: string | undefined): PlanetConfig | null {
  return slug ? (PLANETS[slug] ?? null) : null;
}

export type { PlanetConfig, PathNode, WarmupConfig } from "./types";
