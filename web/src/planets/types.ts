export type NodeKind = "briefing" | "warmup" | "mission" | "ghost";

export interface PathNode {
  id: string;
  kind: NodeKind;
  label: string;
  position: { top: number; left: number };
  problemId?: string;
}

export interface WarmupConfig {
  code: string;
  question: string;
  options: string[];
  correct: string;
  explanation: string;
}

export interface PlanetConfig {
  slug: string;
  name: string;
  artSrc: string;
  artWidth: number;
  artHeight: number;
  backButton: { top: number; left: number; width: number; height: number };
  nodes: PathNode[];
  briefingText: string[];
  warmup: WarmupConfig;
  /** Transfer-mission problem picker: given the problem that triggered a diagnosis, a different-family problem to transfer to. */
  nextProblem: Record<string, string>;
}
