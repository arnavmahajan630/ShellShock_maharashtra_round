import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { api, type ChartStar, type ChartStarState, type ChartView } from "../lib/api";
import { useMissionStore } from "../store/missionStore";

type Mode = "mine" | "full";

const STATE_WORDS: Record<ChartStarState, string> = {
  uncharted: "Uncharted — no levels exist for this part of C yet.",
  unexplored: "Unexplored — no level cleared for this skill yet.",
  attempted: "Attempted — tried, not cleared yet.",
  known: "Known — cleared, and no linked mistake is active.",
  shaky: "Shaky — cleared, but a linked mistake is still active.",
};

const percent = (value: number) => `${Math.round(value * 100)}%`;

type Anchor = "start" | "middle" | "end";

/** Where a star's name goes: pushed away from the middle of its constellation, so names fan out
 *  around the ring instead of piling up underneath it. Uncharted stars keep the name below, turned
 *  inward near the left and right edges of the canvas. */
function labelPlace(star: ChartStar, center: [number, number] | null, width: number): { x: number; y: number; anchor: Anchor } {
  const [x, y] = star.pos;
  if (!center) {
    if (x > width - 10) return { x: x + 0.9, y: y + 1.9, anchor: "end" };
    if (x < 10) return { x: x - 0.9, y: y + 1.9, anchor: "start" };
    return { x, y: y + 1.9, anchor: "middle" };
  }
  const dx = x - center[0];
  const dy = y - center[1];
  if (Math.abs(dx) < 1) return { x, y: dy < 0 ? y - 2.4 : y + 3.2, anchor: "middle" };
  return { x: x + (dx > 0 ? 2.1 : -2.1), y: y + (dy < -1 ? -0.3 : dy > 1 ? 1.1 : 0.4), anchor: dx > 0 ? "start" : "end" };
}

interface StarMarkProps {
  star: ChartStar;
  center: [number, number] | null;
  width: number;
  mode: Mode;
  selected: boolean;
  still: boolean;
  onSelect: () => void;
}

function StarMark({ star, center, width, mode, selected, still, onSelect }: StarMarkProps) {
  const [x, y] = star.pos;
  const label = labelPlace(star, center, width);
  const uncharted = star.kind === "uncharted";
  const brightness = star.brightness ?? 0;
  const lit = star.state === "known" || star.state === "shaky";
  const labelDim = mode === "mine" && !lit && !star.at_risk && !star.suggested && !selected;

  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${star.label}: ${star.state}`}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      className="cursor-pointer outline-none"
    >
      <title>{star.label}</title>
      <circle cx={x} cy={y} r={2} fill="transparent" />

      {mode === "full" ? (
        <circle
          cx={x}
          cy={y}
          r={uncharted ? 0.6 : 0.8}
          strokeWidth={0.18}
          className={(uncharted ? "stroke-slate" : "stroke-light") + " " + (star.cleared ? "fill-warp-cyan" : "fill-deep-space")}
        />
      ) : (
        <>
          {star.suggested && (
            <motion.circle
              cx={x}
              cy={y}
              r={2}
              fill="none"
              strokeWidth={0.2}
              className="stroke-pulsar-magenta"
              animate={still ? undefined : { r: [1.6, 2.6], opacity: [0.95, 0.15] }}
              transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
            />
          )}
          {star.at_risk && (
            <circle cx={x} cy={y} r={1.55} fill="none" strokeWidth={0.16} strokeDasharray="0.5 0.35" className="stroke-alert-red" />
          )}

          {star.state === "uncharted" && <circle cx={x} cy={y} r={0.45} className="fill-void-blue" />}
          {star.state === "unexplored" && <circle cx={x} cy={y} r={0.65} opacity={0.45} className="fill-slate" />}
          {star.state === "attempted" && (
            <circle cx={x} cy={y} r={0.7} opacity={0.75} strokeWidth={0.14} className="fill-slate stroke-starlight" />
          )}
          {star.state === "known" && (
            <>
              <circle
                cx={x}
                cy={y}
                r={1.2 + 1.3 * brightness}
                opacity={0.15 + 0.3 * brightness}
                filter="url(#chart-glow)"
                className="fill-warp-cyan"
              />
              <motion.circle
                cx={x}
                cy={y}
                r={0.6 + 0.35 * brightness}
                className="fill-warp-cyan"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.45 + 0.55 * brightness }}
                transition={{ duration: 0.6 }}
              />
            </>
          )}
          {star.state === "shaky" && (
            <>
              <circle cx={x} cy={y} r={1.9} opacity={0.3} filter="url(#chart-glow)" className="fill-starlight" />
              <motion.circle
                cx={x}
                cy={y}
                r={0.9}
                className="fill-starlight"
                animate={still ? undefined : { opacity: [1, 0.35, 0.9, 0.5, 1] }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
              />
              <path
                d={`M ${x - 0.5} ${y - 0.65} L ${x + 0.12} ${y - 0.1} L ${x - 0.2} ${y + 0.15} L ${x + 0.45} ${y + 0.7}`}
                fill="none"
                strokeWidth={0.2}
                className="stroke-deep-space"
              />
            </>
          )}
        </>
      )}

      {selected && <circle cx={x} cy={y} r={2.1} fill="none" strokeWidth={0.12} className="stroke-light" />}

      <text
        x={label.x}
        y={label.y}
        textAnchor={label.anchor}
        fontSize={uncharted ? 1.25 : 1.45}
        opacity={mode === "full" ? 1 : labelDim ? 0.55 : 1}
        className={"font-ui select-none " + (uncharted || labelDim ? "fill-slate" : "fill-light")}
      >
        {star.label}
      </text>
    </g>
  );
}

function LegendItem({ swatch, children }: { swatch: string; children: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={"inline-block h-3 w-3 shrink-0 rounded-full " + swatch} />
      <span>{children}</span>
    </li>
  );
}

export default function StarChartScreen() {
  const navigate = useNavigate();
  const learnerId = useMissionStore((s) => s.learnerId);
  const still = useReducedMotion() ?? false;

  const [chart, setChart] = useState<ChartView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("mine");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        let view = await api.getChart(learnerId);
        if (view === null) {
          // The server has no record of this pilot yet: create it, then ask again.
          await api.ensureLearner(learnerId);
          view = await api.getChart(learnerId);
        }
        if (!mounted) return;
        if (view === null) {
          setError(
            "The server answered \"not found\" for the chart. It was probably started before the star chart was added: stop it and start it again.",
          );
        }
        else setChart(view);
      } catch (err) {
        if (!mounted) return;
        const detail = err instanceof Error ? err.message : "unknown error";
        setError(`Couldn't load the chart from the server (${detail}). Is it running on port 8000?`);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [learnerId, attempt]);

  const positions = useMemo(() => {
    const map = new Map<string, [number, number]>();
    for (const topic of chart?.topics ?? []) map.set(topic.id, topic.pos);
    for (const star of chart?.stars ?? []) map.set(star.id, star.pos);
    return map;
  }, [chart]);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space px-6 text-light">
        <div className="max-w-lg w-full rounded-lg border border-alert-red p-8 text-center">
          <h1 className="font-display text-sm text-alert-red mb-3">CHART OFFLINE</h1>
          <p className="font-ui text-xl mb-6">{error}</p>
          <div className="flex justify-center gap-3">
            <button
              type="button"
              onClick={() => {
                setError(null);
                setAttempt((n) => n + 1);
              }}
              className="font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={() => navigate("/map")}
              className="font-ui text-xl px-6 py-2 rounded border border-slate text-light hover:border-warp-cyan active:scale-95"
            >
              Back to Galaxy
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!chart) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-deep-space text-light">
        <p className="font-ui text-2xl text-slate">Charting the stars…</p>
      </div>
    );
  }

  const { summary, suggested_next: suggested } = chart;
  const selected = chart.stars.find((star) => star.id === selectedId) ?? null;
  const topicLabel = (id: string | null) => chart.topics.find((topic) => topic.id === id)?.label ?? null;

  return (
    <div className="flex h-screen flex-col bg-deep-space text-light">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-void-blue px-6 py-3 pr-20">
        <button
          type="button"
          onClick={() => navigate("/map")}
          className="font-ui text-xl px-4 py-1 rounded border border-slate text-light hover:border-warp-cyan active:scale-95"
        >
          ← Galaxy
        </button>
        <h1 className="font-display text-sm text-warp-cyan">STAR CHART</h1>
        <p className="font-ui text-xl">
          Explored <span className="text-warp-cyan">{summary.cleared}</span> of {summary.charted} charted skills
          {summary.shaky > 0 && <span className="text-starlight"> · {summary.shaky} shaky</span>}
          {summary.at_risk > 0 && <span className="text-alert-red"> · {summary.at_risk} at risk</span>}
        </p>
        <p className="font-ui text-xl text-slate">{summary.uncharted} regions of C uncharted</p>
        <div className="ml-auto flex overflow-hidden rounded border border-slate font-ui text-lg">
          {(["mine", "full"] as const).map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={mode === option}
              onClick={() => setMode(option)}
              className={"px-4 py-1 " + (mode === option ? "bg-warp-cyan text-deep-space" : "text-light hover:text-warp-cyan")}
            >
              {option === "mine" ? "My chart" : "Full C chart"}
            </button>
          ))}
        </div>
      </header>

      {suggested && (
        <div className="flex flex-wrap items-center gap-3 border-b border-void-blue px-6 py-2 font-ui text-lg">
          <span className="text-pulsar-magenta">Most informative next step:</span>
          <button type="button" onClick={() => setSelectedId(suggested.star_id)} className="text-light underline hover:text-warp-cyan">
            {suggested.label}
          </button>
          <span className="text-slate">{suggested.reason}</span>
          <button
            type="button"
            onClick={() => navigate(suggested.route)}
            className="rounded bg-pulsar-magenta px-4 py-0.5 text-deep-space hover:brightness-110 active:scale-95"
          >
            Go →
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <main className="min-w-0 flex-1 p-3">
          <svg
            viewBox={`-6 -1 ${chart.canvas.width + 12} ${chart.canvas.height + 2}`}
            className="h-full w-full"
            role="img"
            aria-label="Star chart of C skills"
          >
            <defs>
              <filter id="chart-glow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="0.7" />
              </filter>
            </defs>

            {chart.topics.map((topic) => (
              <g key={topic.id}>
                <circle
                  cx={topic.pos[0]}
                  cy={topic.pos[1]}
                  r={topic.radius}
                  strokeWidth={0.08}
                  strokeDasharray="0.3 0.5"
                  opacity={0.5}
                  className="fill-void-blue/20 stroke-slate"
                />
                <text
                  x={topic.pos[0]}
                  y={topic.pos[1] - 0.1}
                  textAnchor="middle"
                  fontSize={1.5}
                  className="font-ui fill-warp-cyan select-none uppercase"
                >
                  {topic.label}
                </text>
                <text
                  x={topic.pos[0]}
                  y={topic.pos[1] + 1.6}
                  textAnchor="middle"
                  fontSize={1.5}
                  className="font-ui fill-slate select-none"
                >
                  {topic.skills_cleared}/{topic.skills_total}
                </text>
              </g>
            ))}

            {chart.edges.map((edge) => {
              const from = positions.get(edge.from);
              const to = positions.get(edge.to);
              if (!from || !to) return null;
              const prereq = edge.kind === "prereq";
              return (
                <line
                  key={`${edge.kind}:${edge.from}:${edge.to}`}
                  x1={from[0]}
                  y1={from[1]}
                  x2={to[0]}
                  y2={to[1]}
                  strokeWidth={prereq ? 0.1 : 0.12}
                  strokeDasharray={prereq ? "0.6 0.6" : undefined}
                  opacity={prereq ? 0.3 : 0.55}
                  className={prereq ? "stroke-slate" : "stroke-warp-cyan"}
                />
              );
            })}

            {chart.stars.map((star) => (
              <StarMark
                key={star.id}
                star={star}
                center={star.topic ? positions.get(star.topic) ?? null : null}
                width={chart.canvas.width}
                mode={mode}
                still={still}
                selected={star.id === selectedId}
                onSelect={() => setSelectedId(star.id)}
              />
            ))}
          </svg>
        </main>

        <aside className="w-80 shrink-0 overflow-y-auto border-l border-void-blue p-5 font-ui text-lg">
          {selected ? (
            <>
              <h2 className="font-display text-xs leading-relaxed text-warp-cyan mb-1">{selected.label}</h2>
              <p className="text-slate mb-3">{topicLabel(selected.topic) ?? "Beyond the charted topics"}</p>
              <p className="mb-3">{STATE_WORDS[selected.state]}</p>

              {selected.brightness !== null && (
                <p className="mb-3">
                  Solid: <span className="text-warp-cyan">{percent(selected.brightness)}</span>{" "}
                  <span className="text-slate">(the models' estimate)</span>
                </p>
              )}
              {selected.risk !== null && (
                <p className="mb-3">
                  Chance a known mistake trips you here:{" "}
                  <span className={selected.at_risk ? "text-alert-red" : "text-light"}>{percent(selected.risk)}</span>
                </p>
              )}
              {selected.risk_from.length > 0 && (
                <p className="mb-3 text-alert-red">At risk because of {selected.risk_from.map((hazard) => hazard.name).join(", ")}.</p>
              )}

              {selected.problems.length > 0 && (
                <>
                  <h3 className="text-slate mt-4">Levels</h3>
                  <ul className="mb-3">
                    {selected.problems.map((problem) => (
                      <li key={problem.problem_id}>
                        {problem.name} <span className="text-slate">({problem.problem_id})</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {selected.hazards.length > 0 && (
                <>
                  <h3 className="text-slate mt-4">Linked mistakes</h3>
                  <ul className="mb-3 space-y-2">
                    {selected.hazards.map((hazard) => (
                      <li key={hazard.id}>
                        <span className={hazard.p_active >= 0.5 ? "text-alert-red" : "text-light"}>{hazard.name}</span>{" "}
                        <span className="text-slate">
                          {percent(hazard.p_active)} · {hazard.state.toLowerCase()}
                        </span>
                        {hazard.subtitle && <span className="block text-base text-slate">{hazard.subtitle}</span>}
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {selected.playable && (
                <button
                  type="button"
                  onClick={() => navigate(selected.playable!)}
                  className="mt-2 font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
                >
                  Go to mission →
                </button>
              )}
            </>
          ) : (
            <>
              <p className="mb-4 text-slate">Click a star to see what it is and what stands in its way.</p>
              <h3 className="mb-2 text-slate">Legend</h3>
              <ul className="space-y-1">
                <LegendItem swatch="bg-warp-cyan">Known — brighter means more solid</LegendItem>
                <LegendItem swatch="bg-starlight">Shaky — cleared, a mistake is still active</LegendItem>
                <LegendItem swatch="border border-starlight bg-slate">Attempted, not cleared</LegendItem>
                <LegendItem swatch="bg-slate/50">Unexplored</LegendItem>
                <LegendItem swatch="border border-dashed border-alert-red">At risk from an active mistake</LegendItem>
                <LegendItem swatch="border-2 border-pulsar-magenta">Most informative next step</LegendItem>
                <LegendItem swatch="bg-void-blue">Uncharted — C we have no levels for</LegendItem>
              </ul>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
