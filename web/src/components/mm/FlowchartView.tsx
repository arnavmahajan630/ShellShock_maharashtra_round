export interface FlowchartNode {
  id: string;
  kind: "start" | "decision" | "process" | "end";
  label: string;
}

export interface FlowchartEdge {
  from: string;
  to: string;
  label: string | null;
}

interface FlowchartViewProps {
  nodes: FlowchartNode[];
  edges: FlowchartEdge[];
  selectedPath: string[];
  onNodeClick: (nodeId: string) => void;
  onUndo: () => void;
  onReset: () => void;
}

// Preset visual layout coordinates for 4-node canonical diagram
const NODE_COORDINATES: Record<string, { x: number; y: number }> = {
  n1: { x: 300, y: 70 },
  n2: { x: 300, y: 190 },
  n3: { x: 170, y: 310 },
  n4: { x: 300, y: 440 },
};

export default function FlowchartView({
  nodes,
  edges,
  selectedPath,
  onNodeClick,
  onUndo,
  onReset,
}: FlowchartViewProps) {
  return (
    <div className="relative rounded-lg border border-void-blue bg-void-blue/40 p-5">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-void-blue pb-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-starlight animate-pulse" />
          <span className="font-display text-xs text-starlight uppercase tracking-wider">
            Control Flow Graph · Tap Nodes in Order
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onUndo}
            disabled={selectedPath.length === 0}
            className="px-4 py-1 rounded font-ui text-xl bg-deep-space border border-void-blue text-slate hover:border-starlight hover:text-starlight disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
          >
            Undo
          </button>
          <button
            type="button"
            onClick={onReset}
            disabled={selectedPath.length === 0}
            className="px-4 py-1 rounded font-ui text-xl bg-alert-red/10 border border-alert-red/30 text-alert-red hover:bg-alert-red/20 disabled:opacity-30 disabled:pointer-events-none transition-colors cursor-pointer"
          >
            Reset
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative flex justify-center overflow-x-auto select-none py-2">
        <svg width="600" height="520" viewBox="0 0 600 520" className="overflow-visible">
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#2de7fc" />
            </marker>
            <marker
              id="arrow-active"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#fcb143" />
            </marker>
          </defs>

          {/* Render Connecting Edges */}
          {edges.map((edge, idx) => {
            const fromCoord = NODE_COORDINATES[edge.from] || { x: 300, y: 70 };
            const toCoord = NODE_COORDINATES[edge.to] || { x: 300, y: 440 };

            // Determine if this transition is part of current tapped path
            const fromStep = selectedPath.indexOf(edge.from);
            const toStep = selectedPath.indexOf(edge.to);
            const isActiveTransition =
              fromStep !== -1 && toStep === fromStep + 1;

            // Generate path string
            let d = "";
            let labelX = (fromCoord.x + toCoord.x) / 2;
            let labelY = (fromCoord.y + toCoord.y) / 2;

            if (fromCoord.x === toCoord.x) {
              // Straight vertical line (or bypass around n3)
              if (edge.from === "n2" && edge.to === "n4") {
                // Curved bypass line to the right
                d = `M ${fromCoord.x + 80} ${fromCoord.y} C ${fromCoord.x + 160} ${fromCoord.y + 70}, ${toCoord.x + 160} ${toCoord.y - 70}, ${toCoord.x + 75} ${toCoord.y}`;
                labelX = fromCoord.x + 140;
                labelY = (fromCoord.y + toCoord.y) / 2;
              } else {
                d = `M ${fromCoord.x} ${fromCoord.y + 35} L ${toCoord.x} ${toCoord.y - 35}`;
              }
            } else {
              // Diagonal / routed line
              d = `M ${fromCoord.x} ${fromCoord.y + 35} Q ${fromCoord.x} ${toCoord.y - 35}, ${toCoord.x} ${toCoord.y - 35}`;
              labelX = (fromCoord.x + toCoord.x) / 2 - 20;
              labelY = (fromCoord.y + toCoord.y) / 2 - 10;
            }

            return (
              <g key={idx}>
                <path
                  d={d}
                  fill="none"
                  stroke={isActiveTransition ? "#fcb143" : "#153268"}
                  strokeWidth={isActiveTransition ? "3" : "2"}
                  strokeDasharray={isActiveTransition ? "none" : "4 3"}
                  markerEnd={isActiveTransition ? "url(#arrow-active)" : "url(#arrow)"}
                  className="transition-all duration-300"
                />
                {edge.label && (
                  <text
                    x={labelX}
                    y={labelY}
                    fill={isActiveTransition ? "#fcb143" : "#7a8fb5"}
                    textAnchor="middle"
                    style={{ fontFamily: "'VT323', monospace", fontSize: "16px" }}
                  >
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}

          {/* Render Nodes */}
          {nodes.map((node) => {
            const coord = NODE_COORDINATES[node.id] || { x: 300, y: 70 };
            const stepIndex = selectedPath.indexOf(node.id);
            const isSelected = stepIndex !== -1;
            const isDecision = node.kind === "decision";

            return (
              <g
                key={node.id}
                onClick={() => onNodeClick(node.id)}
                className="cursor-pointer group"
                transform={`translate(${coord.x}, ${coord.y})`}
              >
                {/* Transparent Click Hitbox */}
                <rect
                  x="-95"
                  y="-35"
                  width="190"
                  height="70"
                  fill="transparent"
                  style={{ pointerEvents: "all" }}
                />

                {/* Node Box Shape */}
                {isDecision ? (
                  // Decision Rhombus / Diamond
                  <polygon
                    points="-85,0 0,-34 85,0 0,34"
                    fill={isSelected ? "#153268" : "#0c1937"}
                    stroke={isSelected ? "#fcb143" : "#2de7fc"}
                    strokeWidth={isSelected ? "3" : "1.5"}
                    className="transition-all duration-200 group-hover:brightness-125"
                  />
                ) : (
                  // Process / Start / End Box
                  <rect
                    x="-90"
                    y="-30"
                    width="180"
                    height="60"
                    rx="6"
                    fill={isSelected ? "#153268" : "#0c1937"}
                    stroke={
                      isSelected
                        ? "#fcb143"
                        : node.kind === "start"
                        ? "#4be975"
                        : node.kind === "end"
                        ? "#fb4afd"
                        : "#2de7fc"
                    }
                    strokeWidth={isSelected ? "3" : "1.5"}
                    className="transition-all duration-200 group-hover:brightness-125 shadow-lg"
                  />
                )}

                {/* Node Label Lines */}
                <text
                  textAnchor="middle"
                  y={node.label.includes("\n") ? -6 : 6}
                  fill={isSelected ? "#fcb143" : "#edf2fd"}
                  style={{ fontFamily: "'VT323', monospace", fontSize: "18px", fontWeight: "bold" }}
                  className="pointer-events-none select-none"
                >
                  {node.label.split("\n")[0]}
                </text>
                {node.label.includes("\n") && (
                  <text
                    textAnchor="middle"
                    y={16}
                    fill={isSelected ? "#fcb143" : "#7a8fb5"}
                    style={{ fontFamily: "'VT323', monospace", fontSize: "16px" }}
                    className="pointer-events-none select-none"
                  >
                    {node.label.split("\n")[1]}
                  </text>
                )}

                {/* Tapped Sequence Badge */}
                {isSelected && (
                  <circle
                    cx="80"
                    cy="-22"
                    r="12"
                    fill="#fcb143"
                    stroke="#0c1937"
                    strokeWidth="2"
                  />
                )}
                {isSelected && (
                  <text
                    x="80"
                    y="-17"
                    fill="#0c1937"
                    style={{ fontFamily: "'VT323', monospace", fontSize: "15px", fontWeight: "bold" }}
                    textAnchor="middle"
                  >
                    {stepIndex + 1}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Node Click Buttons & Path Breadcrumbs */}
      <div className="border-t border-void-blue pt-4 space-y-3">
        <div className="flex items-center justify-between gap-2 flex-wrap font-ui text-xl">
          <div className="flex items-center gap-2">
            <span className="text-slate">Trace Steps:</span>
            {selectedPath.length > 0 ? (
              <div className="flex items-center gap-2 flex-wrap">
                {selectedPath.map((id, idx) => (
                  <span key={idx} className="flex items-center gap-1.5">
                    <span className="px-3 py-0.5 rounded bg-starlight/20 text-starlight font-bold border border-starlight/50 font-ui text-xl">
                      {id}
                    </span>
                    {idx < selectedPath.length - 1 && <span className="text-slate">→</span>}
                  </span>
                ))}
              </div>
            ) : (
              <span className="text-starlight animate-pulse font-semibold">
                Tap nodes on diagram or buttons below to trace execution:
              </span>
            )}
          </div>
        </div>

        {/* Quick Click Badges */}
        <div className="flex items-center gap-2 flex-wrap pt-1 font-ui text-xl">
          <span className="text-slate">Tap node:</span>
          {nodes.map((n) => {
            const isTapped = selectedPath.includes(n.id);
            return (
              <button
                key={n.id}
                type="button"
                onClick={() => onNodeClick(n.id)}
                className={`px-4 py-1 rounded font-ui text-xl transition-all cursor-pointer ${
                  isTapped
                    ? "bg-starlight/20 border border-starlight text-starlight hover:bg-starlight/30 shadow-[0_0_10px_rgba(252,177,67,0.3)]"
                    : "bg-deep-space border border-void-blue text-slate hover:border-warp-cyan hover:text-light"
                }`}
              >
                + {n.id} ({n.kind})
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
