import type { DsaTrial } from "../../lib/api";

interface ProblemSpecPanelProps {
  trial: DsaTrial;
}

export default function ProblemSpecPanel({ trial }: ProblemSpecPanelProps) {
  return (
    <aside className="flex flex-col h-full bg-void/80 border border-warp-cyan/20 rounded-xl p-4 overflow-y-auto select-text scrollbar-thin scrollbar-thumb-warp-cyan/30">
      {/* Title */}
      <h2 className="font-display text-xl text-white font-bold tracking-wide mb-3 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]">
        {trial.title}
      </h2>

      {/* Description Prompt */}
      <div className="text-sm font-ui text-white/90 leading-relaxed mb-6 space-y-2">
        <p>{trial.prompt}</p>
      </div>

      {/* Examples Section */}
      {trial.examples && trial.examples.length > 0 && (
        <div className="mb-6 space-y-3">
          <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-warp-cyan flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-warp-cyan" />
            Examples
          </h3>

          <div className="space-y-3">
            {trial.examples.map((ex, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-black/50 border border-white/10 text-xs font-mono space-y-1.5 shadow-inner"
              >
                <div className="text-white/60 font-semibold">Example {idx + 1}:</div>
                <div>
                  <span className="text-warp-cyan">Input: </span>
                  <span className="text-white">{ex.input}</span>
                </div>
                <div>
                  <span className="text-mint-success">Output: </span>
                  <span className="text-white font-bold">{ex.output}</span>
                </div>
                {ex.explanation && (
                  <div className="text-white/70 text-[11px] pt-1 border-t border-white/5">
                    <span className="text-white/40">Explanation: </span>
                    {ex.explanation}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Constraints Section */}
      {trial.constraints && trial.constraints.length > 0 && (
        <div className="space-y-2 mt-auto pt-4 border-t border-white/10">
          <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-white/60">
            Constraints
          </h3>
          <ul className="space-y-1 text-xs font-mono text-white/70">
            {trial.constraints.map((c, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-warp-cyan font-bold">•</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
}
