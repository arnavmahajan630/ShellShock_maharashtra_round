import { motion } from "framer-motion";

interface IterationDialProps {
  claimedCount?: number | null;
  actualCount: number;
  maxCount?: number;
  label?: string;
  terminates?: boolean;
}

export default function IterationDial({
  claimedCount,
  actualCount,
  maxCount = 6,
  label = "Loop Iteration Counter",
  terminates = true,
}: IterationDialProps) {
  const effectiveMax = Math.max(maxCount, actualCount + 2, (claimedCount ?? 0) + 2);
  const radius = 64;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;

  // Arc angles (270 degrees total arc)
  const arcLength = circumference * 0.75;
  const actualRatio = Math.min(1, actualCount / effectiveMax);
  const actualOffset = arcLength - actualRatio * arcLength;

  const hasClaim = claimedCount !== null && claimedCount !== undefined;
  const matches = hasClaim && claimedCount === actualCount;
  const isDrift = hasClaim && claimedCount !== actualCount;

  return (
    <div className="flex flex-col items-center rounded-lg border border-void-blue bg-void-blue/40 p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="h-2 w-2 rounded-full bg-starlight animate-pulse" />
        <h4 className="font-display text-xs text-starlight uppercase tracking-wider">
          {label}
        </h4>
      </div>

      <div className="relative flex items-center justify-center">
        <svg width="180" height="180" viewBox="0 0 180 180" className="transform -rotate-135">
          {/* Background Track */}
          <circle
            cx="90"
            cy="90"
            r={radius}
            fill="transparent"
            stroke="#153268"
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
          />

          {/* Actual Iterations Arc */}
          <motion.circle
            cx="90"
            cy="90"
            r={radius}
            fill="transparent"
            stroke={terminates ? "#2de7fc" : "#fa5155"}
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={arcLength}
            animate={{ strokeDashoffset: actualOffset }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            strokeLinecap="round"
          />

          {/* Claimed marker if different */}
          {isDrift && (
            <circle
              cx="90"
              cy="90"
              r={radius}
              fill="transparent"
              stroke="#fcb143"
              strokeWidth={strokeWidth}
              strokeDasharray={`4 ${circumference}`}
              strokeDashoffset={arcLength - Math.min(1, (claimedCount || 0) / effectiveMax) * arcLength}
              strokeLinecap="round"
            />
          )}
        </svg>

        {/* Center Readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="font-ui text-5xl font-bold text-starlight">
            {actualCount}
          </span>
          <span className="font-display text-[9px] uppercase tracking-wider text-slate mt-1">
            {terminates ? "Actual Passes" : "Loop Never Stops"}
          </span>
        </div>
      </div>

      {/* Comparison badge */}
      <div className="mt-2 text-center font-ui text-xl">
        {hasClaim ? (
          matches ? (
            <span className="text-mint-success font-bold">
              ✓ Prediction matched: claimed {claimedCount} pass{claimedCount === 1 ? "" : "es"}
            </span>
          ) : (
            <div className="flex flex-col items-center gap-1">
              <span className="text-starlight font-bold">
                ⚠ Boundary Drift: claimed {claimedCount}, actual was {actualCount}
              </span>
              <span className="font-ui text-lg text-slate">
                Notice: condition `&lt;= n` includes the boundary step (0, 1, ..., n)
              </span>
            </div>
          )
        ) : (
          <span className="text-slate">
            Verified execution: {actualCount} iteration{actualCount === 1 ? "" : "s"} evaluated
          </span>
        )}
      </div>
    </div>
  );
}
