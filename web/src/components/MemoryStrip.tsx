interface MemoryStripProps {
  array: string;
  values: unknown[];
  reads: number[];
}

export default function MemoryStrip({ array, values, reads }: MemoryStripProps) {
  const voidIndex = values.length;
  const readsVoid = reads.includes(voidIndex);
  const cells = readsVoid ? [...values, "void"] : values;

  return (
    <div>
      <p className="font-ui text-lg text-slate mb-2">
        <code>{array}</code>
      </p>
      <div className="flex gap-1">
        {cells.map((value, i) => {
          const isVoid = i === voidIndex && readsVoid;
          const read = reads.includes(i);
          return (
            <div
              key={i}
              className={
                "flex h-14 w-14 flex-col items-center justify-center rounded border font-mono text-sm " +
                (isVoid
                  ? "border-alert-red bg-alert-red/20 text-alert-red animate-pulse"
                  : read
                    ? "border-warp-cyan bg-warp-cyan/10 text-light"
                    : "border-void-blue text-slate")
              }
            >
              <span>{String(value)}</span>
              <span className="text-xs text-slate">{i}</span>
            </div>
          );
        })}
      </div>
      {readsVoid && <p className="font-ui text-base text-alert-red mt-2">Index {voidIndex} is one past the end — reading it is undefined.</p>}
    </div>
  );
}
