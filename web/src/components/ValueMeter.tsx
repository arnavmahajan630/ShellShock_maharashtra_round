interface ValueMeterProps {
  exact: unknown;
  shown: unknown;
}

export default function ValueMeter({ exact, shown }: ValueMeterProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="rounded border border-mint-success/50 bg-deep-space/50 p-4 text-center">
        <p className="font-ui text-lg text-slate">Exact (properly initialized)</p>
        <p className="font-mono text-2xl text-mint-success mt-1">{String(exact)}</p>
      </div>
      <div className="rounded border border-alert-red/50 bg-deep-space/50 p-4 text-center">
        <p className="font-ui text-lg text-slate">Shown (your code)</p>
        <p className="font-mono text-2xl text-alert-red mt-1">{String(shown)}</p>
      </div>
    </div>
  );
}
