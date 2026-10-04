import { motion } from "framer-motion";

interface SignalGateWorldProps {
  label: string;
  status: "idle" | "running" | "pass" | "fail";
  passed: number;
  total: number;
}

export default function SignalGateWorld({ label, status, passed, total }: SignalGateWorldProps) {
  const open = status === "pass";
  const sparking = status === "fail";

  return (
    <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-void-blue bg-deep-space/60 py-10">
      <div className="relative h-32 w-32">
        <motion.div
          className="absolute inset-0 rounded-full border-4"
          style={{ borderColor: open ? "var(--color-mint-success)" : sparking ? "var(--color-alert-red)" : "var(--color-slate)" }}
          animate={{
            boxShadow: open
              ? "0 0 36px 10px var(--color-mint-success)"
              : sparking
                ? ["0 0 24px 6px var(--color-alert-red)", "0 0 4px 1px var(--color-alert-red)", "0 0 24px 6px var(--color-alert-red)"]
                : "0 0 0px 0px transparent",
          }}
          transition={{ duration: 0.6, repeat: sparking ? Infinity : 0 }}
        />
        <motion.div
          className="absolute left-1/2 top-1/2 h-20 w-3 -translate-x-1/2 -translate-y-1/2 rounded bg-warp-cyan"
          style={{ transformOrigin: "center" }}
          animate={{ rotate: open ? 90 : 0, backgroundColor: sparking ? "#FA5155" : "#2DE7FC" }}
          transition={{ duration: 0.4 }}
        />
      </div>
      <p className="font-display text-xs text-light">{label}</p>
      <p className="font-ui text-xl text-slate">
        {status === "idle" ? "Awaiting signal…" : status === "running" ? "Scanning…" : `${passed} / ${total} gates held`}
      </p>
    </div>
  );
}
