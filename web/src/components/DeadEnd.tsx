import { useNavigate } from "react-router-dom";

interface DeadEndProps {
  message: string;
  /** Where "Back" goes. Falls back to the galaxy map when omitted (e.g. planet slug unknown). */
  to?: string;
}

/**
 * Shown when a screen is reached without the session state it needs (direct navigation,
 * a refresh, or the back button) — always offers a way out instead of a dead stop.
 */
export default function DeadEnd({ message, to }: DeadEndProps) {
  const navigate = useNavigate();
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-deep-space">
      <p className="font-ui text-2xl text-slate">{message}</p>
      <button
        type="button"
        onClick={() => navigate(to ?? "/map")}
        className="font-ui text-xl px-6 py-2 rounded bg-warp-cyan text-deep-space hover:brightness-110 active:scale-95"
      >
        {to ? "Back to planet path" : "Back to Galaxy"}
      </button>
    </div>
  );
}
