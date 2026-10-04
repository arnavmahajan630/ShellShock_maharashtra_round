import { motion, AnimatePresence } from "framer-motion";

interface SkipModalProps {
  isOpen: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

interface LeaveModalProps {
  isOpen: boolean;
  currentIndex: number;
  totalTrials: number;
  timeSpentSeconds: number;
  onCancel: () => void;
  onConfirm: () => void;
}

export function SkipTrialModal({ isOpen, onCancel, onConfirm }: SkipModalProps) {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md select-none font-ui">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full max-w-md p-6 rounded-2xl bg-deep-space border border-rose-500/50 shadow-[0_0_40px_rgba(244,63,94,0.3)] text-white space-y-5"
        >
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-rose-500/20 border border-rose-500 flex items-center justify-center text-rose-400 text-lg">
              ⚠️
            </div>
            <div>
              <h3 className="font-display text-lg font-bold text-white">Skip this trial?</h3>
              <p className="text-xs text-white/60">
                This will be marked as not passed and may affect your results.
              </p>
            </div>
          </div>

          {/* Explanation Checklist */}
          <div className="p-4 rounded-xl bg-black/40 border border-white/10 space-y-2 text-xs font-mono">
            <span className="text-rose-400 font-bold block mb-1">What happens?</span>
            <ul className="space-y-1.5 text-white/80">
              <li className="flex items-center gap-2">
                <span className="text-rose-400">•</span>
                <span>Current trial will be marked as skipped</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-rose-400">•</span>
                <span>You cannot attempt this trial again in this session</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-rose-400">•</span>
                <span>Your progress will be saved</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-rose-400">•</span>
                <span>You can continue with the next trial</span>
              </li>
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-5 py-2 rounded-lg border border-white/20 hover:bg-white/10 text-xs font-bold text-white transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={onConfirm}
              className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-[0_0_20px_rgba(244,63,94,0.5)] transition-all cursor-pointer"
            >
              Skip Trial
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

export function LeaveTrialsModal({
  isOpen,
  currentIndex,
  totalTrials,
  timeSpentSeconds,
  onCancel,
  onConfirm,
}: LeaveModalProps) {
  if (!isOpen) return null;

  const minutes = Math.floor(timeSpentSeconds / 60);
  const seconds = timeSpentSeconds % 60;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md select-none font-ui">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full max-w-md p-6 rounded-2xl bg-deep-space border border-amber-500/50 shadow-[0_0_40px_rgba(245,158,11,0.3)] text-white space-y-5"
        >
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-amber-500/20 border border-amber-500 flex items-center justify-center text-amber-400 text-lg">
              ⚠️
            </div>
            <div>
              <h3 className="font-display text-lg font-bold text-white">Leave Deep Space Trials?</h3>
              <p className="text-xs text-white/60">
                Your current progress will be saved. You can return later.
              </p>
            </div>
          </div>

          {/* Progress Summary Card */}
          <div className="p-4 rounded-xl bg-black/40 border border-white/10 space-y-2 text-xs font-mono">
            <span className="text-amber-400 font-bold block mb-1">Your progress:</span>
            <ul className="space-y-1.5 text-white/80">
              <li className="flex items-center gap-2">
                <span className="text-warp-cyan">✦</span>
                <span>{currentIndex} / {totalTrials} trials completed</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-amber-400">⏱</span>
                <span>{minutes}m {seconds}s elapsed</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-mint-success">✓</span>
                <span>Results saved automatically</span>
              </li>
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-5 py-2 rounded-lg border border-white/20 hover:bg-white/10 text-xs font-bold text-white transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={onConfirm}
              className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-[0_0_20px_rgba(244,63,94,0.5)] transition-all cursor-pointer"
            >
              Leave
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
