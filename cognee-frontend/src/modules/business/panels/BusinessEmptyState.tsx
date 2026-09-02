"use client";

import { usePrefersReducedMotion } from "@/modules/business/usePrefersReducedMotion";

// The sad counterpart to BusinessLoading's breathing, converging graph: two
// nodes reaching for each other over a dashed line that never resolves into
// a real edge, with a third drooping below as if it gave up. Sits over
// BusinessCanvas's ambient starfield, which keeps rendering underneath
// regardless of this state.
export default function BusinessEmptyState({ label }: { label: string }): React.JSX.Element {
  const reducedMotion = usePrefersReducedMotion();
  const driftClass = reducedMotion ? "" : "animate-pulse";

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center">
      <svg width="88" height="88" viewBox="0 0 88 88" fill="none" aria-hidden="true">
        <line x1="26" y1="28" x2="60" y2="32" stroke="#7E8CA6" strokeOpacity="0.3" strokeWidth="1.5" strokeDasharray="3 5" />
        <line x1="26" y1="28" x2="34" y2="62" stroke="#7E8CA6" strokeOpacity="0.18" strokeWidth="1.5" strokeDasharray="2 6" />
        <circle cx="26" cy="28" r="5" fill="#7E8CA6" fillOpacity="0.55" className={driftClass} />
        <circle cx="60" cy="32" r="4" fill="#7E8CA6" fillOpacity="0.35" className={`${driftClass} delay-300`} />
        <circle cx="34" cy="62" r="3.5" fill="#7E8CA6" fillOpacity="0.25" className={`${driftClass} delay-150`} />
      </svg>
      <span className="max-w-[280px] text-sm text-[#7E8CA6]">{label}</span>
    </div>
  );
}
