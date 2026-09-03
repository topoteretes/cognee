"use client";

import { usePrefersReducedMotion } from "@/modules/business/usePrefersReducedMotion";

// A tiny preview of the graph that's about to form, rather than a generic
// spinner — three nodes breathing in sequence with the edges between them
// already drawn, so the shape of "a connected model" is legible before any
// real data has arrived. Sits over BusinessCanvas's own ambient starfield
// (drawAmbientBackground), which keeps rendering underneath regardless of
// scene.isLoading, so this only needs to add the foreground signal.
const NODES = [
  { cx: 44, cy: 34, r: 5, delay: "delay-0" },
  { cx: 20, cy: 58, r: 4, delay: "delay-300" },
  { cx: 68, cy: 60, r: 4, delay: "delay-150" },
];

export default function BusinessLoading({ label }: { label: string }): React.JSX.Element {
  const reducedMotion = usePrefersReducedMotion();
  const pulseClass = reducedMotion ? "" : "animate-pulse";

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
      <svg width="88" height="88" viewBox="0 0 88 88" fill="none" aria-hidden="true">
        <line x1="44" y1="34" x2="20" y2="58" stroke="#43D9E8" strokeOpacity="0.25" strokeWidth="1.5" />
        <line x1="44" y1="34" x2="68" y2="60" stroke="#43D9E8" strokeOpacity="0.25" strokeWidth="1.5" />
        <line x1="20" y1="58" x2="68" y2="60" stroke="#43D9E8" strokeOpacity="0.15" strokeWidth="1.5" />
        {NODES.map((n) => (
          <circle
            key={`${n.cx}-${n.cy}`}
            cx={n.cx}
            cy={n.cy}
            r={n.r}
            fill="#43D9E8"
            className={`${pulseClass} ${n.delay}`}
          />
        ))}
      </svg>
      <span className="text-sm text-[#7E8CA6] animate-pulse">{label}</span>
    </div>
  );
}
