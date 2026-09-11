"use client";

import type { SourceDetail } from "../computeSourceDetail";

interface SourceDetailCardProps {
  detail: SourceDetail | null;
  onClose: () => void;
}

// Answers "what's actually in this source" beyond what the Sources rail
// tile already shows (entity/doc counts): a type breakdown, a document/
// record name sample, and how many other sources it bridges into. Opens in
// the same right-side slot NodePanel uses — BusinessView renders at most
// one of the two, entity selection taking priority.
export default function SourceDetailCard({ detail, onClose }: SourceDetailCardProps) {
  if (!detail) return null;

  return (
    <div className="absolute right-4 top-16 z-10 w-72 rounded-xl border border-[#2A3652] bg-[#1A2438] p-4 text-sm text-[#E9EEF6]">
      <div className="flex items-start justify-between">
        <div className="flex min-w-0 items-center gap-2 font-semibold">
          <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: detail.color }} />
          <span className="truncate" title={detail.displayName}>{detail.displayName}</span>
        </div>
        <button type="button" onClick={onClose} className="shrink-0 text-[#7E8CA6] hover:text-[#E9EEF6]">
          ✕
        </button>
      </div>
      <div className="mt-1 text-xs text-[#7E8CA6]">
        {detail.entityCount} entities · {detail.docCount} document{detail.docCount === 1 ? "" : "s"}
        {detail.bridgeCount
          ? ` · bridges to ${detail.bridgeCount} other source${detail.bridgeCount === 1 ? "" : "s"}`
          : ""}
      </div>

      {detail.typeBreakdown.length > 0 && (
        <>
          <div className="mt-3 text-xs uppercase tracking-wide text-[#7E8CA6]">kinds of things</div>
          <ul className="mt-1 flex flex-col gap-1">
            {detail.typeBreakdown.map((row) => (
              <li key={row.type} className="flex items-center justify-between gap-2 text-xs">
                <span className="truncate">{row.type}</span>
                <span className="shrink-0 text-[#7E8CA6]">{row.count}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {detail.documentNames.length > 0 && (
        <>
          <div className="mt-3 text-xs uppercase tracking-wide text-[#7E8CA6]">
            records
            {detail.documentTotal > detail.documentNames.length
              ? ` (${detail.documentNames.length} of ${detail.documentTotal})`
              : ""}
          </div>
          <ul className="mt-1 flex flex-col gap-1">
            {detail.documentNames.map((name, i) => (
              <li key={`${name}-${i}`} className="truncate text-xs text-[#E9EEF6]" title={name}>
                {name}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
