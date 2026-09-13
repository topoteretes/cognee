"use client";

export interface ConstructionLogEntry {
  id: string;
  text: string;
}

interface GraphConstructionLogProps {
  entries: ConstructionLogEntry[];
}

// A rolling "what just got added" feed — newest first, capped by the
// caller. Growth events already narrate a one-liner that fades in ~9s
// (useNarration); this keeps a short durable history of the same events so
// "what changed recently" survives past that fade instead of only being
// knowable at the instant it happened.
export default function GraphConstructionLog({ entries }: GraphConstructionLogProps) {
  if (!entries.length) return null;
  return (
    // left-2.5/w-[176px] (not left-0/w-[196px]) to align with the SourcesRail
    // cards above it — that rail's own wrapper is left-0 too, but SourcesRail
    // itself has p-2.5 padding, so its cards visually start ~10px in; this
    // box has to match that inset directly since it isn't nested the same
    // way (COG-6233).
    <div className="absolute left-2.5 bottom-24 h-[104px] w-[176px] overflow-hidden rounded-[10px] border border-[#2A3652] bg-[#1A2438]/90 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-widest text-[#7E8CA6]">activity</div>
      <div className="flex h-[78px] flex-col gap-0.5 overflow-y-auto text-[10px] text-[#7E8CA6]">
        {entries.map((e) => (
          <div key={e.id} className="truncate">{e.text}</div>
        ))}
      </div>
    </div>
  );
}
