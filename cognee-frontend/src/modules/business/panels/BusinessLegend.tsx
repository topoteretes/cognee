"use client";

import { useState } from "react";

// The view piles up visual conventions (rings, colors, moving dots) that a
// first-time viewer has no way to decode — this is the decoder card, the
// same role Mindmap's bottom-left legend plays. Collapsed by default so it
// never competes with the scene it explains.
function Glyph({ kind }: { kind: string }) {
  const base = "inline-block h-3 w-3 flex-none rounded-full";
  switch (kind) {
    case "size":
      return (
        <span className="flex w-3 flex-none items-end justify-center gap-[1px]">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#8A7BD8]" />
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#8A7BD8]" />
        </span>
      );
    case "color":
      return <span className={`${base}`} style={{ background: "linear-gradient(135deg, #56DB7D 50%, #A456DB 50%)" }} />;
    case "amber":
      return <span className={`${base} border-2 border-[#F5A83C]`} />;
    case "answered":
      return <span className={`${base} border border-dashed border-[#43D9E8]`} />;
    case "double":
      return <span className={`${base} border border-[#E9EEF6] ring-1 ring-[#F5A83C] ring-offset-1 ring-offset-[#1A2438]`} />;
    case "path":
      return <span className={`${base} border-2 border-[#56DB7D]`} />;
    case "orphan":
      return <span className={`${base} border border-dashed border-[#7E8CA6]`} />;
    default:
      return <span className="inline-block h-1 w-1 flex-none rounded-full bg-[#43D9E8]" />;
  }
}

const LEGEND_ITEMS = [
  { kind: "size", label: "size = importance" },
  { kind: "color", label: "color = which source it came from" },
  { kind: "amber", label: "amber ring = part of the live answer" },
  { kind: "answered", label: "dashed ring = answered questions before" },
  { kind: "double", label: "double ring = spans sources / agent memory" },
  { kind: "path", label: "green ring = shortest path between two records" },
  { kind: "orphan", label: "faint dashed gray = no connections yet" },
  { kind: "dot", label: "drifting dots = relationships at work" },
];

export default function BusinessLegend() {
  const [open, setOpen] = useState(false);

  return (
    <div className="absolute bottom-3 left-3 z-10">
      {open && (
        <div className="mb-1.5 w-[248px] rounded-[10px] border border-[#2A3652] bg-[#1A2438] p-2.5 text-[10.5px] text-[#7E8CA6]">
          {LEGEND_ITEMS.map((item) => (
            <div key={item.kind} className="mb-1.5 flex items-center gap-2 last:mb-0">
              <Glyph kind={item.kind} />
              <span>{item.label}</span>
            </div>
          ))}
          <div className="mt-1.5 border-t border-[#2A3652] pt-1.5">
            click a record to focus its neighborhood · shift+click a second to trace the path between them
          </div>
        </div>
      )}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`rounded-[8px] border px-2.5 py-[3px] ${
          open ? "border-[#43D9E8] text-[#43D9E8]" : "border-[#2A3652] text-[#7E8CA6] hover:text-[#E9EEF6]"
        }`}
      >
        {open ? "✕ legend" : "? legend"}
      </button>
    </div>
  );
}
