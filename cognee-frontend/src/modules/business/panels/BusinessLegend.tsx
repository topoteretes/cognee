"use client";

import { useState } from "react";
import { CODE_TYPE_COLORS, type ColorMode } from "../colorEncoding";

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
    case "table":
      return <span className="inline-block h-2 w-3 flex-none bg-[#43D9E8]" />;
    case "value":
      return <span className="mx-0.5 inline-block h-2 w-2 flex-none rotate-45 bg-[#43D9E8]" />;
    case "structure":
      return <span className="inline-block w-3 flex-none border-t border-dashed border-[#7E8CA6]" />;
    case "code":
      return <span className="inline-block h-3 w-3 flex-none rounded-[3px]" style={{ background: CODE_TYPE_COLORS.CodeModule }} />;
    case "amber":
      return <span className={`${base} border-2 border-[#F5A83C]`} />;
    case "answered":
      return <span className={`${base} border border-dashed border-[#43D9E8]`} />;
    case "double":
      return <span className={`${base} border border-[#E9EEF6] ring-1 ring-[#E9EEF6] ring-offset-1 ring-offset-[#1A2438]`} />;
    case "path":
      return <span className={`${base} border-2 border-[#56DB7D]`} />;
    case "orphan":
      return <span className={`${base} border border-dashed border-[#7E8CA6]`} />;
    default:
      return <span className="inline-block h-1 w-1 flex-none rounded-full bg-[#43D9E8]" />;
  }
}

const COLOR_LABELS: Record<ColorMode, string> = {
  type: "color = what kind of thing it is (the biggest types in this dataset, the rest gray; the same type can be another color in another dataset)",
  source: "color = which source it came from",
};

const COLOR_MODES: { mode: ColorMode; label: string }[] = [
  { mode: "type", label: "type" },
  { mode: "source", label: "source" },
];

function legendItems(colorMode: ColorMode): { kind: string; label: string }[] {
  return [
    { kind: "size", label: "size = importance" },
    { kind: "color", label: COLOR_LABELS[colorMode] },
    { kind: "code", label: "square = code (module, symbol and dependency each have their own color)" },
    { kind: "table", label: "rectangle = table; circle = row or other record" },
    { kind: "value", label: "diamond = cell value" },
    { kind: "structure", label: "faint dashed line = row belongs to a table or has a cell value; solid line = other recorded relationship" },
    { kind: "amber", label: "amber ring = part of the live answer" },
    { kind: "answered", label: "dashed ring = answered questions before" },
    { kind: "double", label: "double ring = spans sources / agent memory" },
    { kind: "path", label: "green ring = shortest path between two records" },
    { kind: "orphan", label: "faint dashed gray = no connections yet" },
    { kind: "dot", label: "drifting dots = relationships at work" },
  ];
}

interface BusinessLegendProps {
  colorMode: ColorMode;
  onColorModeChange: (mode: ColorMode) => void;
}

export default function BusinessLegend({ colorMode, onColorModeChange }: BusinessLegendProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="absolute bottom-3 left-3 z-10">
      {open && (
        <div className="mb-1.5 w-[248px] rounded-[10px] border border-[#2A3652] bg-[#1A2438] p-2.5 text-[10.5px] text-[#7E8CA6]">
          {legendItems(colorMode).map((item) => (
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
      <div className="mb-1.5 flex items-center gap-1.5 text-[10.5px] text-[#7E8CA6]">
        <span>color by</span>
        <div role="group" aria-label="Color the graph by" className="flex overflow-hidden rounded-[8px] border border-[#2A3652]">
          {COLOR_MODES.map(({ mode, label }) => (
            <button
              key={mode}
              type="button"
              aria-pressed={colorMode === mode}
              onClick={() => onColorModeChange(mode)}
              className={`px-2 py-[3px] ${
                colorMode === mode ? "bg-[#1A2438] text-[#E9EEF6]" : "hover:text-[#E9EEF6]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
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
