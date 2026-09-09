"use client";

import type { NarrationDisplay } from "../useNarration";
import TourControl from "./TourControl";

const ALTIMETER_LABELS = ["Business", "Players", "Connections", "Records"];

interface BusinessDockProps {
  narration: NarrationDisplay;
  altimeter: { level: number; plumbing: boolean };
  onAltimeterLevel: (level: number) => void;
  live: boolean;
  tourPlaying: boolean;
  onTourStart: () => void;
  onTourStop: () => void;
  // Shown on the Records button — unlike the three zoom levels, Records is
  // a toggle whose only visible effect is extra dots, so a count is the one
  // signal that tells the user it has data behind it before they click.
  recordCount: number;
}

// The bottom dock: narration line + altimeter + live indicator + tour —
// extracted from BusinessView unchanged. The container is pointer-events-
// none (only the controls row restores them) so clicks in the dock's empty
// corners fall through to the canvas underneath — that's where the minimap
// lives, and it has to stay clickable.
export default function BusinessDock({
  narration, altimeter, onAltimeterLevel, live, tourPlaying, onTourStart, onTourStop, recordCount,
}: BusinessDockProps) {
  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center gap-1.5 px-4 pb-2.5"
      style={{ background: "linear-gradient(transparent, rgba(14,21,38,.96) 55%)" }}
    >
      {narration.text && (
        <div
          role="status"
          aria-live="polite"
          className="h-6 text-center text-[13px] transition-opacity duration-[400ms]"
          style={{ color: narration.color, opacity: narration.opacity }}
        >
          {narration.text}
        </div>
      )}
      <div className="pointer-events-auto flex items-center gap-2.5">
        <div className="flex items-center gap-0.5 rounded-[8px] border border-[#2A3652] bg-[#1A2438] p-[3px]">
          {ALTIMETER_LABELS.map((label, level) => {
            const active = altimeter.plumbing ? level === 3 : altimeter.level === level;
            const isRecords = level === 3;
            return (
              <button
                key={label}
                type="button"
                onClick={() => onAltimeterLevel(level)}
                title={isRecords ? "toggle the raw records layer — chunks, documents and summaries behind the entities" : undefined}
                className={`rounded-[6px] px-2.5 py-[3px] ${
                  active ? "bg-[#141D33] text-[#E9EEF6]" : "text-[#7E8CA6] hover:text-[#E9EEF6]"
                }`}
              >
                {label}
                {isRecords && recordCount > 0 && (
                  <span className={`ml-1 text-[10px] ${active ? "text-[#43D9E8]" : "text-[#5B6880]"}`}>{recordCount}</span>
                )}
              </button>
            );
          })}
        </div>
        <span
          role="status"
          aria-label={live ? "Live updates connected" : "Live updates reconnecting"}
          className={`text-[11px] ${live ? "text-[#43D9E8]" : "text-[#7E8CA6]"}`}
        >
          {live ? "● LIVE" : "○ live: reconnecting…"}
        </span>
        <TourControl isPlaying={tourPlaying} onStart={onTourStart} onStop={onTourStop} />
      </div>
    </div>
  );
}
