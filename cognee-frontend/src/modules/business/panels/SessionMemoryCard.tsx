"use client";

import type { SessionEvent } from "../types";
import { truncate } from "../textUtils";
import { sourceLabel } from "../computeBrainState";

interface SessionMemoryCardProps {
  principalName: string | null;
  events: SessionEvent[];
  distilledSets: string[];
  onDismiss: () => void;
}

// Ports showSessionMemory (customer_tutorial.html ~7080-7100): clicking a
// user shows their own Q&A history plus whether improve() has distilled any
// of it into session_learnings / user_sessions_from_cache node sets. Session
// memory is the user's conversation history, not any one agent's — source
// scopes history to a clicked agent's own session_id, but this product has
// no per-agent session_id yet (and, so far, only ever one agent), so it
// shows the tenant's recent search history regardless of which user opened
// this — accurate today, gracefully approximate once that link exists.
export default function SessionMemoryCard({ principalName, events, distilledSets, onDismiss }: SessionMemoryCardProps) {
  if (!principalName) return null;
  const qas = events.filter((e) => (e.kind || "search") === "search" && e.question);

  return (
    <div
      className="absolute bottom-[84px] left-1/2 z-10 max-w-[560px] -translate-x-1/2 rounded-xl border px-4 py-3 backdrop-blur-sm"
      style={{ background: "rgba(26,36,56,.96)", borderColor: "rgba(245,168,60,.5)" }}
    >
      <button
        type="button"
        onClick={onDismiss}
        aria-label="dismiss"
        className="absolute right-2.5 top-1.5 text-sm text-[#7E8CA6] hover:text-[#E9EEF6]"
      >
        ✕
      </button>
      <div className="pr-4 text-[13px] font-semibold text-[#F5A83C]">{principalName} — session memory</div>
      <div className="mt-1 text-[11px] text-[#7E8CA6]">
        {qas.length ? `${qas.length} exchange${qas.length === 1 ? "" : "s"}` : "no session activity yet"}
      </div>
      {qas.length > 0 && (
        <div className="mt-1.5 max-h-[160px] overflow-y-auto text-[12px] leading-[1.5]">
          {qas.map((e, i) => (
            // qa_id/time alone aren't guaranteed unique — some backend
            // entries carry neither (manual asks, or events collapsed to
            // the same timestamp), which collided into duplicate React
            // keys. The index is stable here since `qas` is a filtered
            // snapshot of one render, never reordered independently.
            <div key={`${e.qa_id || e.time || "qa"}-${i}`} className="mb-2">
              <div className="font-semibold text-[#E9EEF6]">⌕ {truncate(String(e.question || ""), 90)}</div>
              <div className="text-[#7E8CA6]">{truncate(String(e.answer || ""), 140)}</div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-2 text-[11px]" style={{ color: distilledSets.length ? "#43D9E8" : "#7E8CA6" }}>
        {distilledSets.length
          ? `✦ distilled into the graph as: ${distilledSets.map(sourceLabel).join(", ")} — now lensed below`
          : "nothing distilled into the graph yet — run improve() with session_ids to turn this history into session_learnings"}
      </div>
    </div>
  );
}
