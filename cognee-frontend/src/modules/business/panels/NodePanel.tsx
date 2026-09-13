"use client";

import type { BusinessEntity } from "../sceneTypes";
import type { SessionEvent } from "../types";
import { setsOf } from "../computeBrainState";
import { truncate } from "../textUtils";

interface NodePanelProps {
  entity: BusinessEntity | null;
  docCount: number;
  connectionCount: number;
  events: SessionEvent[];
  onClose: () => void;
  // The shortest-path trace (useShortestPath) — pathHops is null when no
  // second record has been shift-clicked yet, in which case the panel
  // hints at the gesture instead.
  pathTargetName: string | null;
  pathHops: number | null;
  onClearPath: () => void;
  // Runs computeWhatIfRemoval for this entity and surfaces it via the
  // existing spotlight/narration channel (BusinessView owns that logic —
  // this panel just triggers it).
  onSimulateRemoval: () => void;
}

// The click-to-select feature the ticket calls out as new, not ported: the
// canvas click handler used to only clear focus (customer_tutorial.html:6269).
// Selecting a node now opens this panel, which also does the reverse of the
// existing spotlight — "which answers used this node" — from the same
// node_ids data collect_session_events already carries, no new query.
export default function NodePanel({
  entity, docCount, connectionCount, events, onClose, pathTargetName, pathHops, onClearPath, onSimulateRemoval,
}: NodePanelProps) {
  if (!entity) return null;
  const sets = setsOf(entity);
  const usedIn = events.filter((e) => e.node_ids?.includes(entity.id));

  return (
    <div className="absolute right-4 top-16 z-10 w-72 rounded-xl border border-[#2A3652] bg-[#1A2438] p-4 text-sm text-[#E9EEF6]">
      <div className="flex items-start justify-between">
        <div className="font-semibold">{entity.name}</div>
        <button type="button" onClick={onClose} className="text-[#7E8CA6] hover:text-[#E9EEF6]">
          ✕
        </button>
      </div>
      <div className="mt-1 text-xs text-[#7E8CA6]">
        {entity.type || ""}
        {sets.length ? ` · from ${sets.join(", ")}` : ""}
        {docCount ? ` · seen in ${docCount} places` : ""}
        {connectionCount ? ` · ${connectionCount} connection${connectionCount === 1 ? "" : "s"}` : ""}
      </div>
      {/* The shortest-path trace — new in this port, see useShortestPath. */}
      {pathTargetName && pathHops !== null ? (
        <div className="mt-2 flex items-center justify-between rounded-[8px] border border-[#56DB7D] bg-[rgba(86,219,125,0.08)] px-2 py-1.5 text-[11px] text-[#56DB7D]">
          <span>path to {pathTargetName} — {pathHops} hop{pathHops === 1 ? "" : "s"}</span>
          <button type="button" onClick={onClearPath} className="text-[#56DB7D] hover:text-[#E9EEF6]">
            ✕
          </button>
        </div>
      ) : (
        <div className="mt-2 text-[10.5px] text-[#7E8CA6]">
          shift+click another record to trace the path between them
        </div>
      )}
      <button
        type="button"
        onClick={onSimulateRemoval}
        className="mt-2 w-full cursor-pointer rounded-[8px] border border-[#2A3652] px-2 py-1.5 text-left text-[11px] text-[#7E8CA6] transition-colors hover:border-[#F5566B] hover:text-[#F5566B]"
      >
        ⚠ what breaks without this record?
      </button>
      <div className="mt-3 text-xs uppercase tracking-wide text-[#7E8CA6]">
        Used in {usedIn.length} answer{usedIn.length === 1 ? "" : "s"}
      </div>
      {usedIn.length > 0 && (
        <ul className="mt-1 flex flex-col gap-2">
          {usedIn.map((e, i) => (
            <li key={`${e.qa_id ?? e.time ?? i}`} className="text-xs">
              <div className="font-medium text-[#E9EEF6]">
                ⌕ {truncate(e.question || "", 90)}
              </div>
              {e.time && <div className="text-[#7E8CA6]">{e.time}</div>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
