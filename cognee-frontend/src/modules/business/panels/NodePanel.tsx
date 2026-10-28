"use client";

import { useEffect, useId, useMemo, useRef } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import type { BusinessEntity, SemanticLink } from "../sceneTypes";
import type { BusinessGraphNode, SessionEvent } from "../types";
import { setsOf } from "../computeBrainState";
import { codeDisplayName, isCodeNode } from "../codeGraph";
import { inspectionType, loadedConnections } from "../nodeInspection";
import { truncate } from "../textUtils";
import { structuredTypeName } from "@/modules/business/structuredGraph";
import StructuredRowDetails from "./StructuredRowDetails";
import CodeNodeDetails, { NodeSourceDetails } from "./CodeNodeDetails";
import NodeConnections from "./NodeConnections";

interface NodePanelProps {
  entity: BusinessEntity | null;
  docCount: number;
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
  instance: CogneeInstance;
  // Null while no dataset is focused: governance nodes can still be inspected, code details need one.
  datasetId: string | null;
  graphLinks: SemanticLink[];
  docLinks: SemanticLink[];
  nodesById: Record<string, BusinessGraphNode>;
  onSelectEntity: (node: BusinessGraphNode) => void;
}

// The click-to-select feature the ticket calls out as new, not ported: the
// canvas click handler used to only clear focus (customer_tutorial.html:6269).
// Selecting a node now opens this panel, which also does the reverse of the
// existing spotlight — "which answers used this node" — from the same
// node_ids data collect_session_events already carries, no new query.
export default function NodePanel({
  entity, docCount, events, onClose, pathTargetName, pathHops, onClearPath, onSimulateRemoval,
  instance, datasetId, graphLinks, docLinks, nodesById, onSelectEntity,
}: NodePanelProps) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const opener = document.activeElement;
    panelRef.current?.focus({ preventScroll: true });
    return () => {
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, [entity?.id]);
  const entityId = entity?.id;
  // The header count and the list come from these same lists, so they cannot disagree.
  const graphConnections = useMemo(
    () => (entityId ? loadedConnections(entityId, graphLinks, nodesById) : []),
    [entityId, graphLinks, nodesById],
  );
  const recordConnections = useMemo(
    () => (entityId ? loadedConnections(entityId, docLinks, nodesById, "Source records") : []),
    [entityId, docLinks, nodesById],
  );
  if (!entity) return null;
  const sets = setsOf(entity);
  const usedIn = events.filter((e) => e.node_ids?.includes(entity.id));
  const fullName = String(entity.name ?? "");
  const displayName = codeDisplayName(entity);
  const structuredType = structuredTypeName(entity);
  // Same signal HoverTooltip uses: a code node's name shortens, a plain
  // entity's never does, so this is the actual "is there more to show" check.
  const showFullName = displayName !== fullName;
  const code = isCodeNode(entity);
  // Repositories and flows have separate CODE indexes, outside fact-neighbor reads.
  const codeDetails = code && entity.type !== "CodeRepository" && entity.type !== "CodeFlow" && datasetId !== null;
  const inLoadedGraph = Boolean(nodesById[entity.id]);
  const connectionCount = graphConnections.length;
  const connections = [...graphConnections, ...recordConnections];

  return (
    <div ref={panelRef} role="region" aria-labelledby={titleId} tabIndex={-1}
      onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); onClose(); } }}
      className="absolute right-4 top-16 bottom-24 z-20 w-[420px] max-w-[calc(100%_-_32px)] overflow-y-auto rounded-xl border border-[#2A3652] bg-[#1A2438] p-4 text-sm text-[#E9EEF6] focus-visible:outline-2 focus-visible:outline-[#43D9E8]">
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h2 id={titleId} className="break-words font-semibold">{displayName}</h2>
          {showFullName && (
            <div className="mt-1 break-all text-[11px] font-normal text-[#A4B1C8]">
              {fullName}
            </div>
          )}
        </div>
        <button type="button" aria-label="Close item details" onClick={onClose} className="ml-2 shrink-0 text-[#A4B1C8] hover:text-[#E9EEF6]">
          ✕
        </button>
      </div>
      <div className="mt-1 text-xs text-[#7E8CA6]">
        {structuredType || inspectionType(entity)}
        {!code && sets.length ? ` · from ${sets.join(", ")}` : ""}
        {docCount ? ` · linked to ${docCount} source records in this view` : ""}
        {inLoadedGraph ? ` · ${connectionCount} connection${connectionCount === 1 ? "" : "s"} in the loaded view` : " · outside the loaded graph"}
      </div>
      {/* The shortest-path trace — new in this port, see useShortestPath. */}
      {pathTargetName && pathHops !== null ? (
        <div className="mt-2 flex items-center justify-between rounded-[8px] border border-[#56DB7D] bg-[rgba(86,219,125,0.08)] px-2 py-1.5 text-[11px] text-[#56DB7D]">
          <span>path in this view to {pathTargetName} — {pathHops} hop{pathHops === 1 ? "" : "s"}</span>
          <button type="button" aria-label="Clear connection path" onClick={onClearPath} className="text-[#56DB7D] hover:text-[#E9EEF6]">
            ✕
          </button>
        </div>
      ) : (
        <div className="mt-2 text-[10.5px] text-[#7E8CA6]">
          shift+click another record to trace the path between them
        </div>
      )}
      <StructuredRowDetails node={entity} />
      {codeDetails ? (
        <CodeNodeDetails instance={instance} datasetId={datasetId!} nodeId={entity.id} onSelect={onSelectEntity} />
      ) : (
        <>
          {code && <NodeSourceDetails node={entity} scope="view" />}
          <section className="mt-4 border-t border-[#2A3652] pt-3" aria-label="Relationships in this view">
            <h3 className="text-xs font-semibold">Relationships in this view</h3>
            <p className="mt-1 text-[11px] text-[#A4B1C8]">Only loaded relationships are listed. Other dataset connections may not be shown. → goes from this item; ← comes into it.</p>
            {connections.length === 0 && <p className="mt-2 text-xs text-[#A4B1C8]">No relationships for this item are loaded in this view.</p>}
            <NodeConnections connections={connections} onSelect={onSelectEntity} />
          </section>
        </>
      )}
      {!structuredType && inLoadedGraph && (
        <button
          type="button"
          onClick={onSimulateRemoval}
          className="mt-2 w-full cursor-pointer rounded-[8px] border border-[#2A3652] px-2 py-1.5 text-left text-[11px] text-[#7E8CA6] transition-colors hover:border-[#F5566B] hover:text-[#F5566B]"
        >
          Preview disconnections in this view
        </button>
      )}
      <div className="mt-3 text-xs uppercase tracking-wide text-[#7E8CA6]">
        Used in {usedIn.length} answer{usedIn.length === 1 ? "" : "s"} in available activity
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
