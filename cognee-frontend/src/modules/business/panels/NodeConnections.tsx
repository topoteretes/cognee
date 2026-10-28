"use client";

import { useState } from "react";
import { codeDisplayName } from "../codeGraph";
import { connectionGroup, inspectionType, type InspectionConnection } from "../nodeInspection";
import type { BusinessGraphNode } from "../types";

// A hub node can hold thousands of rows: a group mounts none until it is opened,
// then shows them a page at a time.
const PAGE = 100;

function ConnectionGroup({ group, entries, onSelect }: {
  group: string;
  entries: InspectionConnection[];
  onSelect: (node: BusinessGraphNode) => void;
}) {
  const [open, setOpen] = useState(false);
  const [visible, setVisible] = useState(PAGE);
  return (
    <section aria-label={group}>
      <details open={open}>
        <summary onClick={(event) => { event.preventDefault(); setOpen((current) => !current); }}
          className="cursor-pointer text-xs font-semibold text-[#E9EEF6]">{group} · {entries.length} shown</summary>
        {open && (
          <ul className="mt-2 space-y-1">
            {entries.slice(0, visible).map(({ node, direction, relation }, index) => (
              <li key={`${direction}:${relation}:${node.id}:${index}`}>
                <button type="button" onClick={() => onSelect(node)}
                  className="w-full cursor-pointer rounded-lg border border-[#2A3652] px-2.5 py-2 text-left hover:border-[#43D9E8] focus-visible:outline-2 focus-visible:outline-[#43D9E8]">
                  <span className="block break-words font-medium text-[#E9EEF6]">{codeDisplayName(node) || node.id}</span>
                  {node.name && codeDisplayName(node) !== node.name && (
                    <span className="mt-0.5 block break-all text-[11px] text-[#A4B1C8]">{node.name}</span>
                  )}
                  <span className="mt-1 block break-words text-[11px] text-[#A4B1C8]">
                    {inspectionType(node)} · {direction === "out" ? "→" : "←"} {relation}
                  </span>
                </button>
              </li>
            ))}
            {entries.length > visible && (
              <li>
                <button type="button" onClick={() => setVisible((count) => count + PAGE)}
                  className="w-full cursor-pointer rounded-lg border border-[#2A3652] px-2.5 py-2 text-xs text-[#43D9E8]">
                  Show {Math.min(PAGE, entries.length - visible)} more of {entries.length - visible} remaining
                </button>
              </li>
            )}
          </ul>
        )}
      </details>
    </section>
  );
}

export default function NodeConnections({ connections, onSelect }: {
  connections: InspectionConnection[];
  onSelect: (node: BusinessGraphNode) => void;
}) {
  const groups = new Map<string, InspectionConnection[]>();
  for (const connection of connections) {
    const group = connectionGroup(connection);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)!.push(connection);
  }
  return (
    <div className="mt-3 space-y-4">
      {[...groups].map(([group, entries]) => (
        <ConnectionGroup key={group} group={group} entries={entries} onSelect={onSelect} />
      ))}
    </div>
  );
}
