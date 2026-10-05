"use client";

import { useId, useState } from "react";
import type { BusinessGraphNode } from "../types";
import type { AccessSlot } from "../useGovernanceIndex";
import { permissionCode } from "../useGovernanceIndex";

interface AccessChipListProps {
  access: Record<string, AccessSlot>;
  datasets: BusinessGraphNode[];
  principalName: string;
  // Rings whichever chip matches the dataset currently on screen — with the
  // list wrapping to several rows once a principal has more than a couple
  // of grants, "what can THIS person do with what I'm looking at right now"
  // otherwise means hunting the row for a matching name.
  focusedDatasetId?: string | null;
}

const FOCUSED_RING = "ring-1 ring-[#43D9E8] ring-offset-1 ring-offset-[#1A2438]";

interface ChipProps {
  dataset: BusinessGraphNode;
  slot: AccessSlot | undefined;
  focused: boolean;
}

// Read-only: permission editing is a real API call on a future ticket, not
// something this panel simulates locally (see the module header comment).
function Chip({ dataset, slot, focused }: ChipProps) {
  const name = String(dataset.name || "dataset");
  if (!slot) {
    return (
      <span
        title={`${name}: no access`}
        className={`max-w-full truncate rounded border border-dashed border-[#2A3652] px-1.5 py-0.5 text-[9.5px] text-[#7E8CA6] opacity-60 ${
          focused ? FOCUSED_RING : ""
        }`}
      >
        + {name}
      </span>
    );
  }
  return (
    <span
      title={slot.owns ? `${name} — owner, full control` : name}
      className={`flex max-w-full items-baseline gap-1 rounded border px-1.5 py-0.5 text-[9.5px] ${
        slot.owns ? "border-[#43D9E8]/45 text-[#43D9E8]" : "border-[#2A3652] text-[#7E8CA6]"
      } ${focused ? FOCUSED_RING : ""}`}
    >
      <span className="truncate">{name}</span> <b className="shrink-0">{permissionCode(slot)}</b>
    </span>
  );
}

// Ports accessChips (customer_tutorial.html:6022-6037) as a read-only summary
// — granted datasets as solid chips, everything else as dashed "+ dataset"
// ghosts. Collapsed behind a disclosure by default: a principal who owns/
// reaches most of the workspace renders one chip per dataset, and an
// always-expanded block wraps into a chip stack tall enough to grow the card
// past the operators rail's own scroll region and collide with the cards
// below it (reported live: an owner of 10/11 datasets produced exactly
// this). "access · G/N ▸" keeps every card a predictable height while
// still making the full grant list one click away.
export default function AccessChipList({
  access, datasets, principalName, focusedDatasetId,
}: AccessChipListProps) {
  const [expanded, setExpanded] = useState(false);
  const listId = useId();
  // Nothing to disclose before any dataset exists: an "access · 0/0" row on
  // every card says less than no row at all.
  const hasDatasets = datasets.length > 0;
  // The disclosure counts GRANTS, not list length: the list is every dataset
  // in the workspace (grants first, then the "+ name" ghosts), so a bare
  // "access · 7" beside a principal with one grant out of seven read as full
  // access to all seven — the exact opposite of the truth.
  const granted = datasets.filter((d) => access[d.id]);
  // Granted access is the more informative half of the list, so it leads —
  // a principal with many grants and many ghosts sees their real access
  // first once expanded.
  const ordered = [...granted, ...datasets.filter((d) => !access[d.id])];

  if (!hasDatasets) return null;

  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setExpanded((prev) => !prev); }}
        aria-expanded={expanded}
        aria-controls={listId}
        aria-label={`${principalName} access: ${granted.length} of ${datasets.length} datasets`}
        className="text-[9.5px] text-[#7E8CA6] hover:text-[#F5A83C]"
      >
        access · {granted.length}/{datasets.length} {expanded ? "▾" : "▸"}
      </button>
      {expanded && (
        <div id={listId} className="mt-1 flex flex-wrap items-center gap-1">
          {ordered.map((d) => (
            <Chip key={d.id} dataset={d} slot={access[d.id]} focused={d.id === focusedDatasetId} />
          ))}
        </div>
      )}
    </div>
  );
}
