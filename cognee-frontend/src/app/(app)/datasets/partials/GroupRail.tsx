"use client";

import type { ReactElement } from "react";
import GroupAvatar from "@/ui/elements/GroupAvatar";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import { BORDER, TEXT, TEXT_FAINT, TEXT_MUTED, WEIGHT_BOLD, eyebrowStyle } from "@/ui/elements/panelTokens";
import type { DatasetFacet } from "@/app/(app)/datasets/partials/datasetFacets";
import type { FacetsStatus } from "@/app/(app)/datasets/partials/useDatasetFacets";

interface GroupRailProps {
  facets: DatasetFacet[];
  selectedId: string;
  status: FacetsStatus;
  /** Datasets on the page: what "All datasets" counts, even before (or
   *  without) the access list that the other entries are built from. */
  allCount: number;
  onSelect: (facetId: string) => void;
}

const RAIL_WIDTH = 220;
const SKELETON_ROWS = 4;

function FacetButton({ facet, count, active, onSelect }: { facet: DatasetFacet; count: number; active: boolean; onSelect: (id: string) => void }): ReactElement {
  return (
    <button
      aria-pressed={active}
      onClick={() => onSelect(facet.id)}
      className={active ? "flex items-center gap-2 w-full text-left" : "flex items-center gap-2 w-full text-left hover:bg-white/[0.06]"}
      style={{ padding: "8px 12px", border: "none", cursor: "pointer", background: active ? "rgba(188,155,255,0.20)" : "transparent" }}
    >
      <GroupAvatar name={facet.label} size={26} />
      <span className="flex-1 min-w-0">
        <span className="block truncate" style={{ fontSize: 13, fontWeight: WEIGHT_BOLD, color: TEXT }}>{facet.label}</span>
        <span className="block truncate" style={{ fontSize: 11, color: TEXT_FAINT }}>{facet.hint}</span>
      </span>
      <span style={{ fontSize: 11, color: TEXT_MUTED, fontVariantNumeric: "tabular-nums" }}>{count}</span>
    </button>
  );
}

/** Filters over the one dataset list: everything, mine, shared with me, and
 *  one entry per group I am in that reaches a dataset. */
export default function GroupRail({ facets, selectedId, status, allCount, onSelect }: GroupRailProps): ReactElement {
  // Without the access list only "All" is true; the rest would all read 0.
  const shown = status === "error" ? facets.filter((facet) => facet.kind === "all") : facets;

  return (
    <nav aria-label="Dataset groups" style={{ width: RAIL_WIDTH, flexShrink: 0, borderRight: `1px solid ${BORDER}`, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ height: 44, padding: "0 14px", borderBottom: `1px solid ${BORDER}`, display: "flex", alignItems: "center", flexShrink: 0 }}>
        <span style={{ ...eyebrowStyle, fontSize: 11, color: TEXT_MUTED }}>Groups</span>
      </div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {status === "loading" && Array.from({ length: SKELETON_ROWS }, (_, index) => (
          <div key={index} style={{ padding: "12px 14px" }}><SkeletonBar width="70%" height={12} /></div>
        ))}
        {status !== "loading" && shown.map((facet) => (
          <FacetButton key={facet.id} facet={facet} count={facet.kind === "all" ? allCount : facet.datasetIds.size}
            active={facet.id === selectedId} onSelect={onSelect} />
        ))}
        {status === "error" && (
          <p style={{ padding: "10px 14px", margin: 0, fontSize: 11, color: TEXT_FAINT }}>Groups could not be loaded; every dataset is still listed.</p>
        )}
      </div>
    </nav>
  );
}
