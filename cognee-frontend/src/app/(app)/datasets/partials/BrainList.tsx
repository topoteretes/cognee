"use client";

import type { ReactElement } from "react";
import { Tooltip } from "@mantine/core";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import PlusIcon from "@/ui/elements/PlusIcon";

export type BrainStatus = "pending" | "running" | "completed" | "failed" | "empty" | "loading";

export interface BrainListItem {
  id: string;
  name: string;
  status: BrainStatus;
  documents: number;
}

const STATUS_DOT: Record<BrainStatus, string> = {
  pending: "#F59E0B",
  running: "#3B82F6",
  completed: "#22C55E",
  failed: "#EF4444",
  empty: "#D4D4D8",
  loading: "#D4D4D8",
};

const STATUS_LABEL: Record<BrainStatus, string> = {
  pending: "Pending",
  running: "Processing",
  completed: "Ready",
  failed: "Failed",
  empty: "Empty",
  loading: "Loading",
};

const STATUS_HINT: Record<BrainStatus, string> = {
  pending: "Queued, not started yet",
  running: "Building the knowledge graph",
  completed: "Processed and ready to query",
  failed: "Processing failed",
  empty: "No documents added yet",
  loading: "Loading",
};

const OUTDATED_DOT = "#F97316";
const OUTDATED_LABEL = "Outdated";
const OUTDATED_HINT = "Config changed — needs rebuilding";

export default function BrainList<T extends BrainListItem>({
  brains,
  selectedId,
  outdatedIds,
  onSelect,
  onCreate,
  onDelete,
}: {
  brains: T[];
  selectedId: string | null;
  outdatedIds: Set<string>;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (brain: T) => void;
}): ReactElement {
  return (
    <div style={{ width: 312, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Brain</span>
        <button onClick={onCreate} className="hover:bg-[#5A0ED6] cursor-pointer" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
          <PlusIcon /> New brain
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {brains.map((ds, i) => {
          const active = ds.id === selectedId;
          const statusLoading = ds.status === "loading";
          const docsLoadingRow = ds.documents < 0;
          const isOutdated = outdatedIds.has(ds.id);
          const dotColor = isOutdated ? OUTDATED_DOT : STATUS_DOT[ds.status];
          const statusLabel = isOutdated ? OUTDATED_LABEL : STATUS_LABEL[ds.status];
          const statusHint = isOutdated ? OUTDATED_HINT : STATUS_HINT[ds.status];
          return (
            <div key={ds.id} onClick={() => onSelect(ds.id)}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "8px 14px",
                borderBottom: i < brains.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none",
                cursor: "pointer",
                background: active ? "rgba(188,155,255,0.20)" : "transparent",
                userSelect: "none",
              }}
              onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
              onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
            >
              {statusLoading ? (
                <SkeletonBar width={7} height={7} />
              ) : (
                <Tooltip
                  label={<span><strong>{statusLabel}</strong> — {statusHint}</span>}
                  withArrow
                  position="top-start"
                  color="dark"
                  events={{ hover: true, focus: true, touch: true }}
                >
                  <span
                    role="status"
                    aria-label={statusLabel}
                    style={{ width: 7, height: 7, borderRadius: "50%", background: dotColor, flexShrink: 0, cursor: "help" }}
                  />
                </Tooltip>
              )}
              <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {ds.name}
              </span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", flexShrink: 0, minWidth: 16, textAlign: "right" }}>
                {docsLoadingRow ? <SkeletonBar width={14} height={8} /> : ds.documents}
              </span>
              {ds.name !== "default_dataset" && (
                <button
                  onClick={(e) => { e.stopPropagation(); onDelete(ds); }}
                  style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)", cursor: "pointer", flexShrink: 0 }}
                  title="Delete brain"
                >
                  Delete
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
