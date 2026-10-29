"use client";

import type { ReactElement } from "react";
import { BORDER_SOFT, TEXT, TEXT_MUTED, inputStyle } from "@/ui/elements/panelTokens";
import { STATUS_FILTERS, type StatusFilter } from "@/app/(app)/datasets/partials/datasetListFilter";

interface DatasetListFiltersProps {
  search: string;
  status: StatusFilter;
  onSearch: (value: string) => void;
  onStatus: (value: StatusFilter) => void;
}

/** Search and status chips above the dataset list. */
export default function DatasetListFilters({ search, status, onSearch, onStatus }: DatasetListFiltersProps): ReactElement {
  return (
    <div style={{ padding: "10px 12px", borderBottom: `1px solid ${BORDER_SOFT}`, display: "flex", flexDirection: "column", gap: 8, flexShrink: 0 }}>
      <input
        aria-label="Search datasets"
        placeholder="Search datasets"
        value={search}
        onChange={(event) => onSearch(event.target.value)}
        style={{ ...inputStyle, height: 32, fontSize: 13 }}
      />
      <div role="group" aria-label="Filter by status" className="flex" style={{ gap: 4 }}>
        {STATUS_FILTERS.map((filter) => {
          const active = filter.value === status;
          return (
            <button
              key={filter.value}
              aria-pressed={active}
              onClick={() => onStatus(filter.value)}
              style={{
                fontSize: 11,
                padding: "3px 8px",
                borderRadius: 6,
                cursor: "pointer",
                border: `1px solid ${active ? "rgba(255,255,255,0.25)" : BORDER_SOFT}`,
                background: active ? "rgba(255,255,255,0.10)" : "transparent",
                color: active ? TEXT : TEXT_MUTED,
              }}
            >
              {filter.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
