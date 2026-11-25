"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { projectStructuredRows } from "./projectStructuredRows";
import { buildStructuredExplorer, type StructuredExplorer } from "./structuredExplorer";
import type { BrainState } from "./sceneTypes";

export interface RowFilter {
  column: string | null;
  value: string | null;
}
const EMPTY_FILTER: RowFilter = { column: null, value: null };

// `enabled` false skips building the explorer index, which only the explorer
// panel reads; the projection `view` is still returned for the canvas.
export function useStructuredExplorer(brain: BrainState | null, datasetId: string | null, enabled = true): {
  index: StructuredExplorer;
  filter: RowFilter;
  setFilter: (filter: RowFilter) => void;
  rowIds: Set<string> | null;
  filterKey: string;
  view: BrainState | null;
} {
  const index = useMemo(() => buildStructuredExplorer(enabled ? brain : null), [brain, enabled]);
  const [selection, setSelection] = useState<{ datasetId: string | null; filter: RowFilter } | null>(null);
  useEffect(() => setSelection(null), [datasetId]);
  const filter = selection?.datasetId === datasetId ? selection.filter : EMPTY_FILTER;
  const setFilter = useCallback((next: RowFilter) => setSelection({ datasetId, filter: next }), [datasetId]);
  const rowIds = useMemo(() => {
    if (filter.column === null || filter.value === null) return null;
    return index.columns.get(filter.column)?.get(filter.value) ?? new Set<string>();
  }, [index, filter]);
  const view = useMemo(() => brain ? projectStructuredRows(brain, rowIds) : null, [brain, rowIds]);
  return { index, filter, setFilter, rowIds, view, filterKey: rowIds === null ? "" : JSON.stringify(filter) };
}
