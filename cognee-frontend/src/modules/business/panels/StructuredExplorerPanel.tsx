"use client";

import { useMemo } from "react";
import { Select } from "@mantine/core";
import type { RowFilter } from "@/modules/business/useStructuredExplorer";
import type { StructuredExplorer } from "@/modules/business/structuredExplorer";
import { truncate } from "@/modules/business/textUtils";

interface Props {
  index: StructuredExplorer;
  filter: RowFilter;
  onChange: (filter: RowFilter) => void;
  matchingCount: number | null;
}
const MAX_SUGGESTIONS = 5;

function connectionDescription(index: StructuredExplorer): string {
  if (index.sourceCount < 2) return "Cross-source comparison needs rows from at least two identified sources.";
  if (index.recordedRowLinks) return `${index.recordedRowLinks} recorded row-to-row links between sources in the loaded graph.`;
  if (index.sharedValueNodes) return "Sources share value nodes, but no direct row-to-row links are loaded.";
  return "No row-to-row links or shared value nodes between sources in the loaded graph.";
}

export default function StructuredExplorerPanel({ index, filter, onChange, matchingCount }: Props): React.JSX.Element | null {
  const values = filter.column === null ? undefined : index.columns.get(filter.column);
  const columnOptions = useMemo(() => [...index.columns.keys()].sort(), [index.columns]);
  const valueOptions = useMemo(() => [...(values ?? [])].sort(([a], [b]) => a.localeCompare(b)).map(([value, ids]) => ({
    value: JSON.stringify(value), label: `${value === "" ? "(empty)" : truncate(value, 120)} · ${ids.size} rows`,
  })), [values]);
  // Stays mounted while a filter is applied, even if a refresh left no rows,
  // so the filter can still be cleared.
  if (!index.rowIds.size && matchingCount === null) return null;
  return (
    <section aria-label="CSV and table explorer" className="absolute left-1/2 top-[70px] z-10 max-h-[calc(100%-190px)] w-[360px] -translate-x-1/2 overflow-y-auto rounded-[10px] border border-[#2A3652] bg-[#1A2438] p-2.5 text-[11px] text-[#E9EEF6]">
      <details className="peer">
        <summary className="cursor-pointer font-medium">CSV &amp; tables · {index.rowIds.size} loaded rows</summary>
        <div className="mt-2 space-y-2">
          <p className="text-[#7E8CA6]">Search columns and values, then select an exact value to filter. Only matching rows, their context nodes and existing links are shown. Source and type filters also apply.</p>
          <Select label="Column" placeholder="Search columns" searchable clearable size="xs"
            data={columnOptions} value={filter.column}
            onChange={(column) => onChange({ column, value: null })} nothingFoundMessage="No loaded columns" />
          <Select label="Value equals" placeholder="Search values" searchable clearable size="xs"
            disabled={filter.column === null} value={filter.value === null ? null : JSON.stringify(filter.value)}
            data={valueOptions}
            onChange={(value) => onChange({ column: filter.column, value: value === null ? null : JSON.parse(value) as string })}
            nothingFoundMessage="No loaded values" limit={100} />
          {index.unavailableRows > 0 && <p className="text-[#F5A83C]">Column values are unavailable for {index.unavailableRows} loaded rows; those rows cannot match a column filter.</p>}
          <div className="border-t border-[#2A3652] pt-2">
            <p>{connectionDescription(index)}</p>
            {index.sharedValueNodes > 0 && <p>{index.sharedValueNodes} shared value nodes connect rows across sources. These are structural paths, not inferred business relationships.</p>}
            <p className="mt-1 text-[#7E8CA6]">Equal names or cell values do not prove that processing linked the records. Creating relationships requires ingestion support. This comparison covers loaded data only; additional rows or links may exist.</p>
          </div>
          {index.possibleMatches.length > 0 && (
            <div>
              <p className="font-medium">Equal values across sources · {index.possibleMatches.length}</p>
              <p className="text-[#7E8CA6]">Possible matches only. Selecting one filters rows; it does not create a link.</p>
              {index.possibleMatches.slice(0, MAX_SUGGESTIONS).map((match) => (
                <button type="button" key={JSON.stringify([match.column, match.value])}
                  title={`${match.column} = ${match.value}\n${[...match.sources].join("\n")}`}
                  className="mt-1 block w-full truncate rounded border border-[#2A3652] px-2 py-1 text-left hover:border-[#43D9E8]"
                  onClick={() => onChange({ column: match.column, value: match.value })}>
                  {match.column} = {truncate(match.value, 60)} · {match.rowIds.size} rows
                </button>
              ))}
              {index.possibleMatches.length > MAX_SUGGESTIONS && <p className="text-[#7E8CA6]">Showing {MAX_SUGGESTIONS} examples. Use the column filter to explore all values.</p>}
            </div>
          )}
        </div>
      </details>
      {matchingCount !== null ? (
        <div className="mt-2 flex items-center justify-between gap-2" role="status">
          <span>{matchingCount === 0 ? "No loaded rows match this value." : `${matchingCount} matching loaded rows`} · {filter.column} = {truncate(filter.value ?? "", 40) || "(empty)"}</span>
          <button type="button" onClick={() => onChange({ column: null, value: null })} className="shrink-0 text-[#43D9E8]">Clear filter</button>
        </div>
      ) : <p className="mt-1 text-[#7E8CA6] peer-open:hidden">{connectionDescription(index)}</p>}
    </section>
  );
}
