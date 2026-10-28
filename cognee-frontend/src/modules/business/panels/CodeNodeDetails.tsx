"use client";

import type { CogneeInstance } from "@/modules/instances/types";
import type { BusinessGraphNode } from "../types";
import { codeFactEntity } from "../nodeInspection";
import { useCodeNodeDetails } from "../useCodeNodeDetails";
import NodeConnections from "./NodeConnections";

export function NodeSourceDetails({ node, scope = "import" }: { node: BusinessGraphNode; scope?: "import" | "view" }) {
  const description = typeof node.doc === "string" && node.doc.trim() ? node.doc :
    typeof node.description === "string" && node.description.trim() ? node.description : null;
  const file = typeof node.file_path === "string" ? node.file_path : typeof node.file === "string" ? node.file : null;
  const repo = typeof node.repo === "string" ? node.repo : null;
  const line = typeof node.line === "number" ? node.line : null;
  const endLine = typeof node.end_line === "number" ? node.end_line : null;
  return (
    <>
      <section className="mt-4" aria-label="Description">
        <h3 className="text-xs font-semibold text-[#E9EEF6]">Description</h3>
        <p className="mt-1 whitespace-pre-wrap break-words text-xs text-[#A4B1C8]">
          {description || (scope === "import" ? "No description is available in the imported data." : "No description is included in this view.")}
        </p>
      </section>
      <section className="mt-4" aria-label="Source location">
        <h3 className="text-xs font-semibold text-[#E9EEF6]">Source location</h3>
        {repo && <p className="mt-1 break-all text-xs text-[#A4B1C8]">Repository: {repo}</p>}
        {file ? (
          <p className="mt-1 break-all font-mono text-xs text-[#A4B1C8]">
            {file}{line !== null && <span className="whitespace-nowrap">:{line}{endLine !== null && endLine !== line ? `–${endLine}` : ""}</span>}
          </p>
        ) : <p className="mt-1 text-xs text-[#A4B1C8]">{scope === "import" ? "No file location is available in the imported data." : "No file location is included in this view."}</p>}
      </section>
    </>
  );
}

export default function CodeNodeDetails({ instance, datasetId, nodeId, onSelect }: {
  instance: CogneeInstance;
  datasetId: string;
  nodeId: string;
  onSelect: (node: BusinessGraphNode) => void;
}) {
  const query = useCodeNodeDetails(instance, datasetId, nodeId);
  const pages = query.data?.pages ?? [];
  const first = pages[0];
  const total = pages[pages.length - 1]?.total;
  const connections = pages.flatMap((page) => page.neighbors.map((entry) => ({
    ...entry, node: codeFactEntity(entry.node),
  })));
  return (
    <div aria-busy={query.isFetching}>
      {query.isPending && <p role="status" className="mt-4 text-xs text-[#A4B1C8]">Loading description, source and relationships…</p>}
      {first && (
        <>
          <NodeSourceDetails node={codeFactEntity(first.anchor)} />
          <section className="mt-4 border-t border-[#2A3652] pt-3" aria-label="Code relationships">
            <h3 className="text-xs font-semibold">Relationships in this dataset</h3>
            <p className="mt-1 text-[11px] text-[#A4B1C8]">
              {connections.length} of {total} relationships loaded. Expand a group to explore its items.
              {" "}→ goes from this item; ← comes into it.
            </p>
            {first.total === 0 && <p className="mt-2 text-xs text-[#A4B1C8]">No relationships were found for this item in the dataset’s code graph.</p>}
            <NodeConnections connections={connections} onSelect={onSelect} />
          </section>
        </>
      )}
      {query.isError && (
        <div role="alert" className="mt-4 rounded-lg border border-[#F5566B] p-3 text-xs">
          <p>{first ? "Could not load more relationships." : "Could not load this item’s code details."}</p>
          <p className="mt-1 break-words text-[#A4B1C8]">{query.error.message}</p>
          <button type="button" className="mt-2 cursor-pointer text-[#43D9E8] hover:underline"
            onClick={() => { if (first) void query.fetchNextPage(); else void query.refetch(); }}>Retry details</button>
        </div>
      )}
      {query.hasNextPage && !query.isError && (
        <button type="button" aria-disabled={query.isFetching}
          onClick={() => { if (!query.isFetching && query.hasNextPage) void query.fetchNextPage(); }}
          className="mt-3 w-full cursor-pointer rounded-lg border border-[#2A3652] px-3 py-2 text-xs text-[#43D9E8] aria-disabled:opacity-50">
          {query.isFetchingNextPage ? "Loading relationships…" : "Load more relationships"}
        </button>
      )}
    </div>
  );
}
