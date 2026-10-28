import type { VisualizationPayload } from "../types";

const SHOWN = 5;
const names = (values: Set<string>) => {
  const all = [...values];
  return all.slice(0, SHOWN).join(", ") + (all.length > SHOWN ? ` and ${all.length - SHOWN} more` : "");
};

export default function GraphReadScope({ graph, sources, types }: {
  graph: VisualizationPayload | null;
  sources: Set<string> | null;
  types: Set<string> | null;
}) {
  if (!graph) return null;
  return (
    <div className="mx-2.5 mb-2 rounded-lg border border-[#2A3652] bg-[#1A2438] px-2.5 py-2 text-[11px] text-[#A4B1C8]">
      <details>
        <summary className="cursor-pointer text-[#E9EEF6]">Loaded view · {graph.nodes.length} dataset nodes</summary>
        <p className="mt-1">{graph.links.length} links loaded. Source and type counts below describe this neighborhood of the dataset. Governance nodes are drawn on top and not counted here.</p>
      </details>
      <p className="mt-1">Other dataset items may not be loaded.</p>
      {(sources || types) && (
        <p className="mt-1 break-words">Filters: {[sources ? `source: ${names(sources)}` : "", types ? `type: ${names(types)}` : ""].filter(Boolean).join(" · ")}</p>
      )}
    </div>
  );
}
