import type { CogneeInstance } from "@/modules/instances/types";
import type { VisualizationPayload } from "./types";

// Matches the per-dataset graph size /brains used per entry, so the focused
// dataset renders with the same density the all-datasets payload gave it.
//
// Note how this sits against ACTIVE_CAP_ENTITY_THRESHOLD (500) in
// simulation/useViewportActiveIds.ts: entities are a subset of these nodes, so
// while this cap is <= that threshold ONE dataset can never contain enough
// entities to trigger viewport-capping — the fetch cap already delivers that
// optimization's win, and the capping path stays reachable for the case it was
// written for (several datasets merged into one scene, CLO-578). Raising this
// number therefore hands the tail of that work back to the viewport cap rather
// than making the canvas slower.
const MAX_NODES = 500;

// One focused dataset's content graph. Goes through the same cognee-core
// `preprocess()` as a /brains entry, so nodes/links/color_maps carry the
// exact fields computeBrainState and the canvas read — but for ONE dataset,
// instead of /brains' full graph for every readable dataset (CLO-597).
export default function getBrainGraph(
  instance: CogneeInstance,
  datasetId: string,
): Promise<VisualizationPayload> {
  const params = new URLSearchParams({ dataset_id: datasetId, max_nodes: String(MAX_NODES) });
  return instance.fetch(`/v1/visualize/json?${params.toString()}`).then((response) => response.json());
}
