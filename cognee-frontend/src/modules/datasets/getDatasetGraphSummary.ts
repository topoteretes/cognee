import { CogneeInstance } from "@/modules/instances/types";

export interface DatasetGraphSummary {
  datasetId: string;
  pipelineRunId: string | null;
  numNodes: number;
  numEdges: number;
  // null while pipelineRunId is set means the last count attempt degraded
  // (graph store unavailable) and wasn't cached — the backend retries on
  // the next poll, so callers should treat the counts as possibly stale.
  computedAt: string | null;
}

// Precomputed counts backed by GraphMetrics (see COG-5726) — orders of
// magnitude cheaper than /graph, which does a full node/edge traversal.
// Omit datasetIds to get every dataset the caller can read; foreign or
// unknown ids in datasetIds are silently omitted from the response.
export default function getDatasetGraphSummary(
  instance: CogneeInstance,
  datasetIds?: string[],
  signal?: AbortSignal,
  timeoutMs?: number,
): Promise<DatasetGraphSummary[]> {
  const query = datasetIds?.length
    ? `?${datasetIds.map((id) => `dataset_ids=${encodeURIComponent(id)}`).join("&")}`
    : "";
  const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs };
  return instance.fetch(`/v1/datasets/graph-summary${query}`, init).then((r) => r.json());
}
