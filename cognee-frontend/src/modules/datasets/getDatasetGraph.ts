import { CogneeInstance } from "@/modules/instances/types";

export default function getDatasetGraph(dataset: { id: string }, instance: CogneeInstance, signal?: AbortSignal, timeoutMs?: number) {
  const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs };
  return instance.fetch(`/v1/datasets/${dataset.id}/graph`, init)
      .then((response) => {
        if (!response.ok) throw new Error(`graph fetch failed: ${response.status}`);
        return response.json();
      });
}
