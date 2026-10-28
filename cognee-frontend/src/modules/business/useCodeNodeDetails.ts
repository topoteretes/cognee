"use client";

import { useInfiniteQuery } from "@tanstack/react-query";
import { getCodeNeighbors } from "@/modules/datasets/searchCode";
import type { CogneeInstance } from "@/modules/instances/types";

export function useCodeNodeDetails(instance: CogneeInstance, datasetId: string, nodeId: string) {
  return useInfiniteQuery({
    queryKey: ["business-code-node", instance.instanceId, datasetId, nodeId],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => getCodeNeighbors(instance, datasetId, nodeId, pageParam, signal),
    getNextPageParam: (page) => page.has_more ? page.offset + page.neighbors.length : undefined,
    // Re-selecting a node must not re-read every page it already loaded.
    staleTime: 60_000,
    retry: false,
    throwOnError: false,
    // Details are read on selection; background graph refreshes do not need another code snapshot read.
    refetchOnWindowFocus: false,
  });
}
