"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";

const PROCESSING_POLL_MS = 5000;
const PROCESSING_TIMEOUT_MS = 25_000;

export interface DatasetProcessing {
  total: number;
  completed: number;
  pending: number;
  items: { id: string; name: string; completed: boolean }[];
}

// Stored item completion survives navigation, connector dialogs, and server
// restarts. A run's status alone cannot tell how much of a brain is searchable.
export function useDatasetProcessing(datasetId: string | null): UseQueryResult<DatasetProcessing> {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  return useQuery({
    queryKey: ["dataset-processing", tenant?.tenant_id ?? null, datasetId],
    queryFn: async ({ signal }): Promise<DatasetProcessing> => {
      if (!cogniInstance) throw new Error("Workspace is not ready");
      const response = await cogniInstance.fetch(`/v1/datasets/${datasetId}/processing-status`, { signal, timeoutMs: PROCESSING_TIMEOUT_MS });
      if (!response.ok) throw new Error("Could not load processing progress");
      return response.json();
    },
    enabled: !!datasetId && !!cogniInstance && !isInitializing && tenantReady,
    refetchInterval: PROCESSING_POLL_MS,
    retry: false,
    staleTime: 0,
  });
}
