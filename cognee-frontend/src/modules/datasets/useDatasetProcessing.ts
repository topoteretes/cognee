"use client";

import { useQuery } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";

export interface DatasetProcessing {
  total: number;
  completed: number;
  pending: number;
  items: { id: string; name: string; completed: boolean }[];
}

// Stored item completion survives navigation, connector dialogs, and server
// restarts. A run's status alone cannot tell how much of a brain is searchable.
export function useDatasetProcessing(datasetId: string | null) {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  return useQuery({
    queryKey: ["dataset-processing", tenant?.tenant_id ?? null, datasetId],
    queryFn: async ({ signal }): Promise<DatasetProcessing> => {
      const response = await cogniInstance!.fetch(`/v1/datasets/${datasetId}/processing-status`, { signal, timeoutMs: 25_000 });
      if (!response.ok) throw new Error("Could not load processing progress");
      return response.json();
    },
    enabled: !!datasetId && !!cogniInstance && !isInitializing && tenantReady,
    refetchInterval: 5000,
    retry: false,
    staleTime: 0,
  });
}
