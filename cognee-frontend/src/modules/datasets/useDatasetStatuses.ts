"use client";

import { useQuery, type QueryKey, type UseQueryResult } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import type { DatasetProcessingStatus } from "./pollDatasetStatus";

// Tenant-scoped so switching workspaces doesn't show a stale cache, and so
// upload/poll flows elsewhere can invalidate exactly this tenant's entry.
export function datasetStatusQueryKey(tenantId?: string): QueryKey {
  return ["dataset-statuses", tenantId ?? null];
}

interface UseDatasetStatusesResult {
  statuses: Record<string, DatasetProcessingStatus>;
  refetch: UseQueryResult["refetch"];
}

export function useDatasetStatuses(enabled: boolean): UseDatasetStatusesResult {
  const { cogniInstance } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();

  const { data, refetch } = useQuery({
    queryKey: datasetStatusQueryKey(tenant?.tenant_id),
    queryFn: async (): Promise<Record<string, DatasetProcessingStatus>> => {
      if (!cogniInstance) return {};
      const response = await cogniInstance.fetch("/v1/datasets/status");
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
      }
      return response.json();
    },
    // tenantReady, not just cogniInstance: the sidebar hides links to the
    // pages that use this hook (DatasetsPage, DatasetDetailPage,
    // knowledge-graph) while the pod isn't ready, but that only blocks
    // clicking through the nav — a direct URL, bookmark, or back/forward
    // navigation still mounts the page. Without this, the 5s interval below
    // hammers a genuinely unreachable pod indefinitely (no backoff, no
    // circuit breaker, unlike the one-shot background queries elsewhere).
    enabled: enabled && !!cogniInstance && tenantReady,
    refetchInterval: 5000,
    staleTime: 0,
    retry: false,
    throwOnError: false,
  });

  return { statuses: data ?? {}, refetch };
}
