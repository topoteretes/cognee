"use client";

import { useMemo } from "react";
import { useQuery, type QueryKey, type UseQueryResult } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import type { DatasetProcessingStatus } from "./pollDatasetStatus";
import { normalizeDatasetStatusResponse, type DatasetStatusDetail } from "./datasetStatusDetail";

// Tenant-scoped so switching workspaces doesn't show a stale cache, and so
// upload/poll flows elsewhere can invalidate exactly this tenant's entry.
export function datasetStatusQueryKey(tenantId?: string): QueryKey {
  return ["dataset-statuses", tenantId ?? null];
}

interface UseDatasetStatusesResult {
  // Bare status per dataset — unchanged shape, every existing caller keeps
  // working without touching a single comparison against a status string.
  statuses: Record<string, DatasetProcessingStatus>;
  // Additive: status + failure reason (e.g. "insufficient_credits", CLO-307),
  // for the handful of call sites that render a reason-specific status chip.
  statusDetails: Record<string, DatasetStatusDetail>;
  refetch: UseQueryResult["refetch"];
}

export function useDatasetStatuses(enabled: boolean): UseDatasetStatusesResult {
  const { cogniInstance } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();

  const { data, refetch } = useQuery({
    queryKey: datasetStatusQueryKey(tenant?.tenant_id),
    queryFn: async (): Promise<Record<string, DatasetStatusDetail>> => {
      if (!cogniInstance) return {};
      // include_error_detail is ignored by pods that predate CLO-306 (plain
      // bare-status response) — normalizeDatasetStatusResponse handles both.
      const response = await cogniInstance.fetch("/v1/datasets/status?include_error_detail=true");
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
      }
      return normalizeDatasetStatusResponse(await response.json());
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

  const statusDetails = useMemo(() => data ?? {}, [data]);
  const statuses = useMemo(() => {
    const result: Record<string, DatasetProcessingStatus> = {};
    for (const [datasetId, detail] of Object.entries(statusDetails)) {
      result[datasetId] = detail.status;
    }
    return result;
  }, [statusDetails]);

  return { statuses, statusDetails, refetch };
}
