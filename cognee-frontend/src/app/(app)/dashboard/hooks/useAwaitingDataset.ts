"use client";

import { useState, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { notifications } from "@mantine/notifications";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import pollDatasetStatus, { DatasetProcessingError } from "@/modules/datasets/pollDatasetStatus";
import { datasetStatusQueryKey } from "@/modules/datasets/useDatasetStatuses";
import { getAwaitingDataset, clearAwaitingDataset } from "@/utils/browserStorage";

/**
 * Returns true while a freshly-provisioned default dataset (handed off from
 * onboarding via sessionStorage) is still processing.
 *
 * Any error or a missing dataset resolves to false so the UI is never blocked
 * indefinitely. A 30s safety timeout also guards against a pod that's still
 * starting (cogniInstance null) when this hook first runs.
 */
export function useAwaitingDataset(): boolean {
  const { cogniInstance } = useCogniInstance();
  const { tenant } = useTenant();
  const queryClient = useQueryClient();
  const [awaiting, setAwaiting] = useState<boolean>(() => getAwaitingDataset() !== null);

  useEffect(() => {
    const datasetId = getAwaitingDataset();
    if (!datasetId) return;

    let cancelled = false;
    // Set once the safety timeout gives up, so a slower pipeline rejection
    // that arrives afterward (pollDatasetStatus's own timeout is 10 minutes)
    // doesn't surface a "processing failed" toast for a wait the dashboard
    // already stopped tracking minutes earlier.
    let gaveUpEarly = false;
    const clear = () => {
      if (cancelled) return;
      clearAwaitingDataset();
      setAwaiting(false);
    };

    // Safety net: never block the dashboard longer than 30s regardless of pod
    // state. If cogniInstance is null (pod still starting), the poll below
    // never runs — this timeout prevents that deadlock.
    const safetyTimeout = setTimeout(() => {
      gaveUpEarly = true;
      clear();
    }, 30_000);

    if (!cogniInstance) {
      return () => {
        cancelled = true;
        clearTimeout(safetyTimeout);
      };
    }

    pollDatasetStatus(datasetId, cogniInstance, { intervalMs: 5000 })
      .finally(() => queryClient.invalidateQueries({ queryKey: datasetStatusQueryKey(tenant?.tenant_id) }))
      .then(clear, (err: unknown) => {
        // The dashboard must unblock either way, but a failed pipeline is not
        // a silent event — the user was told their dataset is being prepared.
        if (!cancelled && !gaveUpEarly) {
          notifications.show({
            color: "red",
            title: "Dataset processing failed",
            message:
              err instanceof DatasetProcessingError
                ? "Your dataset could not be processed. Try uploading it again."
                : "We could not confirm your dataset finished processing. Check the Brain page for its status.",
          });
        }
        clear();
      });

    return () => {
      cancelled = true;
      clearTimeout(safetyTimeout);
    };
  }, [cogniInstance, tenant?.tenant_id, queryClient]);

  return awaiting;
}
