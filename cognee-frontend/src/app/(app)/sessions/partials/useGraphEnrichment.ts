"use client";

import { useEffect, useState } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { getGraphEnrichmentRuns } from "@/modules/sessions/getSessions";
import type { EnrichmentRun } from "@/modules/sessions/getSessions";

export function useGraphEnrichment(
  datasetId: string | null,
  refreshNonce: number,
): { runs: EnrichmentRun[]; loading: boolean } {
  const { cogniInstance } = useCogniInstance();
  const [runs, setRuns] = useState<EnrichmentRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!cogniInstance || !datasetId) {
      setRuns([]);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    getGraphEnrichmentRuns(cogniInstance, datasetId).then((res) => {
      if (cancelled) return;
      setRuns(res);
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, [cogniInstance, datasetId, refreshNonce]);

  return { runs, loading };
}
