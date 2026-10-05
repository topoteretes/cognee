import type { DatasetProcessingStatus } from "./pollDatasetStatus";

// Set by the pod's dataset-status wrapper (CLO-306/CLO-307) when a run failed
// specifically because the workspace ran out of credits mid-run — the only
// case that never produces a client-visible 402 (see CLO-252's acceptance
// notes), so this status chip is the only way a user finds out about it.
export const INSUFFICIENT_CREDITS_REASON = "insufficient_credits";

export interface DatasetStatusDetail {
  status: DatasetProcessingStatus;
  // Non-null only for a failed run whose cause the pod could classify (e.g.
  // INSUFFICIENT_CREDITS_REASON). Always null until the pod's
  // include_error_detail patch ships (CLO-306) — the pod ignores that query
  // param today and returns bare status strings, which normalizeStatusEntry
  // below already accounts for.
  reason: string | null;
}

// GET /v1/datasets/status returns a bare status string today; once CLO-306
// ships, requesting ?include_error_detail=true additionally returns
// {status, reason, error} objects — this normalizes either shape to one
// type so callers never need to branch on which pod version answered.
type RawDatasetStatusEntry =
  | DatasetProcessingStatus
  | { status: DatasetProcessingStatus; reason?: string | null; error?: string | null };

export function normalizeDatasetStatusEntry(raw: RawDatasetStatusEntry): DatasetStatusDetail {
  if (typeof raw === "string") return { status: raw, reason: null };
  return { status: raw.status, reason: raw.reason ?? null };
}

export function normalizeDatasetStatusResponse(
  raw: Record<string, RawDatasetStatusEntry>,
): Record<string, DatasetStatusDetail> {
  const result: Record<string, DatasetStatusDetail> = {};
  for (const [datasetId, entry] of Object.entries(raw)) {
    result[datasetId] = normalizeDatasetStatusEntry(entry);
  }
  return result;
}
