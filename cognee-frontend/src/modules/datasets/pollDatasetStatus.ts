import { captureException } from "@/utils/monitoring";
import { CogneeInstance } from "../instances/types";
import { INSUFFICIENT_CREDITS_REASON, normalizeDatasetStatusEntry } from "./datasetStatusDetail";

export type DatasetProcessingStatus =
  | "DATASET_PROCESSING_INITIATED"
  | "DATASET_PROCESSING_STARTED"
  | "DATASET_PROCESSING_COMPLETED"
  | "DATASET_PROCESSING_ERRORED";

const TERMINAL_STATUSES: DatasetProcessingStatus[] = [
  "DATASET_PROCESSING_COMPLETED",
  "DATASET_PROCESSING_ERRORED",
];

const IN_PROGRESS_STATUSES: DatasetProcessingStatus[] = [
  "DATASET_PROCESSING_INITIATED",
  "DATASET_PROCESSING_STARTED",
];

export class DatasetProcessingError extends Error {
  // Failure cause classified by the pod (e.g. INSUFFICIENT_CREDITS_REASON);
  // null when the pod could not classify it or predates CLO-306.
  readonly reason: string | null;

  constructor(datasetId: string, status: DatasetProcessingStatus, reason: string | null = null) {
    super(`Dataset ${datasetId} processing ended with status: ${status}`);
    this.name = "DatasetProcessingError";
    this.reason = reason;
  }
}

// The run died mid-pipeline because LiteLLM rejected an LLM call on the
// tenant's max_budget. The pre-flight check admitted the run, so no 402 ever
// reached the client — this is the only place the upload flow learns why.
// Subclasses DatasetProcessingError so existing instanceof checks still match.
export class InsufficientCreditsProcessingError extends DatasetProcessingError {
  constructor(datasetId: string, status: DatasetProcessingStatus) {
    super(datasetId, status, INSUFFICIENT_CREDITS_REASON);
    this.name = "InsufficientCreditsProcessingError";
  }
}

export class DatasetProcessingTimeoutError extends Error {
  constructor(datasetId: string, timeoutMs: number) {
    super(`Dataset ${datasetId} processing timed out after ${timeoutMs / 1000}s`);
    this.name = "DatasetProcessingTimeoutError";
  }
}

/**
 * Polls GET /v1/datasets/status every `intervalMs` until the dataset reaches
 * a terminal status (COMPLETED or ERRORED), then resolves with that status.
 *
 * Waits `initialDelayMs` before the first poll to give the backend time to
 * register the job. The "stale prior run" risk is handled with a baseline
 * snapshot: if the very first poll already returns COMPLETED/ERRORED *and*
 * we never see an in-progress status differ, we still resolve after
 * `staleConfirmAttempts` consecutive identical terminal reads — that way
 * fast pipelines (INITIATED → STARTED → COMPLETED inside one poll interval,
 * which happens regularly with runInBackground) don't hang the caller
 * forever. A genuinely stale prior-run status would have been overwritten
 * by INITIATED/STARTED by the time we confirm.
 */
export default function pollDatasetStatus(
  datasetId: string,
  instance: CogneeInstance,
  {
    intervalMs = 5000,
    initialDelayMs = 3000,
    timeoutMs = 10 * 60 * 1000,
    staleConfirmAttempts = 2,
    onStatus,
  }: {
    intervalMs?: number;
    initialDelayMs?: number;
    timeoutMs?: number;
    staleConfirmAttempts?: number;
    onStatus?: (status: DatasetProcessingStatus) => void;
  } = {},
): Promise<DatasetProcessingStatus> {
  return new Promise((resolve, reject) => {
    let sawInProgress = false;
    let terminalReadsInARow = 0;
    const deadline = Date.now() + timeoutMs;

    const check = async () => {
      if (Date.now() >= deadline) {
        const err = new DatasetProcessingTimeoutError(datasetId, timeoutMs);
        captureException(err, { datasetId, timeoutMs });
        reject(err);
        return;
      }

      try {
        const response = await instance.fetch(
          `/v1/datasets/status?dataset=${datasetId}&include_error_detail=true`,
        );
        if (!response.ok) {
          throw new Error(`Status check failed: ${response.status}`);
        }
        const data = await response.json();
        const entry = data[datasetId];
        // Bare string from pods predating CLO-306, {status, reason, error} otherwise.
        const detail = entry ? normalizeDatasetStatusEntry(entry) : null;
        const status = detail?.status;

        if (status) {
          onStatus?.(status);

          if (IN_PROGRESS_STATUSES.includes(status)) {
            sawInProgress = true;
            terminalReadsInARow = 0;
          } else if (TERMINAL_STATUSES.includes(status)) {
            terminalReadsInARow += 1;
            if (sawInProgress || terminalReadsInARow >= staleConfirmAttempts) {
              if (status === "DATASET_PROCESSING_ERRORED") {
                const err =
                  detail?.reason === INSUFFICIENT_CREDITS_REASON
                    ? new InsufficientCreditsProcessingError(datasetId, status)
                    : new DatasetProcessingError(datasetId, status, detail?.reason ?? null);
                captureException(err, { datasetId, status });
                reject(err);
                return;
              }
              resolve(status);
              return;
            }
          }
        }
      } catch (err) {
        // Transient failure (network blip, or the status row not written yet
        // right after a background /v1/remember returns) — the upload itself
        // already succeeded by this point, so keep polling instead of failing
        // the whole upload. Only the deadline check above ends the poll.
        const normalized = err instanceof Error ? err : new Error(String(err));
        captureException(normalized, { datasetId, transient: true });
      }

      setTimeout(check, intervalMs);
    };

    setTimeout(check, initialDelayMs);
  });
}
