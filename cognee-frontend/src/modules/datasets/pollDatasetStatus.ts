import { CogneeInstance } from "../instances/types";

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

/**
 * Polls GET /v1/datasets/status every `intervalMs` until the dataset reaches
 * a terminal status (COMPLETED or ERRORED), then resolves with that status.
 *
 * Waits `initialDelayMs` before the first poll to give the backend time to
 * register the job. Only treats COMPLETED/ERRORED as terminal after we have
 * seen an in-progress status (INITIATED or STARTED) at least once, to avoid
 * resolving on stale status from a previous run.
 */
export default function pollDatasetStatus(
  datasetId: string,
  instance: CogneeInstance,
  {
    intervalMs = 5000,
    initialDelayMs = 3000,
    onStatus,
  }: {
    intervalMs?: number;
    initialDelayMs?: number;
    onStatus?: (status: DatasetProcessingStatus) => void;
  } = {},
): Promise<DatasetProcessingStatus> {
  return new Promise((resolve, reject) => {
    let sawInProgress = false;

    const check = async () => {
      try {
        const response = await instance.fetch(
          `/v1/datasets/status?dataset=${datasetId}`,
        );
        if (!response.ok) {
          throw new Error(`Status check failed: ${response.status}`);
        }
        const data: Record<string, DatasetProcessingStatus> =
          await response.json();
        const status = data[datasetId];

        if (status) {
          onStatus?.(status);

          if (IN_PROGRESS_STATUSES.includes(status)) {
            sawInProgress = true;
          }

          // Only resolve on terminal status if we've confirmed processing started
          if (sawInProgress && TERMINAL_STATUSES.includes(status)) {
            resolve(status);
            return;
          }
        }

        setTimeout(check, intervalMs);
      } catch (err) {
        reject(err);
      }
    };

    // Wait before the first poll so the backend has time to register the job
    setTimeout(check, initialDelayMs);
  });
}
