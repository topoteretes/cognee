import { captureException } from "@/utils/monitoring";
import { CogneeInstance } from "../instances/types";
import getDatasetData from "./getDatasetData";

export class WaitForDocsTimeoutError extends Error {
  constructor(datasetId: string, minCount: number) {
    super(`Timed out waiting for ${minCount} document(s) in dataset ${datasetId}`);
    this.name = "WaitForDocsTimeoutError";
  }
}

export default async function waitForDatasetDocs<T = unknown>(
  datasetId: string,
  instance: CogneeInstance,
  minCount: number,
  {
    intervalMs = 3000,
    timeoutMs = 90000,
    backgroundTimeoutMs = 10 * 60 * 1000,
    onTimeout,
  }: {
    intervalMs?: number;
    timeoutMs?: number;
    backgroundTimeoutMs?: number;
    onTimeout?: (docs: T[]) => void;
  } = {},
): Promise<T[]> {
  const deadline = Date.now() + timeoutMs;
  let latest: T[] = [];

  for (;;) {
    try {
      const data = await getDatasetData(datasetId, instance);
      latest = Array.isArray(data) ? data : [];
      if (latest.length >= minCount) return latest;
    } catch {
      // transient fetch failure — keep polling until the deadline
    }

    if (Date.now() >= deadline) {
      const error = new WaitForDocsTimeoutError(datasetId, minCount);
      captureException(error, { datasetId, minCount, timeoutMs });

      if (onTimeout) {
        pollInBackground(datasetId, instance, minCount, intervalMs, backgroundTimeoutMs, onTimeout);
      }

      throw error;
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function pollInBackground<T>(
  datasetId: string,
  instance: CogneeInstance,
  minCount: number,
  intervalMs: number,
  timeoutMs: number,
  onDone: (docs: T[]) => void,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    if (Date.now() >= deadline) return;
    try {
      const data = await getDatasetData(datasetId, instance);
      const docs = Array.isArray(data) ? (data as T[]) : [];
      if (docs.length >= minCount) {
        onDone(docs);
        return;
      }
    } catch {
      // ignore transient failures
    }
  }
}
