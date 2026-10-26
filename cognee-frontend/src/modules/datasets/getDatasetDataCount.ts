import { CogneeInstance } from "../instances/types";
import { HttpError } from "@/services/http/errors";
import { fetchAllDatasetDataPages } from "./getAllDatasetData";

interface DataCountResponse {
  count?: unknown;
}

const COUNT_ENDPOINT_MISSING_STATUSES = new Set([404, 405]);

function isDataCountResponse(value: unknown): value is DataCountResponse {
  return typeof value === "object" && value !== null;
}

// GET /data/count only exists on cognee 1.6.1+. A pod predating it doesn't
// 404 but 405s: the path matches its DELETE /{dataset_id}/data/{data_id}
// route with data_id="count", and Starlette answers a method mismatch on a
// known path with 405. Either status means "no count endpoint here". Every
// CogneeInstance.fetch implementation routes through http.request, which
// rejects with HttpError on any non-ok response instead of resolving (see
// toHttpError in services/http/client.ts), so that status is caught here,
// not read off response.ok. A dataset that genuinely doesn't exist 404s the
// same way and falls into the same fallback, where /data 404s too and that
// HttpError propagates to the caller.
//
// Returns null when the fallback itself got capped (see getAllDatasetData):
// a capped page-through is "at least this many", not an exact count, and
// presenting it as one would be worse than admitting it's unknown.
export default async function getDatasetDataCount(
  datasetId: string,
  instance: CogneeInstance,
  signal?: AbortSignal,
): Promise<number | null> {
  try {
    const response = await instance.fetch(`/v1/datasets/${datasetId}/data/count`, { signal });
    const body: unknown = await response.json();
    if (isDataCountResponse(body) && typeof body.count === "number") {
      return body.count;
    }
  } catch (error) {
    if (!(error instanceof HttpError) || !COUNT_ENDPOINT_MISSING_STATUSES.has(error.status)) {
      throw error;
    }
  }

  const { rows, capped } = await fetchAllDatasetDataPages(datasetId, instance, signal);
  return capped ? null : rows.length;
}
