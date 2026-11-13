import { CogneeInstance } from "../instances/types";

export const DEFAULT_DATASET_DATA_LIMIT = 100;

export interface GetDatasetDataOptions {
  limit?: number;
  offset?: number;
  signal?: AbortSignal;
}

// Return type is `unknown`, not the DataDTO shape: the response can also be an
// error body (e.g. a 404 ErrorResponseDTO), and every caller already narrows
// with its own `Array.isArray` check before touching fields.
//
// `limit`/`offset` are always sent explicitly, never left to the backend
// default: cognee <1.6.1 ignores both and returns every row, cognee 1.6.1+
// defaults limit to 100. Sending them keeps both pod versions on the same
// page size during the rollout instead of one silently truncating.
export default function getDatasetData(
  datasetId: string,
  instance: CogneeInstance,
  { limit = DEFAULT_DATASET_DATA_LIMIT, offset = 0, signal }: GetDatasetDataOptions = {},
): Promise<unknown> {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return instance
    .fetch(`/v1/datasets/${datasetId}/data?${query.toString()}`, { signal })
    .then((response) => response.json());
}
