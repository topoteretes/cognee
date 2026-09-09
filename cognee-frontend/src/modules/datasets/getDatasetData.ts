import { CogneeInstance } from "../instances/types";

/**
 * The server caps `/v1/datasets/{id}/data` at 100 items unless asked otherwise,
 * so a request without a limit returns a page, not the dataset. Anything that
 * needs a total must ask for one — see `getDatasetDataCount`.
 */
export const DEFAULT_DATASET_DATA_LIMIT = 100;

interface DatasetDataOptions {
  limit?: number;
  offset?: number;
  signal?: AbortSignal;
}

export default function getDatasetData(
  datasetId: string,
  instance: CogneeInstance,
  { limit = DEFAULT_DATASET_DATA_LIMIT, offset = 0, signal }: DatasetDataOptions = {},
) {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });

  return instance.fetch(`/v1/datasets/${datasetId}/data?${query}`, { signal })
      .then((response) => response.json());
}

/**
 * Number of documents in a dataset.
 *
 * Callers used to read `length` off the full row set to get this, which meant
 * transferring every row — tens of megabytes on a large dataset — to produce
 * one integer.
 */
export function getDatasetDataCount(
  datasetId: string,
  instance: CogneeInstance,
  signal?: AbortSignal,
): Promise<number> {
  return instance.fetch(`/v1/datasets/${datasetId}/data/count`, { signal })
      .then((response) => response.json())
      .then((body) => (typeof body?.count === "number" ? body.count : 0));
}
