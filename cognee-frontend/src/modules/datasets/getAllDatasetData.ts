import { CogneeInstance } from "../instances/types";
import getDatasetData from "./getDatasetData";

const PAGE_LIMIT = 1000;
// 200 pages * 1000 rows = 200k items, comfortably above any real dataset
// while still terminating against a backend bug or pathological data.
const MAX_PAGES = 200;

export interface DatasetDataRow {
  id: string;
  [key: string]: unknown;
}

export interface AllDatasetDataResult {
  rows: DatasetDataRow[];
  // True when MAX_PAGES was hit before a short page confirmed the list was
  // fully read — `rows` holds everything fetched so far, but that's not
  // proven to be everything the dataset has.
  capped: boolean;
}

function isDatasetDataRow(value: unknown): value is DatasetDataRow {
  return typeof value === "object" && value !== null && typeof (value as { id?: unknown }).id === "string";
}

// Pages through GET /data with limit=1000 for callers that need every row
// (rendering a full file list, or counting when /data/count isn't available).
// A mid-page fetch failure is left to propagate — swallowing it and returning
// whatever was read so far would silently truncate the list, exactly the bug
// this helper exists to avoid.
export async function fetchAllDatasetDataPages(
  datasetId: string,
  instance: CogneeInstance,
  signal?: AbortSignal,
): Promise<AllDatasetDataResult> {
  const rows: DatasetDataRow[] = [];
  const seenIds = new Set<string>();
  let firstRowId: string | null = null;

  for (let page = 0; page < MAX_PAGES; page++) {
    const offset = page * PAGE_LIMIT;
    const data = await getDatasetData(datasetId, instance, { limit: PAGE_LIMIT, offset, signal });
    const rawBatch = Array.isArray(data) ? data : [];

    if (rawBatch.length === 0) return { rows, capped: false };

    // Filtered for shape, but every pagination decision below reads
    // rawBatch.length, not batch.length: one malformed row on an otherwise
    // full page must not make the page look short and stop reading early.
    const batch = rawBatch.filter(isDatasetDataRow);

    if (batch.length > 0) {
      if (page === 0) {
        firstRowId = batch[0].id;
      } else if (batch[0].id === firstRowId) {
        // Pod predates 1.6.1 and ignores `offset` — every page returns the
        // same full list, so page 0 already had everything. Stop instead of
        // looping MAX_PAGES times appending duplicates of it.
        return { rows, capped: false };
      }
    }

    // Ordered by created_at DESC (see cognee's route contract): an insert
    // landing while this loop is mid-walk shifts every row after it down one
    // slot, so the row sitting at a page boundary can be handed back twice.
    // Dedupe by id instead of trusting offset-based paging to be exact.
    for (const row of batch) {
      if (seenIds.has(row.id)) continue;
      seenIds.add(row.id);
      rows.push(row);
    }

    if (rawBatch.length < PAGE_LIMIT) return { rows, capped: false };
  }

  console.warn(
    `getAllDatasetData: hit the ${MAX_PAGES}-page cap for dataset ${datasetId}; the list may be missing rows beyond page ${MAX_PAGES}.`,
  );
  return { rows, capped: true };
}

export default async function getAllDatasetData(
  datasetId: string,
  instance: CogneeInstance,
): Promise<DatasetDataRow[]> {
  const { rows } = await fetchAllDatasetDataPages(datasetId, instance);
  return rows;
}
