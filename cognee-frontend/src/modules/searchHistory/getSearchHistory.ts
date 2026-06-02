import { CogneeInstance } from "../instances/types";

export interface SearchHistoryEntry {
  id: string;
  query: string;
  answer: string;
  dataset_id?: string;
  dataset_name?: string;
  created_at: string;
}

/**
 * Fetch search history from GET /v1/search.
 * Returns entries sorted by most recent first.
 */
export default async function getSearchHistory(
  instance: CogneeInstance,
): Promise<SearchHistoryEntry[]> {
  try {
    const r = await instance.fetch("/v1/search");
    if (!r.ok) return [];
    const data = await r.json();
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}
