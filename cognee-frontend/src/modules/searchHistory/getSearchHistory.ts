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
 * Legacy search history from GET /v1/search — single Q&A pairs recorded
 * before search conversations became sessions. Kept as a read-only,
 * supplementary sidebar source until these entries age out of Redis.
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
