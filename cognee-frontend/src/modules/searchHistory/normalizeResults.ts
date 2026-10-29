export interface SearchResultItem {
  search_result?: string[];
  dataset_id?: string;
  dataset_name?: string;
  question?: string;
  answer?: string;
  text?: string;
  raw?: { value?: string };
  kind?: string;
  origin_function?: string;
  method_return_value?: unknown;
  session_feedback?: string;
  _source?: string;
}

export const NO_RESULTS_TEXT = "No results found.";

/** The chat bubble text for a recall payload, whatever shape its rows take. */
export function normalizeResults(data: SearchResultItem[]): string {
  const parts: string[] = [];
  for (const raw of data) {
    const r = raw as SearchResultItem;
    // Recall response format: { text, raw: { value }, kind, ... }
    if (r.text) {
      parts.push(r.text);
    } else if (r.raw?.value) {
      parts.push(typeof r.raw.value === "string" ? r.raw.value : JSON.stringify(r.raw.value));
    } else if (Array.isArray(r.search_result)) {
      for (const x of r.search_result) {
        parts.push(typeof x === "string" ? x : JSON.stringify(x));
      }
    } else if (typeof r.search_result === "string") {
      parts.push(r.search_result);
    } else if (r.answer) {
      parts.push(r.answer);
    } else if (r.session_feedback) {
      parts.push(r.session_feedback);
    } else if (r.question) {
      parts.push(`Q: ${r.question}`);
    } else if (typeof raw === "string") {
      parts.push(raw as unknown as string);
    } else {
      try { parts.push(JSON.stringify(raw)); } catch { parts.push(String(raw)); }
    }
  }
  return parts.join("\n\n") || NO_RESULTS_TEXT;
}
