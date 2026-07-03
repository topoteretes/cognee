/**
 * Pure helpers for resolving a chunk/document-level citation to the right file.
 *
 * Citations carry only a document *basename* (Cognee's `document_name`), so when
 * a workspace has several files with that name (e.g. multiple `README.md`), we
 * disambiguate by checking which candidate actually contains the cited snippet.
 * The matching is whitespace-tolerant because chunking can normalize whitespace.
 *
 * These functions are `vscode`-free so they can be unit-tested and reused by the
 * planned JetBrains sidecar.
 */

/** Collapse runs of whitespace to single spaces and trim. */
export function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** The minimum snippet length worth matching on (avoids false positives). */
const MIN_MATCH_LENGTH = 8;

/** Longest prefix of the snippet used as a match needle. */
const MAX_NEEDLE_LENGTH = 200;

/**
 * True when `content` contains the cited `snippet`, comparing with normalized
 * whitespace so minor reformatting between ingest and now doesn't defeat it.
 */
export function contentMatchesSnippet(content: string, snippet: string): boolean {
  const needle = normalizeWhitespace(snippet).slice(0, MAX_NEEDLE_LENGTH);
  if (needle.length < MIN_MATCH_LENGTH) {
    return false;
  }
  return normalizeWhitespace(content).includes(needle);
}

/**
 * Build the ordered list of literal needles to search for when revealing a
 * snippet in an already-open document: the leading slice first, then the first
 * substantial line as a fallback. Empty/too-short candidates are dropped.
 */
export function snippetSearchNeedles(snippet: string): string[] {
  const leading = snippet.trim().slice(0, MAX_NEEDLE_LENGTH);
  const firstLine = (snippet.split("\n").find((line) => line.trim().length > 12) ?? "").trim();
  const seen = new Set<string>();
  const needles: string[] = [];
  for (const candidate of [leading, firstLine]) {
    if (candidate.length >= MIN_MATCH_LENGTH && !seen.has(candidate)) {
      seen.add(candidate);
      needles.push(candidate);
    }
  }
  return needles;
}
