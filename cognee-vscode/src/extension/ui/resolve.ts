/**
 * Pure helpers for resolving a chunk/document-level citation to the right file.
 *
 * Citations carry only a document *basename* (Cognee's `document_name`), so when
 * a workspace has several files with that name (e.g. multiple `README.md`), we
 * disambiguate three ways, most reliable first:
 *   1. the `Source: <path>` provenance header the extension prepends at ingest,
 *      which names the exact file (see `extractSourcePath`);
 *   2. the file the user actually remembered (path index);
 *   3. the file whose content contains the cited snippet.
 * The matching is whitespace-tolerant because chunking can normalize whitespace.
 *
 * These functions are `vscode`-free so they can be unit-tested and reused by the
 * planned JetBrains sidecar.
 */

/**
 * The extension prepends a `Source: <relative/path>` header (optionally with a
 * `(lines a-b)` suffix) to everything it ingests. Matches that header at the
 * start of a snippet so callers can recover the exact source path and the file's
 * real leading text.
 */
const PROVENANCE_RE = /^\s*Source:\s*(\S+)(?:\s+\(lines\s+[^)]*\))?\s*/i;

/**
 * The exact source path recorded in the snippet's provenance header, if present.
 * Deterministic — no guessing among same-named files.
 */
export function extractSourcePath(snippet: string | undefined): string | undefined {
  const match = PROVENANCE_RE.exec(snippet ?? "");
  const path = match?.[1]?.trim();
  return path ? path : undefined;
}

/**
 * The snippet with the `Source: …` provenance header removed, so content matching
 * and in-file reveal run against the document's real text rather than the header.
 */
export function stripProvenanceHeader(snippet: string | undefined): string {
  return (snippet ?? "").replace(PROVENANCE_RE, "").trim();
}

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
 * Locate the cited snippet inside a document and return the `[start, end)`
 * character offsets of the match, or undefined when it can't be found.
 *
 * The server collapses each snippet's whitespace to single spaces and truncates
 * it with a trailing "…" (references.py), so a literal search against the file's
 * real text — with its newlines and indentation — would never match source code.
 * We search a whitespace-normalized copy of the document and project the match
 * back onto the raw text via an index map.
 */
export function findSnippetRange(text: string, snippet: string): [number, number] | undefined {
  const needle = normalizeWhitespace(snippet)
    .replace(/…+$/, "")
    .trim()
    .slice(0, MAX_NEEDLE_LENGTH);
  if (needle.length < MIN_MATCH_LENGTH) {
    return undefined;
  }

  // Normalized document text plus, for each normalized character, the raw offset
  // it came from — so a match in `normalized` maps back to a range in `text`.
  let normalized = "";
  const offsets: number[] = [];
  let pendingSpace = false;
  for (let i = 0; i < text.length; i += 1) {
    if (/\s/.test(text[i])) {
      pendingSpace = normalized.length > 0; // collapse runs; drop leading whitespace
      continue;
    }
    if (pendingSpace) {
      normalized += " ";
      offsets.push(i);
      pendingSpace = false;
    }
    normalized += text[i];
    offsets.push(i);
  }

  const start = normalized.indexOf(needle);
  if (start === -1) {
    return undefined;
  }
  return [offsets[start], offsets[start + needle.length - 1] + 1];
}
