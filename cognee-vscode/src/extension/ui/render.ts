import { parseAnswer, type Citation, type RecallResponseItem } from "../../core";
import { basenameOf } from "../paths";
import { extractSourcePath } from "./resolve";

export interface RenderedRecall {
  /** The combined prose answer, with Evidence blocks stripped. */
  answer: string;
  /** De-duplicated citations gathered from all result items. */
  citations: Citation[];
  /** True when no renderable answer was produced. */
  isEmpty: boolean;
}

/**
 * Collapse the recall response (a union of graph/context/session items) into a
 * single answer plus a flat, de-duplicated citation list for display.
 */
export function renderRecall(items: RecallResponseItem[]): RenderedRecall {
  const answers: string[] = [];
  const citations: Citation[] = [];

  for (const item of items) {
    const text = extractText(item);
    if (!text) {
      continue;
    }
    const parsed = parseAnswer(text);
    const trimmed = parsed.answer.trim();
    if (trimmed) {
      answers.push(trimmed);
    }
    citations.push(...parsed.citations);
  }

  return {
    answer: answers.join("\n\n"),
    citations: dedupeCitations(citations),
    isEmpty: answers.length === 0,
  };
}

/**
 * Collapse citations that point at the same source file into one entry, keeping
 * the first occurrence (so, applied after ranking, the best-matching chunk wins).
 *
 * Cognee emits one Evidence line per chunk, so a single file can appear several
 * times (e.g. three chunks of the same README). Showing every unique *file* once
 * is the useful, honest unit for provenance — all sources, no repeats. Files are
 * keyed by their `Source: <path>` header when present, else the document name.
 *
 * Only the first chunk of an ingested file carries the `Source:` header, so a
 * later chunk of that same file would key by its bare document name and slip
 * through as a duplicate. To avoid that, a headerless chunk whose document name
 * matches the basename of an already-kept `Source:` path is treated as the same
 * file — without merging genuinely distinct files that share a basename, since
 * those each carry their own distinct `Source:` path.
 */
export function dedupeByFile(citations: Citation[]): Citation[] {
  const seenKeys = new Set<string>();
  const seenSourceBasenames = new Set<string>();
  const unique: Citation[] = [];
  for (const citation of citations) {
    const sourcePath = extractSourcePath(citation.snippet);
    const key = (sourcePath ?? citation.documentName).toLowerCase();
    if (seenKeys.has(key)) {
      continue;
    }
    if (!sourcePath && seenSourceBasenames.has(citation.documentName.toLowerCase())) {
      continue;
    }
    seenKeys.add(key);
    if (sourcePath) {
      seenSourceBasenames.add(basenameOf(sourcePath).toLowerCase());
    }
    unique.push(citation);
  }
  return unique;
}

/**
 * Order citations by lexical relevance to the query, most relevant first.
 *
 * Cognee lists Evidence chunks in retrieval order and does not populate a
 * per-citation `score`, so a large, incidental document can appear above the one
 * the query is actually about. We rank client-side by how many distinct query
 * terms appear in each citation's document name and snippet (the snippet includes
 * our `Source: <path>` header, so a path like `themes/aurora/…` counts too). The
 * sort is stable: equal scores keep Cognee's original order, and this only
 * reorders the source list — never the answer.
 */
export function rankCitations(citations: Citation[], query: string): Citation[] {
  const terms = queryTerms(query);
  if (terms.length === 0 || citations.length < 2) {
    return citations;
  }
  return citations
    .map((citation, index) => ({ citation, index, score: relevanceScore(citation, terms) }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((entry) => entry.citation);
}

const STOP_WORDS = new Set([
  "how", "to", "the", "a", "an", "is", "are", "in", "of", "and", "or", "for", "on",
  "what", "why", "do", "does", "did", "i", "my", "me", "we", "it", "this", "that",
  "about", "with", "can", "should", "would", "please", "tell",
]);

/** Distinct, meaningful lower-cased query terms (stop words and stubs dropped). */
function queryTerms(query: string): string[] {
  const seen = new Set<string>();
  for (const token of query.toLowerCase().split(/[^a-z0-9]+/)) {
    if (token.length >= 3 && !STOP_WORDS.has(token)) {
      seen.add(token);
    }
  }
  return [...seen];
}

/** Count how many distinct query terms appear in the citation's name + snippet. */
function relevanceScore(citation: Citation, terms: string[]): number {
  const haystack = `${citation.documentName} ${citation.snippet ?? ""}`.toLowerCase();
  let score = 0;
  for (const term of terms) {
    if (haystack.includes(term)) {
      score += 1;
    }
  }
  return score;
}

function extractText(item: RecallResponseItem): string {
  if ("text" in item && typeof item.text === "string") {
    return item.text;
  }
  if ("content" in item && typeof item.content === "string") {
    return item.content;
  }
  if (item.source === "session" && typeof item.answer === "string") {
    return item.answer;
  }
  return "";
}

function dedupeCitations(citations: Citation[]): Citation[] {
  const seen = new Set<string>();
  const unique: Citation[] = [];
  for (const citation of citations) {
    if (seen.has(citation.raw)) {
      continue;
    }
    seen.add(citation.raw);
    unique.push(citation);
  }
  return unique;
}
