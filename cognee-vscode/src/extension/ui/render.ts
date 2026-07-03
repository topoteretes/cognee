import { parseAnswer, type Citation, type RecallResponseItem } from "../../core";

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
