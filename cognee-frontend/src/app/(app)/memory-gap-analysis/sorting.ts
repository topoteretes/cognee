import { impactOf } from "@/app/(app)/memory-gap-analysis/scoring";
import type { CoverageQuestion } from "@/app/(app)/memory-gap-analysis/types";

export type SortColumn = "question" | "coverage" | "relevance" | "topic";
export type SortDirection = "asc" | "desc";

/** `column: null` is the default order — impact, i.e. demand × shortfall. */
export interface SortState {
  column: SortColumn | null;
  direction: SortDirection;
}

export const DEFAULT_SORT: SortState = { column: null, direction: "desc" };

/** What the first click on a header should do — the direction people want. */
const FIRST_CLICK: Record<SortColumn, SortDirection> = {
  question: "asc",
  coverage: "asc",
  relevance: "desc",
  topic: "asc",
};

/** The direction a header will sort on its next click while inactive. */
export function firstDirection(column: SortColumn): SortDirection {
  return FIRST_CLICK[column];
}

/**
 * Up always means "the interesting end on top" — most asked, worst covered,
 * A first — so the glyph tracks each column's primary ordering rather than the
 * raw numeric direction, which would point down for Relevance.
 */
export function arrowFor(column: SortColumn, direction: SortDirection): "↑" | "↓" {
  return direction === FIRST_CLICK[column] ? "↑" : "↓";
}

/**
 * Cycles a column through its useful direction, the reverse, then back to the
 * impact default — so the ranked view is always one more click away.
 */
export function nextSort(current: SortState, column: SortColumn): SortState {
  if (current.column !== column) return { column, direction: FIRST_CLICK[column] };
  if (current.direction === FIRST_CLICK[column]) {
    return { column, direction: current.direction === "asc" ? "desc" : "asc" };
  }
  return DEFAULT_SORT;
}

function compare(a: CoverageQuestion, b: CoverageQuestion, column: SortColumn, topicLabels: ReadonlyMap<string, string>): number {
  switch (column) {
    case "question":
      return a.question_text.localeCompare(b.question_text);
    case "coverage":
      return a.judge_score - b.judge_score;
    case "relevance":
      return a.occurrence_count - b.occurrence_count;
    case "topic":
      return (topicLabels.get(a.topic_id) ?? a.topic_id).localeCompare(topicLabels.get(b.topic_id) ?? b.topic_id);
  }
}

export function sortQuestions(
  questions: CoverageQuestion[],
  sort: SortState,
  topicLabels: ReadonlyMap<string, string>,
): CoverageQuestion[] {
  if (sort.column === null) {
    return [...questions].sort((a, b) => impactOf(b) - impactOf(a) || a.judge_score - b.judge_score);
  }
  const column = sort.column;
  const factor = sort.direction === "asc" ? 1 : -1;
  // Impact breaks ties so equal-scoring rows keep a stable, meaningful order.
  return [...questions].sort((a, b) => factor * compare(a, b, column, topicLabels) || impactOf(b) - impactOf(a));
}

/** The user-facing sort control: worst score first, or most recalls first. */
export type SortMode = "score" | "recalls";

export function sortStateFor(mode: SortMode): SortState {
  if (mode === "recalls") return { column: "relevance", direction: "desc" };
  return { column: "coverage", direction: "asc" };
}
