import { completedRunScore } from "@/app/(app)/memory-gap-analysis/aggregate";
import { MAX_SCORE, type Brain } from "@/app/(app)/memory-gap-analysis/types";
import type { TopicScore } from "@/app/(app)/dashboard/partials/redesign/PerformancePanel";

export interface CoverageView {
  /** One row per brain, in the order it was given. */
  topics: TopicScore[];
  /** Mean of the raw (unrounded) scores, rounded once at the end. Rounding
   *  each row first and then averaging the rounded numbers would compound
   *  the rounding error into the headline. Null when nothing is scored. */
  recallPct: number | null;
  /** True when the scores are missing because the fetch failed, not because
   *  nothing has been scored yet. See the CLO-726 note below. */
  unavailable: boolean;
}

/**
 * Turns a set of brains into what the dashboard's Memory Coverage panel
 * renders: a row per brain and a workspace headline.
 *
 * Pulled out of OverviewPage so it can be tested against `Brain[]` directly
 * instead of through the page's render. The brains passed in are exactly the
 * rows the panel shows. Nothing here slices or reorders them, so the caller
 * owns which brains this describes and in what order.
 */
export function deriveCoverageView(brains: Brain[]): CoverageView {
  const scored = brains.map((brain) => {
    const score = completedRunScore(brain.result);
    return { id: brain.id, name: brain.name, pct: score === null ? null : (score / MAX_SCORE) * 100 };
  });

  const topics: TopicScore[] = scored.map((entry) => ({
    id: entry.id,
    name: entry.name,
    // Null, not zero: a dataset nobody has scored yet is unknown, and drawing
    // it as 0% would read as "answers nothing" instead of "not measured".
    pct: entry.pct === null ? null : Math.round(entry.pct),
  }));

  // At least one brain failed and no brain is merely unscored, so the numbers
  // that are missing are missing because the fetch failed, not because nobody
  // has run an analysis. The panel has to say so rather than read as a
  // workspace nobody has scored (CLO-726). Brains that did load keep their own
  // scores on screen, so this can be true while a headline exists.
  const unavailable =
    brains.length > 0 &&
    brains.some((brain) => brain.state === "error") &&
    brains.every((brain, index) => brain.state === "error" || scored[index].pct !== null);

  const scoredPcts = scored.map((entry) => entry.pct).filter((pct): pct is number => pct !== null);
  const recallPct = scoredPcts.length
    ? Math.round(scoredPcts.reduce((total, pct) => total + pct, 0) / scoredPcts.length)
    : null;

  return { topics, recallPct, unavailable };
}
