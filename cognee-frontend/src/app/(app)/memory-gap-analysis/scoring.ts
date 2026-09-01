import { T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { MAX_SCORE, type CoverageQuestion, type ScoreBand } from "@/app/(app)/memory-gap-analysis/types";

const GAP_CEILING = 1.9;
const PARTIAL_CEILING = 3.9;

const BAND_COLOR: Record<ScoreBand, string> = { gap: T.red, partial: T.amber, covered: T.green };
const BAND_VERDICT: Record<ScoreBand, string> = { gap: "Gap", partial: "Partial", covered: "Covered" };

export function scoreBand(score: number): ScoreBand {
  if (score <= GAP_CEILING) return "gap";
  if (score <= PARTIAL_CEILING) return "partial";
  return "covered";
}

export function scoreColor(score: number): string {
  return BAND_COLOR[scoreBand(score)];
}

export function scoreVerdict(score: number): string {
  return BAND_VERDICT[scoreBand(score)];
}

/**
 * Demand × shortfall — roughly how many answers this gap spoiled over the
 * window. Ranking by it stops a once-asked miss outranking a daily one.
 */
export function impactOf(question: CoverageQuestion): number {
  return question.occurrence_count * (MAX_SCORE - question.judge_score);
}

export function scoreToPct(score: number): number {
  return Math.max(0, Math.min(100, (score / MAX_SCORE) * 100));
}

export function formatScore(score: number): string {
  return score.toFixed(1);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}
