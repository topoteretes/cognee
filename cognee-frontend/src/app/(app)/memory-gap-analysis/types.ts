/**
 * Wire types for a recall-based data coverage run.
 *
 * Field names deliberately mirror the backend payload (snake_case) so the
 * fixture can be swapped for the endpoint response without a mapping layer.
 */

export type RunStatus = "pending" | "running" | "complete" | "failed";

/** Judge bands: 0–1 nothing usable retrieved, 2–3 partial, 4–5 answered. */
export type ScoreBand = "gap" | "partial" | "covered";

/**
 * Topics are user-owned and permanent; each run assigns questions to them.
 * Anything that matches no topic confidently lands in this bucket, which is
 * why the id is a constant rather than something a run mints.
 */
export const SINK_TOPIC_ID = "other";

export interface CoverageRun {
  run_id: string;
  dataset_id: string;
  status: RunStatus;
  created_at: string;
  recall_count: number;
  deduped_question_count: number;
  topic_count: number;
}

export interface CoverageTopic {
  topic_id: string;
  label: string;
  question_count: number;
  avg_score: number;
}

export interface CoverageQuestion {
  question_text: string;
  answer: string;
  judge_score: number;
  topic_id: string;
  first_asked_at: string;
  reference: string | null;
  /** Size of the dedup cluster — observed demand, not a prediction. */
  occurrence_count: number;
  /** Set only for sink questions that clustered together into a proposal. */
  suggested_topic_id: string | null;
}

/** A dense cluster found inside the sink, offered to the user as a new topic. */
export interface SuggestedTopic {
  suggestion_id: string;
  label: string;
  question_count: number;
}

export interface CoverageResult {
  run: CoverageRun;
  /** Mean of topic averages, 0–5. Excludes the sink. */
  overall_score: number;
  topics: CoverageTopic[];
  questions: CoverageQuestion[];
  suggested_topics: SuggestedTopic[];
}

/** A brain and its latest coverage run — the unit the whole page is scoped to. */
export interface Brain {
  id: string;
  name: string;
  /** Null when the brain exists but has never been scored. */
  result: CoverageResult | null;
}

export const MAX_SCORE = 5;
