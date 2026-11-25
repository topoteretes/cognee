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
  /**
   * How many distinct questions this run actually replayed.
   *
   * Optional because the endpoint response is cast and never validated (see
   * `useCoverageRuns`), so this interface is the whole contract: a pod that
   * predates these fields sends none of them, and declaring them required
   * would make the guards in `runSample` branches the compiler proves
   * unreachable while they are the only thing standing between a staggered
   * rollout and every healthy run in the fleet reading as degraded.
   */
  n_replayed?: number | null;
  /**
   * How many of those reached no verdict — the replay or the judge call
   * failed. The score covers `n_replayed - n_errored`, so without this a
   * reader cannot tell 68% over 180 questions from 68% over nine. While a
   * failed replay was averaged in as a 0.0 the score itself gave a degraded
   * run away; now that such a question is excluded instead, this is the only
   * thing on the wire that says so.
   */
  n_errored?: number | null;
}

export interface CoverageTopic {
  topic_id: string;
  label: string;
  question_count: number;
  /**
   * How many of `question_count` the average actually covers. 4.6 over 5
   * measured of 50 is a different claim from 4.6 over 50, and without this the
   * two are indistinguishable without recounting nulls in `questions[]`.
   */
  measured_count?: number | null;
  /** Mean of this topic's measured questions. Null when none were measured. */
  avg_score: number | null;
}

export interface CoverageQuestion {
  question_text: string;
  answer: string;
  /**
   * Judge verdict 0–5, or null when this question reached none: the replay or
   * the judge call failed. Null is not a zero, and nothing may average it as
   * one — the backend excludes it from its own means for the same reason.
   */
  judge_score: number | null;
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
  /**
   * Mean of topic averages, 0–5, sink excluded. Null when the run reached no
   * verdict at all: one still pending, one whose every replay or judge call
   * failed, or a dataset that was never asked anything. "Not measured" is a
   * different claim from "answers nothing", and only the second is a score.
   */
  overall_score: number | null;
  topics: CoverageTopic[];
  questions: CoverageQuestion[];
  suggested_topics: SuggestedTopic[];
}

/**
 * How one brain's own latest-run fetch is doing. Per brain, never per page: a
 * dataset the pod cannot answer for says nothing about the ones it can.
 */
export type BrainLoadState = "loading" | "error" | "loaded";

/** A brain and its latest coverage run — the unit the whole page is scoped to. */
export interface Brain {
  id: string;
  name: string;
  /** Null when the brain has never been scored, and while loading or failed. */
  result: CoverageResult | null;
  state: BrainLoadState;
  /** True while this brain's own latest run is pending or running. */
  isRunning: boolean;
}

export const MAX_SCORE = 5;
