import { SINK_TOPIC_ID, type CoverageQuestion, type CoverageResult, type CoverageRun, type CoverageTopic } from "@/app/(app)/memory-gap-analysis/types";

/** Flag the taxonomy when the sink is this large a share of all questions. */
export const SINK_ALERT_SHARE = 0.2;
/** …or when a single cluster inside the sink gets this big, whatever the share. */
export const SINK_ALERT_CLUSTER = 8;

export interface TopicStats {
  /** Every question in the topic, measured or not. */
  count: number;
  /** How many of them reached a verdict. */
  measured: number;
  /** Mean over the measured ones only. Null when none of them were. */
  avg: number | null;
}

/**
 * A topic's size and its score. The two counts are separate on purpose: an
 * unmeasured question is still a question the dataset was asked, but averaging
 * its null as a zero would invent the verdict the judge never reached — the
 * same rule the backend applies to its own topic averages.
 */
export function statsFor(questions: CoverageQuestion[], topicId: string): TopicStats {
  const inTopic = questions.filter((q) => q.topic_id === topicId);
  const scores = inTopic.map((q) => q.judge_score).filter((score): score is number => score !== null);
  return {
    count: inTopic.length,
    measured: scores.length,
    avg: scores.length === 0 ? null : scores.reduce((sum, score) => sum + score, 0) / scores.length,
  };
}

/**
 * Mean of topic averages, unweighted, with the sink counted as a topic —
 * unassigned questions weigh on the score like any other bucket. Null with no
 * question in any bucket: there is no average to take, and 0 would read as a
 * verdict.
 */
export function overallScore(questions: CoverageQuestion[], topicIds: string[]): number | null {
  const averages = [...new Set([...topicIds, SINK_TOPIC_ID])]
    .map((id) => statsFor(questions, id).avg)
    .filter((avg): avg is number => avg !== null);
  if (averages.length === 0) return null;
  return averages.reduce((sum, avg) => sum + avg, 0) / averages.length;
}

/**
 * The score to render for a run, or null when the run measured nothing.
 *
 * The run's own `overall_score` decides whether there is a score at all, not
 * the question rows: a run from before CLO-732 still persists its failed
 * replays with a 0.0 judge score (those rows were deliberately not
 * backfilled), so averaging them here would invent the "answers nothing"
 * verdict the backend refused to give. A run since then stores no score for
 * them at all — `coverage_questions` carries a CHECK constraint making
 * `judge_score IS NULL` exactly equivalent to having an error — which is why
 * `runSample` prefers the run's own counts over what these rows show. The
 * weighting stays local — this page counts the sink as a topic and the backend
 * does not — but the decision to score does not.
 */
export function displayScore(result: CoverageResult): number | null {
  if (result.overall_score === null) return null;
  return overallScore(result.questions, result.topics.map((topic) => topic.topic_id));
}

/**
 * A finished run's own score, or null when it has none — a run still going, one
 * that reached no verdict, or a brain nobody has scored.
 *
 * The backend's number verbatim, unlike `displayScore`: the dashboard reports
 * the workspace roll-up, which excludes the sink, rather than this page's own
 * weighting. Either way a null must stay null — averaged in as a zero, one
 * unmeasurable dataset drags the whole workspace's recall down.
 */
export function completedRunScore(result: CoverageResult | null): number | null {
  if (result === null || result.run.status !== "complete") return null;
  return result.overall_score;
}

export function sinkShare(questions: CoverageQuestion[]): number {
  if (questions.length === 0) return 0;
  return questions.filter((q) => q.topic_id === SINK_TOPIC_ID).length / questions.length;
}

/**
 * Share of a sample that reached no verdict, above which the score stops being
 * a fair summary of the corpus and has to read as degraded rather than merely
 * partial: a fifth of the questions missing is enough to change what the
 * number means.
 */
export const DEGRADED_SHARE = 0.2;

/**
 * A count that arrived from the pod, or undefined when it did not.
 *
 * Undefined and zero are kept apart on purpose. The endpoint response is cast,
 * never validated (see `useCoverageRuns`), so a pod that predates these fields
 * sends none of them — and reading that as "nothing was measured" would paint
 * every healthy run in the fleet as degraded, on the exact signal that exists
 * to catch a fleet-wide failure. A count nobody sent is unknown, not zero.
 */
function counted(value: number | null | undefined): number | undefined {
  if (!Number.isFinite(value)) return undefined;
  return Math.max(0, Math.round(value as number));
}

export interface Sample {
  /**
   * Whether there is a sample at all. False for a pod that sent no counts, and
   * every consumer has to check it: a `measured` of 0 read without this is the
   * claim that nothing was scored, which is how the topic chip came to read
   * "0/12" over twelve scored questions.
   */
  known: boolean;
  /** How many questions the score actually covers. */
  measured: number;
  /** How many questions the run knows about. Zero when the sample is unknown. */
  total: number;
  /** Questions with no verdict — replay or judge failures, plus never-replayed. */
  unmeasured: number;
  /** `unmeasured / total`, 0 when there is nothing to measure. */
  unmeasuredShare: number;
  /** Enough of the sample is missing that the score is not representative. */
  degraded: boolean;
}

/** Nothing to say about the sample: never run, empty, or a pod that sent no counts. */
const UNKNOWN_SAMPLE: Sample = { known: false, measured: 0, total: 0, unmeasured: 0, unmeasuredShare: 0, degraded: false };

function sample(measured: number, total: number): Sample {
  if (total === 0) return UNKNOWN_SAMPLE;
  const unmeasured = Math.max(0, total - measured);
  const unmeasuredShare = unmeasured / total;
  return { known: true, measured, total, unmeasured, unmeasuredShare, degraded: unmeasuredShare >= DEGRADED_SHARE };
}

/**
 * What the run's score is actually measured over.
 *
 * `n_replayed - n_errored`, not the corpus size: a fleet-wide throttle that
 * fails 45 of 50 replays used to drag the score toward zero and so looked
 * broken, and once those questions were excluded from the mean instead it
 * rendered as a clean, confident 4.6 — indistinguishable from a run that
 * measured all 50. These counts are what tell the two apart.
 *
 * The denominator is the widest of the corpus, what was replayed and what is
 * on screen, so a question the run never reached counts as missing rather than
 * silently dropping out of both halves of the fraction.
 *
 * The numerator is the lower of the pod's count and the questions this page
 * can see a verdict on, because the score beside it is recomputed locally from
 * those questions (`displayScore`): a note claiming to cover more than the
 * score was averaged over would qualify a number that does not exist.
 *
 * With no questions on screen there is no sample either, whatever the counts
 * say. The grid is already saying the run replayed nothing and the meter has
 * no number to qualify, so a "covers 5 of 50" beside them would be the third
 * of three mutually exclusive lines.
 */
export function runSample(run: CoverageRun, questions: CoverageQuestion[]): Sample {
  const replayed = counted(run.n_replayed);
  const errored = counted(run.n_errored);
  if (replayed === undefined || errored === undefined || questions.length === 0) return UNKNOWN_SAMPLE;
  const onWire = Math.max(0, replayed - errored);
  const onScreen = questions.filter((q) => q.judge_score !== null).length;
  const total = Math.max(counted(run.deduped_question_count) ?? 0, replayed, questions.length);
  return sample(Math.min(onWire, onScreen), total);
}

/**
 * What one topic's average is measured over. 4.6 over 5 measured of 50 is a
 * different claim from 4.6 over 50.
 *
 * `count` is this page's own question count for the topic rather than the
 * topic's `question_count`, because a local topic delete moves questions into
 * the sink and the wire counts then describe a taxonomy the reader is no
 * longer looking at. The wire `measured_count` is clamped to it for the same
 * reason — it can only ever be part of a count, never more than it.
 */
export function topicSample(topic: CoverageTopic, count: number): Sample {
  const measured = counted(topic.measured_count);
  if (measured === undefined) return UNKNOWN_SAMPLE;
  return sample(Math.min(measured, count), count);
}
