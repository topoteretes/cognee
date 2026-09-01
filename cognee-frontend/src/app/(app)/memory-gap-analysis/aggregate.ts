import { SINK_TOPIC_ID, type CoverageQuestion } from "@/app/(app)/memory-gap-analysis/types";

/** Flag the taxonomy when the sink is this large a share of all questions. */
export const SINK_ALERT_SHARE = 0.2;
/** …or when a single cluster inside the sink gets this big, whatever the share. */
export const SINK_ALERT_CLUSTER = 8;

export interface TopicStats {
  count: number;
  avg: number;
}

export function statsFor(questions: CoverageQuestion[], topicId: string): TopicStats {
  const inTopic = questions.filter((q) => q.topic_id === topicId);
  if (inTopic.length === 0) return { count: 0, avg: 0 };
  return {
    count: inTopic.length,
    avg: inTopic.reduce((sum, q) => sum + q.judge_score, 0) / inTopic.length,
  };
}

/**
 * Mean of topic averages, unweighted, with the sink counted as a topic —
 * unassigned questions weigh on the score like any other bucket.
 */
export function overallScore(questions: CoverageQuestion[], topicIds: string[]): number {
  const averages = [...new Set([...topicIds, SINK_TOPIC_ID])]
    .map((id) => statsFor(questions, id))
    .filter((stats) => stats.count > 0)
    .map((stats) => stats.avg);
  if (averages.length === 0) return 0;
  return averages.reduce((sum, avg) => sum + avg, 0) / averages.length;
}

export function sinkShare(questions: CoverageQuestion[]): number {
  if (questions.length === 0) return 0;
  return questions.filter((q) => q.topic_id === SINK_TOPIC_ID).length / questions.length;
}
