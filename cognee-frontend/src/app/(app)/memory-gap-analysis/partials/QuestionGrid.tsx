"use client";

import React from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { scoreColor } from "@/app/(app)/memory-gap-analysis/scoring";
import { SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";
import type { CoverageQuestion } from "@/app/(app)/memory-gap-analysis/types";

export type QuestionView = "grid" | "list";

interface QuestionGridProps {
  questions: CoverageQuestion[];
  topicLabels: ReadonlyMap<string, string>;
  view: QuestionView;
}

const CHIP = {
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  borderRadius: 100,
  padding: "2px 8px",
  whiteSpace: "nowrap",
  flexShrink: 0,
  fontVariantNumeric: "tabular-nums",
} as const;

function ScoreChip({ score }: { score: number }): React.ReactElement {
  const color = scoreColor(score);
  return (
    <span style={{ ...FONT, ...CHIP, color, background: `color-mix(in srgb, ${color} 12%, transparent)` }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: color }} />
      {score.toFixed(1)}
    </span>
  );
}

function RecallsChip({ count }: { count: number }): React.ReactElement {
  return (
    <span
      title={`Asked ${count}× in this window — the size of its dedup cluster, not a forecast`}
      style={{ ...FONT, ...CHIP, color: "var(--color-cognee-lavender)", background: "var(--color-cognee-lavender-tint-10)" }}
    >
      {count}× recalls
    </span>
  );
}

function TopicChip({ label }: { label: string }): React.ReactElement {
  return (
    <span style={{ ...FONT, fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)", background: "rgba(255,255,255,0.08)", borderRadius: 100, padding: "2px 10px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0, flexShrink: 1 }}>
      {label}
    </span>
  );
}

// Fixed list-view tracks so score, recalls and topic chips line up in
// straight columns across every row.
const LIST_GRID = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr) 120px 110px 170px",
  columnGap: SPACE.md,
  alignItems: "center",
} as const;

function QuestionCard({ question, topicLabel, list }: { question: CoverageQuestion; topicLabel: string; list: boolean }): React.ReactElement {
  const card: React.CSSProperties = list
    ? { ...LIST_GRID, background: "#000000", border: `1px solid ${T.frame}`, borderRadius: 0, padding: `${SPACE.sm}px ${SPACE.md}px` }
    : { display: "flex", flexDirection: "column", gap: SPACE.sm, background: "#000000", border: `1px solid ${T.frame}`, borderRadius: 0, padding: SPACE.md };
  const text = (
    <span
      title={question.question_text}
      style={{
        ...FONT, fontSize: SIZE.body, color: T.text, lineHeight: 1.45,
        display: "-webkit-box", WebkitLineClamp: list ? 1 : 3, WebkitBoxOrient: "vertical", overflow: "hidden",
        ...(list ? { minWidth: 0 } : {}),
      }}
    >
      {question.question_text}
    </span>
  );
  if (list) {
    return (
      <div style={card}>
        {text}
        <div><ScoreChip score={question.judge_score} /></div>
        <div><RecallsChip count={question.occurrence_count} /></div>
        <div style={{ display: "flex", minWidth: 0 }}><TopicChip label={topicLabel} /></div>
      </div>
    );
  }
  return (
    <div style={card}>
      {text}
      {/* Topic gets its own line so long labels never truncate in 4-up cards. */}
      <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: SPACE.xs }}>
        <div style={{ display: "flex", alignItems: "center", gap: SPACE.sm }}>
          <ScoreChip score={question.judge_score} />
          <RecallsChip count={question.occurrence_count} />
        </div>
        <TopicChip label={topicLabel} />
      </div>
    </div>
  );
}

/** The questions, ranked by the page's sort — packed cards in a grid or stacked rows. */
export function QuestionGrid({ questions, topicLabels, view }: QuestionGridProps): React.ReactElement {
  if (questions.length === 0) {
    return <span style={{ ...FONT, fontSize: SIZE.body, color: T.faint, padding: `${SPACE.md}px 0` }}>No questions match.</span>;
  }
  const layout: React.CSSProperties = view === "grid"
    ? { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: SPACE.sm }
    : { display: "flex", flexDirection: "column", gap: SPACE.xs };
  return (
    <div style={layout}>
      {questions.map((question) => (
        <QuestionCard
          key={`${question.question_text}-${question.first_asked_at}`}
          question={question}
          topicLabel={topicLabels.get(question.topic_id) ?? question.topic_id}
          list={view === "list"}
        />
      ))}
    </div>
  );
}
