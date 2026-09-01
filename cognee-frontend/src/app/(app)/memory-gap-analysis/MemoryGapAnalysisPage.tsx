"use client";

import React, { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { TrackPageView } from "@/modules/analytics";
import { AsciiFrame } from "@/app/(app)/dashboard/partials/redesign/AsciiFrame";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { SINK_ALERT_SHARE, overallScore, sinkShare, statsFor } from "@/app/(app)/memory-gap-analysis/aggregate";
import { downloadCsv, questionsToCsv } from "@/app/(app)/memory-gap-analysis/exportCsv";
import { useCoverageRuns } from "@/app/(app)/memory-gap-analysis/hooks/useCoverageRuns";
import { BrainSwitcher, type BrainOption } from "@/app/(app)/memory-gap-analysis/partials/BrainSwitcher";
import { DeleteTopicModal } from "@/app/(app)/memory-gap-analysis/partials/DeleteTopicModal";
import { QuestionGrid, type QuestionView } from "@/app/(app)/memory-gap-analysis/partials/QuestionGrid";
import { ExportButton, QuestionToolbar, type ScopeMetrics } from "@/app/(app)/memory-gap-analysis/partials/QuestionToolbar";
import { ViewToggle } from "@/app/(app)/memory-gap-analysis/partials/ViewToggle";
import { AddMemoryButton, RunButton, ScoreLine } from "@/app/(app)/memory-gap-analysis/partials/RunSummary";
import { TopicChips, type TopicChipRow } from "@/app/(app)/memory-gap-analysis/partials/TopicChips";
import { formatDate } from "@/app/(app)/memory-gap-analysis/scoring";
import { matchesQuery } from "@/app/(app)/memory-gap-analysis/search";
import { sortQuestions, sortStateFor, type SortMode } from "@/app/(app)/memory-gap-analysis/sorting";
import { SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";
import { SINK_TOPIC_ID, type Brain, type CoverageResult } from "@/app/(app)/memory-gap-analysis/types";

const PROSE_MAX_WIDTH = 720;

/** Local topic edits, keyed on brain — see handleConfirmDelete. */
type ResultsByBrain = Record<string, CoverageResult>;

/** An unscored brain still gets a chip, so the scope list is always the real one. */
function brainOption(brain: Brain, result: CoverageResult | null): BrainOption {
  if (result === null) {
    return { id: brain.id, name: brain.name, score: null, sinkShare: 0, needsAttention: false };
  }
  const share = sinkShare(result.questions);
  return {
    id: brain.id,
    name: brain.name,
    score: overallScore(result.questions, result.topics.map((t) => t.topic_id)),
    sinkShare: share,
    needsAttention: share >= SINK_ALERT_SHARE,
  };
}

function LegendItem({ color, label }: { color: string; label: string }): React.ReactElement {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: SPACE.xs, whiteSpace: "nowrap" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0 }} />
      <span style={{ ...FONT, fontSize: SIZE.meta, color: T.faint, fontVariantNumeric: "tabular-nums" }}>{label}</span>
    </span>
  );
}

function FootnoteStat({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <span style={{ display: "inline-flex", alignItems: "baseline", gap: SPACE.xs, whiteSpace: "nowrap" }}>
      <span style={{ ...FONT, fontSize: SIZE.meta, color: T.ghost }}>{label}</span>
      <span style={{ ...FONT, fontSize: SIZE.meta, color: T.faint, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </span>
  );
}

export default function MemoryGapAnalysisPage(): React.ReactElement {
  const router = useRouter();
  const { brains, loading, unavailable, isRunning, startRun } = useCoverageRuns();
  const [edits, setEdits] = useState<ResultsByBrain>({});
  const [requestedBrainId, setRequestedBrainId] = useState<string | null>(null);
  const [selectedTopicId, setSelectedTopicId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("score");
  const [view, setView] = useState<QuestionView>("list");
  const [topicPendingDelete, setTopicPendingDelete] = useState<string | null>(null);

  // Falls back to the first brain so the page always has a scope, including on
  // the first render and after the selected brain disappears.
  const selectedBrain = brains.find((b) => b.id === requestedBrainId) ?? brains[0] ?? null;
  const selectedBrainId = selectedBrain?.id ?? "";
  const result = selectedBrain === null ? null : edits[selectedBrain.id] ?? selectedBrain.result;

  const brainOptions = useMemo(
    () => brains.map((brain) => brainOption(brain, edits[brain.id] ?? brain.result)),
    [brains, edits],
  );

  // Empty arrays until a run exists, so every derivation below stays valid for
  // a brain that has never been scored.
  const topics = useMemo(() => result?.topics ?? [], [result]);
  const questions = useMemo(() => result?.questions ?? [], [result]);
  const run = result?.run ?? null;

  const topicLabels = useMemo(() => {
    const labels = new Map(topics.map((topic) => [topic.topic_id, topic.label]));
    labels.set(SINK_TOPIC_ID, "Other");
    return labels;
  }, [topics]);

  const chipRows: TopicChipRow[] = useMemo(
    () => topics.map((topic) => ({ topicId: topic.topic_id, label: topic.label, questionCount: statsFor(questions, topic.topic_id).count })),
    [questions, topics],
  );

  const sink = useMemo(() => statsFor(questions, SINK_TOPIC_ID), [questions]);
  const score = useMemo(() => overallScore(questions, topics.map((t) => t.topic_id)), [questions, topics]);


  const scope = useMemo(() => {
    return questions.filter((q) => {
      if (selectedTopicId !== null && q.topic_id !== selectedTopicId) return false;
      // Search the whole card, not just the question: answer, topic and
      // reference all carry words people remember.
      const haystack = [q.question_text, q.answer, topicLabels.get(q.topic_id) ?? "", q.reference ?? ""].join(" ");
      return matchesQuery(haystack, query);
    });
  }, [query, questions, selectedTopicId, topicLabels]);

  const metrics: ScopeMetrics = useMemo(() => {
    if (scope.length === 0) return { coverage: 0, asked: 0 };
    return {
      coverage: scope.reduce((sum, q) => sum + q.judge_score, 0) / scope.length,
      asked: scope.reduce((sum, q) => sum + q.occurrence_count, 0),
    };
  }, [scope]);

  const visible = useMemo(() => sortQuestions(scope, sortStateFor(sortMode), topicLabels), [scope, sortMode, topicLabels]);


  /** Topic ids are per-brain, so every filter resets when the scope changes. */
  const handleBrainSelect = useCallback((brainId: string) => {
    setRequestedBrainId(brainId);
    setSelectedTopicId(null);
    setQuery("");
  }, []);

  /** Deleting a topic never deletes questions — they fall back to the sink. */
  const handleConfirmDelete = useCallback(() => {
    const topicId = topicPendingDelete;
    if (topicId === null || result === null) return;
    // MISSING ENDPOINT: topic edits live only in this tab until the coverage
    // API can persist them (accept suggestion, create/delete topic).
    setEdits((current) => ({
      ...current,
      [selectedBrainId]: {
        ...result,
        topics: result.topics.filter((topic) => topic.topic_id !== topicId),
        questions: result.questions.map((q) => (q.topic_id === topicId ? { ...q, topic_id: SINK_TOPIC_ID } : q)),
      },
    }));
    setSelectedTopicId((current) => (current === topicId ? null : current));
    setTopicPendingDelete(null);
  }, [result, selectedBrainId, topicPendingDelete]);

  /**
   * Whole-run replay: every question is re-judged, so the button never offers a
   * partial. The POST returns as soon as the run is queued — it takes minutes —
   * and `isRunning` then comes from the run's own status while the hook polls.
   */
  const handleRun = useCallback((): void => {
    if (selectedBrainId === "") return;
    void startRun(selectedBrainId);
  }, [selectedBrainId, startRun]);

  const handleExport = useCallback(() => {
    const scopeName = selectedTopicId ?? "all-topics";
    downloadCsv(`memory-gap-${selectedBrainId}-${scopeName}.csv`, questionsToCsv(visible, topicLabels));
  }, [selectedBrainId, selectedTopicId, topicLabels, visible]);

  return (
    <div style={{ minHeight: "100%", padding: "24px 32px 32px", display: "flex", flexDirection: "column", gap: SPACE.xl }}>
      <TrackPageView page="MemoryGapAnalysis" />

      <header style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: SPACE.lg, marginBottom: SPACE.lg }}>
        <div style={{ display: "flex", flexDirection: "column", gap: SPACE.sm }}>
          <h1 style={{ ...FONT, margin: 0, fontSize: SIZE.title, fontWeight: 300, color: T.text, lineHeight: 1.2 }}>
            Memory Coverage
          </h1>
          <p style={{ ...FONT, margin: 0, fontSize: SIZE.body, color: T.muted, maxWidth: PROSE_MAX_WIDTH, lineHeight: 1.5 }}>
            Every question you asked, scored by how well memory answered it. Low scores are your gaps — add the missing knowledge to fix them.
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: SPACE.sm, flexShrink: 0 }}>
          <AddMemoryButton onClick={() => router.push("/datasets")} />
          <RunButton isRunning={isRunning} onRun={handleRun} />
        </div>
      </header>

      {topicPendingDelete !== null && (
        <DeleteTopicModal
          topicLabel={topicLabels.get(topicPendingDelete) ?? topicPendingDelete}
          questionCount={statsFor(questions, topicPendingDelete).count}
          onConfirm={handleConfirmDelete}
          onCancel={() => setTopicPendingDelete(null)}
        />
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: SPACE.md, opacity: isRunning ? 0.55 : 1, transition: "opacity 160ms" }}>
        {/* Score band: the meter spans the full row, the number rides the fill edge. */}
        {result !== null && (
          <div style={{ marginBottom: SPACE.lg + 8 }}>
            <ScoreLine overallScore={score} />
          </div>
        )}

        {brainOptions.length > 0 && (
          <BrainSwitcher brains={brainOptions} selectedBrainId={selectedBrainId} onSelect={handleBrainSelect} />
        )}

        {result === null ? (
          <CoverageEmptyState loading={loading} unavailable={unavailable} brainName={selectedBrain?.name ?? null} />
        ) : (
        <AsciiFrame label={selectedTopicId === null ? "Questions" : topicLabels.get(selectedTopicId) ?? "Questions"}>
          <div style={{ display: "flex", flexDirection: "column", gap: SPACE.lg }}>
            {/* Filters row: topic chips left, view toggle and export right. */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: SPACE.lg }}>
              <TopicChips
                rows={chipRows}
                sinkCount={sink.count}
                totalQuestions={questions.length}
                selectedTopicId={selectedTopicId}
                onSelect={setSelectedTopicId}
                onRequestDelete={setTopicPendingDelete}
              />
              <span style={{ display: "inline-flex", alignItems: "center", gap: SPACE.sm, flexShrink: 0 }}>
                <ViewToggle view={view} onChange={setView} />
                <ExportButton onExport={handleExport} />
              </span>
            </div>
            <QuestionToolbar
              query={query}
              onQueryChange={setQuery}
              metrics={metrics}
              topicScoped={selectedTopicId !== null}
              sortMode={sortMode}
              onSortModeChange={setSortMode}
            />
            <QuestionGrid questions={visible} topicLabels={topicLabels} view={view} />
          </div>
        </AsciiFrame>
        )}

        {/* Footnote: which run this page shows, plus the score-colour legend. */}
        {run !== null && (
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: `${SPACE.xs}px ${SPACE.xl}px` }}>
            <FootnoteStat label="Run" value={run.run_id} />
            <FootnoteStat label="Status" value={run.status} />
            <FootnoteStat label="Created" value={formatDate(run.created_at)} />
            <span style={{ flex: 1 }} />
            <span style={{ display: "inline-flex", alignItems: "center", gap: SPACE.lg }}>
              <LegendItem color={T.red} label="Gap 0–1.9" />
              <LegendItem color={T.amber} label="Partial 2.0–3.9" />
              <LegendItem color={T.green} label="Covered 4.0–5.0" />
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * MISSING ENDPOINT: there is no coverage API yet (see useCoverageRuns), so a
 * workspace with real brains still lands here. Says so plainly rather than
 * showing example questions that would read as this tenant's own.
 */
function CoverageEmptyState({
  loading,
  unavailable,
  brainName,
}: {
  loading: boolean;
  unavailable: boolean;
  brainName: string | null;
}): React.ReactElement {
  const message = loading
    ? "Loading…"
    : unavailable
      ? `Coverage scoring isn't available yet${brainName === null ? "" : ` for ${brainName}`}. Once a run has been scored, its questions and their coverage show up here.`
      : "Add a brain and ask it something — this page scores every question by how well memory answered it.";

  return (
    <AsciiFrame label={null}>
      <span style={{ ...FONT, fontSize: SIZE.body, color: T.muted, maxWidth: PROSE_MAX_WIDTH, display: "block", lineHeight: 1.5 }}>
        {message}
      </span>
    </AsciiFrame>
  );
}
