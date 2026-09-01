import { formatScore } from "@/app/(app)/memory-gap-analysis/scoring";
import type { CoverageQuestion } from "@/app/(app)/memory-gap-analysis/types";

const HEADERS = ["Question", "Coverage", "Relevance", "Topic", "First asked", "Reference", "Replayed answer"] as const;

function escapeCell(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

export function questionsToCsv(questions: CoverageQuestion[], topicLabels: ReadonlyMap<string, string>): string {
  const rows = questions.map((q) =>
    [
      q.question_text,
      formatScore(q.judge_score),
      String(q.occurrence_count),
      topicLabels.get(q.topic_id) ?? q.topic_id,
      q.first_asked_at,
      q.reference ?? "",
      q.answer,
    ]
      .map(escapeCell)
      .join(","),
  );
  return [HEADERS.join(","), ...rows].join("\n");
}

export function downloadCsv(filename: string, csv: string): void {
  // Prepend a BOM so Excel reads the UTF-8 question text correctly.
  const blob = new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
