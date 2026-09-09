"use client";

import classNames from "classnames";
import { SURVEY_SCORE_MAX, scoreBucketFor } from "@/modules/survey/surveyConfig";
import type { ScoreBucket } from "@/modules/survey/types";

const SCORES = Array.from({ length: SURVEY_SCORE_MAX + 1 }, (_, score) => score);

const BUCKET_STYLES: Record<ScoreBucket, string> = {
  detractor: "border-transparent bg-red-400/90 text-black",
  passive: "border-transparent bg-amber-400/90 text-black",
  promoter: "border-transparent bg-cognee-lavender text-cognee-lavender-text",
};

interface SurveyScoreStepProps {
  question: string;
  selectedScore: number | null;
  onSelect: (score: number) => void;
}

export default function SurveyScoreStep({
  question,
  selectedScore,
  onSelect,
}: SurveyScoreStepProps): React.ReactElement {
  return (
    <div>
      <p className="m-0 mb-3.5 text-[13.5px] font-semibold text-cognee-dark">{question}</p>

      <div className="mb-2 grid grid-cols-11 gap-1">
        {SCORES.map((score) => {
          const isSelected = score === selectedScore;
          return (
            <button
              key={score}
              type="button"
              onClick={() => onSelect(score)}
              className={classNames(
                "aspect-square cursor-pointer rounded-md border text-xs font-semibold tabular-nums transition-colors",
                isSelected ? BUCKET_STYLES[scoreBucketFor(score)] : "border-cognee-border bg-cognee-bg text-cognee-body hover:border-cognee-lavender",
              )}
            >
              {score}
            </button>
          );
        })}
      </div>

      <div className="mb-4 flex justify-between text-[10.5px] text-cognee-placeholder">
        <span>0 · Not likely</span>
        <span>10 · Extremely likely</span>
      </div>
    </div>
  );
}
