export type ScoreBucket = "detractor" | "passive" | "promoter";

export interface SurveyEligibility {
  eligible: boolean;
  responseId: string | null;
}

export interface SubmitSurveyResponseResult {
  id: string;
  score: number;
  scoreBucket: ScoreBucket;
}
