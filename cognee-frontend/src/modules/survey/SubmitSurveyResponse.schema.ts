import { boolean, integer, maxValue, minValue, nullable, number, object, pipe, string, type InferOutput } from "valibot";
import { SURVEY_SCORE_MAX, SURVEY_SCORE_MIN } from "./surveyConfig";

export const SubmitSurveyResponseSchema = object({
  responseId: string(),
  score: pipe(number(), integer(), minValue(SURVEY_SCORE_MIN), maxValue(SURVEY_SCORE_MAX)),
  followupQuestionId: nullable(string()),
  followupAnswer: nullable(string()),
  consentToQuote: boolean(),
});

export type SubmitSurveyResponseInput = InferOutput<typeof SubmitSurveyResponseSchema>;
