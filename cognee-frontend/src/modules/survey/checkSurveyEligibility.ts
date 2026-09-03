"use server";

import type { SurveyEligibility } from "./types";

interface CheckSurveyEligibilityInput {
  surveyKey: string;
  surveyVersion: number;
  trigger: string;
  page: string;
}

// Open-source stub — survey eligibility lives in the cloud management
// backend. Returning null means "not eligible", so the shared survey UI
// (SurveyProvider/SurveyWidget) compiles but never shows in local mode.
export default async function checkSurveyEligibility(
  _input: CheckSurveyEligibilityInput,
): Promise<SurveyEligibility | null> {
  return null;
}
