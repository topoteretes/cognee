"use server";

import type { SubmitSurveyResponseInput } from "./SubmitSurveyResponse.schema";
import type { SubmitSurveyResponseResult } from "./types";

// Open-source stub — survey responses are stored by the cloud management
// backend. Unreachable in local mode because checkSurveyEligibility's stub
// never reports an eligible survey, but it must compile for the shared UI.
export default async function submitSurveyResponse(
  _input: SubmitSurveyResponseInput,
): Promise<SubmitSurveyResponseResult> {
  throw new Error("Survey responses are not available in local mode.");
}
