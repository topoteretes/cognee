import type { ScoreBucket } from "./types";

// First survey shipped with this feature — see CLO-363.
export const NPS_SURVEY_KEY = "nps_quarterly";
export const NPS_SURVEY_VERSION = 1;

// One qualification decides WHETHER a user may be asked at all (see
// SurveyProvider's qualifyingTrigger): either the account predates the
// onboarding flow, or it has aged past the trial window. Adding a data source
// is only an OCCASION to run the already-qualified check, never a way around
// it — createDataset also fires for datasets the app seeds itself (see
// waitForPodReady and useDatasetUpload), so treating it as its own
// qualification handed brand-new signups an NPS prompt on day one.
export const TRIGGER_ACCOUNT_AGE = "account_age_15_days";
export const TRIGGER_PRE_EXISTING_ACCOUNT = "pre_existing_account";
export const TRIGGER_DATASOURCE_ADDED = "datasource_added";
export const ACCOUNT_AGE_TRIGGER_DAYS = 15;

export const SURVEY_SCORE_MIN = 0;
export const SURVEY_SCORE_MAX = 10;

const DETRACTOR_MAX_SCORE = 6;
const PASSIVE_MAX_SCORE = 8;

export function scoreBucketFor(score: number): ScoreBucket {
  if (score <= DETRACTOR_MAX_SCORE) return "detractor";
  if (score <= PASSIVE_MAX_SCORE) return "passive";
  return "promoter";
}

export interface FollowupQuestion {
  id: string;
  question: string;
  showQuoteConsent: boolean;
}

export const FOLLOWUP_QUESTIONS: Record<ScoreBucket, FollowupQuestion> = {
  detractor: {
    id: "why_low_score",
    question: "What's the main reason for that score?",
    showQuoteConsent: false,
  },
  passive: {
    id: "what_missing",
    question: "What would make this a 10 for you?",
    showQuoteConsent: false,
  },
  promoter: {
    id: "what_valued",
    question: "What do you value most about Cognee?",
    showQuoteConsent: true,
  },
};
