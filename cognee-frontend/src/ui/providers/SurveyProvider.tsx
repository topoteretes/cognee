"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import checkSurveyEligibility from "@/modules/survey/checkSurveyEligibility";
import {
  ACCOUNT_AGE_TRIGGER_DAYS,
  NPS_SURVEY_KEY,
  NPS_SURVEY_VERSION,
  TRIGGER_ACCOUNT_AGE,
  TRIGGER_DATASOURCE_ADDED,
  TRIGGER_PRE_EXISTING_ACCOUNT,
} from "@/modules/survey/surveyConfig";
import { acquireCrossTabSurveyLock } from "@/modules/survey/surveyCrossTabLock";
import { useUser } from "@/modules/users/UserContext";
import { setSurveyTriggerListener } from "@/services/survey/surveyTriggerBridge";
import SurveyWidget from "@/ui/elements/SurveyWidget";
import type { UserMe } from "@/modules/users/UserContext";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

// Returns the trigger label this user qualifies under, or null if they may not
// be asked at all. Both call sites gate on it, so there is exactly one place
// that decides eligibility-by-tenure.
function qualifyingTrigger(userMe: UserMe | null): string | null {
  // null is "/me hasn't resolved", not an answer — qualifying on it would ask
  // a brand-new signup during the fetch window.
  if (!userMe) return null;
  // Accounts that predate the onboarding flow have no completion date and are
  // long-established by definition, so no waiting period applies to them.
  if (userMe.onboardingCompletedAt === null) return TRIGGER_PRE_EXISTING_ACCOUNT;
  // Everyone else signed up in the onboarding era. Onboarding completes within
  // minutes of signup, so it stands in for the signup date on the rare
  // backfilled row whose Principal.created_at was never set — without the
  // fallback those accounts could never qualify at all.
  const signupAt = userMe.accountCreatedAt ?? userMe.onboardingCompletedAt;
  const ageMs = Date.now() - new Date(signupAt).getTime();
  return ageMs >= ACCOUNT_AGE_TRIGGER_DAYS * ONE_DAY_MS ? TRIGGER_ACCOUNT_AGE : null;
}

// Single mount point (in the (app) layout). Two occasions can start the
// eligibility check — user load (checked on mount) and "datasource added"
// (fired via surveyTriggerBridge from anywhere in the app, e.g.
// createDataset.ts) — but both first have to clear qualifyingTrigger, and only
// one check may ever actually run: `hasChecked` blocks a second occasion from
// re-firing after the first has gone out, and it also defeats React
// StrictMode's dev-only double effect invocation. `acquireCrossTabSurveyLock`
// additionally stops two browser tabs opened together from both firing the
// (side-effecting) request at once.
export default function SurveyProvider({ children }: { children: ReactNode }): React.ReactElement {
  const pathname = usePathname();
  const pathnameRef = useRef(pathname);
  pathnameRef.current = pathname;

  const { userMe } = useUser();
  // Read through a ref in the bridge listener below: the listener is
  // registered once, so closing over userMe directly would pin it to the
  // value at registration time (null on first mount) forever.
  const userMeRef = useRef(userMe);
  userMeRef.current = userMe;
  const hasChecked = useRef(false);
  const [responseId, setResponseId] = useState<string | null>(null);

  const runEligibilityCheck = useCallback((trigger: string) => {
    if (hasChecked.current) return;
    if (!acquireCrossTabSurveyLock(NPS_SURVEY_KEY)) return;
    hasChecked.current = true;

    checkSurveyEligibility({
      surveyKey: NPS_SURVEY_KEY,
      surveyVersion: NPS_SURVEY_VERSION,
      trigger,
      page: pathnameRef.current,
    }).then((result) => {
      if (result?.eligible && result.responseId) {
        setResponseId(result.responseId);
      }
    });
  }, []);

  useEffect(() => {
    const trigger = qualifyingTrigger(userMe);
    if (trigger) {
      runEligibilityCheck(trigger);
    }
  }, [userMe, runEligibilityCheck]);

  useEffect(() => {
    setSurveyTriggerListener((event) => {
      if (event.reason !== "datasource_added") return;
      if (!qualifyingTrigger(userMeRef.current)) return;
      runEligibilityCheck(TRIGGER_DATASOURCE_ADDED);
    });
    return () => setSurveyTriggerListener(null);
  }, [runEligibilityCheck]);

  const handleDone = useCallback(() => setResponseId(null), []);

  return (
    <>
      {children}
      {responseId && <SurveyWidget responseId={responseId} onDone={handleDone} />}
    </>
  );
}
