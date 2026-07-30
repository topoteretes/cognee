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
} from "@/modules/survey/surveyConfig";
import { acquireCrossTabSurveyLock } from "@/modules/survey/surveyCrossTabLock";
import { useUser } from "@/modules/users/UserContext";
import { setSurveyTriggerListener } from "@/services/survey/surveyTriggerBridge";
import SurveyWidget from "@/ui/elements/SurveyWidget";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

function isAccountOldEnough(accountCreatedAt: string | null | undefined): boolean {
  if (!accountCreatedAt) return false;
  const ageMs = Date.now() - new Date(accountCreatedAt).getTime();
  return ageMs >= ACCOUNT_AGE_TRIGGER_DAYS * ONE_DAY_MS;
}

// Single mount point (in the (app) layout). Two independent conditions can
// trigger the eligibility check — account age (checked on mount/user-load)
// and "datasource added" (fired via surveyTriggerBridge from anywhere in the
// app, e.g. createDataset.ts) — but only one check may ever actually run:
// `hasChecked` blocks a second trigger from re-firing after the first has
// gone out, and it also defeats React StrictMode's dev-only double effect
// invocation. `acquireCrossTabSurveyLock` additionally stops two browser tabs
// opened together from both firing the (side-effecting) request at once.
export default function SurveyProvider({ children }: { children: ReactNode }): React.ReactElement {
  const pathname = usePathname();
  const pathnameRef = useRef(pathname);
  pathnameRef.current = pathname;

  const { userMe } = useUser();
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
    if (isAccountOldEnough(userMe?.accountCreatedAt)) {
      runEligibilityCheck(TRIGGER_ACCOUNT_AGE);
    }
  }, [userMe, runEligibilityCheck]);

  useEffect(() => {
    setSurveyTriggerListener((event) => {
      if (event.reason === "datasource_added") {
        runEligibilityCheck(TRIGGER_DATASOURCE_ADDED);
      }
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
