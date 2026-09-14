"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import submitSurveyResponse from "@/modules/survey/submitSurveyResponse";
import { FOLLOWUP_QUESTIONS, NPS_SURVEY_KEY, scoreBucketFor } from "@/modules/survey/surveyConfig";
import type { ScoreBucket } from "@/modules/survey/types";
import { trackEvent } from "@/modules/analytics";

const THANKS_AUTO_CLOSE_MS = 2800;

export type SurveyWidgetStep = "score" | "followup" | "thanks";

export interface SurveyWidgetState {
  step: SurveyWidgetStep;
  score: number | null;
  bucket: ScoreBucket | null;
  answer: string;
  consentToQuote: boolean;
  sending: boolean;
  error: string | null;
  portalTarget: Element | null;
  selectScore: (score: number) => void;
  goBackToScore: () => void;
  setAnswer: (value: string) => void;
  setConsentToQuote: (value: boolean) => void;
  skip: () => void;
  send: () => void;
  close: () => void;
}

export function useSurveyWidgetState(responseId: string, onDone: () => void): SurveyWidgetState {
  const [step, setStep] = useState<SurveyWidgetStep>("score");
  const [score, setScore] = useState<number | null>(null);
  const [answer, setAnswer] = useState("");
  const [consentToQuote, setConsentToQuote] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [portalTarget, setPortalTarget] = useState<Element | null>(null);
  const hasTrackedShown = useRef(false);

  useEffect(() => {
    setPortalTarget(document.body);

    // Guards against React StrictMode's dev-only double effect invocation —
    // without it this fires twice per real mount and inflates the dev "shown"
    // count (harmless in production, where effects only run once).
    if (!hasTrackedShown.current) {
      hasTrackedShown.current = true;
      trackEvent({ pageName: "Survey Widget", eventName: "survey_widget_shown", additionalProperties: { survey_key: NPS_SURVEY_KEY } });
    }
  }, []);

  const selectScore = useCallback((selected: number) => {
    setScore(selected);
    // A bucket change must never carry over an answer/consent given under a
    // different question — otherwise Back → pick a different score can submit
    // consentToQuote or free text that belongs to a bucket the user never saw.
    setAnswer("");
    setConsentToQuote(false);
    setError(null);
    trackEvent({ pageName: "Survey Widget", eventName: "survey_score_selected", additionalProperties: { survey_key: NPS_SURVEY_KEY, score: String(selected) } });
    setStep("followup");
  }, []);

  const goBackToScore = useCallback(() => setStep("score"), []);

  const submit = useCallback(
    async (skipped: boolean) => {
      if (score === null || sending) return;
      const activeFollowup = FOLLOWUP_QUESTIONS[scoreBucketFor(score)];
      setSending(true);
      setError(null);
      try {
        await submitSurveyResponse({
          responseId,
          score,
          followupQuestionId: skipped ? null : activeFollowup.id,
          followupAnswer: skipped ? null : answer.trim() || null,
          consentToQuote: !skipped && activeFollowup.showQuoteConsent ? consentToQuote : false,
        });
        trackEvent({
          pageName: "Survey Widget",
          eventName: skipped ? "survey_response_skipped" : "survey_response_submitted",
          additionalProperties: { survey_key: NPS_SURVEY_KEY, score: String(score) },
        });
        setStep("thanks");
      } catch (e) {
        console.error("[survey] failed to submit response:", e instanceof Error ? e.message : String(e));
        setError("Could not send your answer. Please try again.");
      } finally {
        setSending(false);
      }
    },
    [responseId, score, answer, consentToQuote, sending],
  );

  const skip = useCallback(() => void submit(true), [submit]);
  const send = useCallback(() => void submit(false), [submit]);

  const close = useCallback(() => {
    trackEvent({ pageName: "Survey Widget", eventName: "survey_widget_dismissed", additionalProperties: { survey_key: NPS_SURVEY_KEY, step } });
    onDone();
  }, [onDone, step]);

  useEffect(() => {
    if (step !== "thanks") return;
    const timer = setTimeout(onDone, THANKS_AUTO_CLOSE_MS);
    return () => clearTimeout(timer);
  }, [step, onDone]);

  return {
    step,
    score,
    bucket: score !== null ? scoreBucketFor(score) : null,
    answer,
    consentToQuote,
    sending,
    error,
    portalTarget,
    selectScore,
    goBackToScore,
    setAnswer,
    setConsentToQuote,
    skip,
    send,
    close,
  };
}
