"use client";

import { useEffect } from "react";
import { createPortal } from "react-dom";
import { FOLLOWUP_QUESTIONS } from "@/modules/survey/surveyConfig";
import { useSurveyWidgetState } from "./useSurveyWidgetState";
import SurveyScoreStep from "./SurveyScoreStep";
import SurveyFollowupStep from "./SurveyFollowupStep";
import SurveyThanksStep from "./SurveyThanksStep";

const SCORE_QUESTION = "How likely are you to recommend Cognee to a colleague?";

interface SurveyWidgetProps {
  responseId: string;
  onDone: () => void;
}

export default function SurveyWidget({ responseId, onDone }: SurveyWidgetProps): React.ReactElement | null {
  const survey = useSurveyWidgetState(responseId, onDone);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.key !== "Escape") return;
      // Ignore Escape while typing — otherwise a reflexive Escape mid-answer
      // silently discards a half-written follow-up instead of just doing
      // nothing, which the user never asked for.
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
      survey.close();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [survey]);

  if (!survey.portalTarget) return null;

  const followup = survey.bucket ? FOLLOWUP_QUESTIONS[survey.bucket] : null;

  return createPortal(
    // Not aria-modal: this is a non-blocking corner prompt, the rest of the
    // app stays fully interactive and keyboard-reachable behind it.
    // aria-live: this appears without any user action, so screen reader users
    // need it announced the same way a toast would be (WCAG 4.1.3).
    <div
      className="fixed bottom-6 right-6 z-[1000] w-[400px] max-w-[calc(100vw-48px)]"
      role="dialog"
      aria-labelledby="survey-widget-title"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="rounded-xl border border-cognee-border bg-white p-5 shadow-2xl">
        <div className="mb-1 flex items-start justify-between gap-3">
          <h2 id="survey-widget-title" className="m-0 text-sm font-bold text-cognee-dark">
            Quick question
          </h2>
          <button type="button" onClick={survey.close} aria-label="Close" className="cursor-pointer rounded-md p-1 text-base text-cognee-placeholder hover:text-cognee-body">
            ✕
          </button>
        </div>

        {survey.step === "score" && (
          <SurveyScoreStep question={SCORE_QUESTION} selectedScore={survey.score} onSelect={survey.selectScore} />
        )}

        {survey.step === "followup" && survey.score !== null && survey.bucket && followup && (
          <SurveyFollowupStep
            score={survey.score}
            question={followup.question}
            showQuoteConsent={followup.showQuoteConsent}
            answer={survey.answer}
            consentToQuote={survey.consentToQuote}
            error={survey.error}
            sending={survey.sending}
            onAnswerChange={survey.setAnswer}
            onConsentChange={survey.setConsentToQuote}
            onBack={survey.goBackToScore}
            onSkip={survey.skip}
            onSend={survey.send}
          />
        )}

        {survey.step === "thanks" && <SurveyThanksStep message="Your score and note are on their way to the team." />}
      </div>
    </div>,
    survey.portalTarget,
  );
}
