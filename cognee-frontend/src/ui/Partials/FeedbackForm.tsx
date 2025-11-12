"use client";

import { useState } from "react";
import { LoadingIndicator } from "@/ui/App";
import { fetch, useBoolean } from "@/utils";
import { CTAButton, TextArea } from "@/ui/elements";

interface SignInFormPayload extends HTMLFormElement {
  feedback: HTMLTextAreaElement;
}

interface FeedbackFormProps {
  onSuccess: () => void;
}

export default function FeedbackForm({ onSuccess }: FeedbackFormProps) {
  const {
    value: isSubmittingFeedback,
    setTrue: disableFeedbackSubmit,
    setFalse: enableFeedbackSubmit,
  } = useBoolean(false);

  const [feedbackValue, setFeedbackValue] = useState("");
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const signIn = (event: React.FormEvent<SignInFormPayload>) => {
    event.preventDefault();

    setFeedbackError(null);
    disableFeedbackSubmit();

    fetch("/v1/crewai/feedback", {
      method: "POST",
      body: JSON.stringify({
        feedback: feedbackValue,
      }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then(response => response.json())
      .then(() => {
        onSuccess();
        setFeedbackValue("");
      })
      .catch(error => setFeedbackError(error.detail))
      .finally(() => enableFeedbackSubmit());
  };

  return (
    <form onSubmit={signIn} className="flex flex-col gap-2">
      <div className="flex flex-col gap-2">
        <div className="mb-4">
          <label className="block text-white" htmlFor="feedback">Feedback on agent&apos;s reasoning</label>
          <TextArea
            id="feedback"
            name="feedback"
            placeholder="Your feedback"
            value={feedbackValue}
            onChange={setFeedbackValue}
          />
        </div>
      </div>

      <CTAButton type="submit">
        <span>Submit feedback</span>
        {isSubmittingFeedback && <LoadingIndicator />}
      </CTAButton>

      {feedbackError && (
        <span className="text-s text-white">{feedbackError}</span>
      )}
    </form>
  )
}
