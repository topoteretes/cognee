// Bridges plain module code (e.g. dataset creation, anywhere in the app) to
// SurveyProvider (a React tree, mounted once) without prop drilling. Mirrors
// the insufficientCreditsBridge pattern — a single active listener, not a
// generic multi-subscriber bus, since there is only ever one SurveyProvider.

export type SurveyTriggerReason = "datasource_added";

export type SurveyTriggerEvent = {
  reason: SurveyTriggerReason;
  at: number;
};

export type SurveyTriggerListener = (event: SurveyTriggerEvent) => void;

let activeListener: SurveyTriggerListener | null = null;

export function setSurveyTriggerListener(listener: SurveyTriggerListener | null): void {
  activeListener = listener;
}

export function notifySurveyTrigger(reason: SurveyTriggerReason): void {
  try {
    activeListener?.({ reason, at: Date.now() });
  } catch {
    // Listener must never crash the call site that triggered it.
  }
}
