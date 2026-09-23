const LOCK_KEY_PREFIX = "cognee_survey_check_lock";
const LOCK_TTL_MS = 5000;

// Best-effort mitigation for multiple tabs open at once — not a real
// distributed lock (localStorage has no compare-and-swap across tabs), just
// enough to stop two tabs launched together from both firing the eligibility
// check within the same few seconds. A short TTL, not a permanent flag, so a
// genuinely new session later can still check again.
export function acquireCrossTabSurveyLock(surveyKey: string): boolean {
  if (typeof window === "undefined") return true;

  const key = `${LOCK_KEY_PREFIX}:${surveyKey}`;
  const now = Date.now();

  try {
    const existing = window.localStorage.getItem(key);
    if (existing) {
      const existingAt = Number(existing);
      if (!Number.isNaN(existingAt) && now - existingAt < LOCK_TTL_MS) {
        return false;
      }
    }
    window.localStorage.setItem(key, String(now));
    return true;
  } catch {
    // Private browsing / storage disabled — fail open rather than silently
    // never showing the survey.
    return true;
  }
}
