const KEY = "cognee-onboarding-complete";

// Mirrors the "onboarding complete" signal into both localStorage (existing
// client-side reads) and a cookie (readable by middleware.ts). The cookie is
// what lets the dashboard route redirect server-side, before any HTML is
// sent — a client-only flag can't avoid at least one flash-of-dashboard
// render, since React has to mount before an effect can act on it.
export function markOnboardingCompleteLocally() {
  try {
    localStorage.setItem(KEY, "1");
  } catch {
    /* ignore */
  }
  try {
    document.cookie = `${KEY}=1; path=/; max-age=31536000; samesite=lax`;
  } catch {
    /* ignore */
  }
}

export function clearOnboardingCompleteLocally() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  try {
    document.cookie = `${KEY}=; path=/; max-age=0`;
  } catch {
    /* ignore */
  }
}
