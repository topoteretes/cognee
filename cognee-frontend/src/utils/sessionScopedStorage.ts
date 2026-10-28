// Storage tied to a specific Auth0 account, not to the browser/device. If a
// second account logs in on the same tab (or a stale tab never ran the
// signout script), these keys must not leak into the new session — they
// carry tenant selection, cached credentials, and workspace state.
//
// Deliberately excludes: analytics/attribution keys (cognee_anonymous_id,
// cognee_first_touch, etc. — track the device, not the account) and cosmetic
// UI preferences (cognee-pipeline-settings, cognee-credits-banner-dismissed,
// cognee-terminal-demo-shown — losing these on account switch is not a
// correctness or security issue).
// Deliberately NOT in the list below, and kept in localStorage rather than
// sessionStorage. All three store the user id they belong to, so a second
// account on this browser is a different key and nothing needs clearing; and
// clearing them was actively harmful, because UserProvider is an ancestor of
// the component that sets them and React commits child effects first, so the
// wipe landed immediately after the write and the next mount emitted a
// duplicate `Signed Up`. Declared here rather than at the use site so there is
// one copy of each string, next to the list that must not contain them.
export const SIGNUP_TRACKED_FLAG = "cognee_signup_tracked";
export const FIRST_LOGIN_CHECK_FLAG = "cognee_first_login_checked";
export const FIRST_LOGIN_RETRY_FLAG = "cognee_first_login_retry_after";
// Throttles the consent-failure report to once per account, while the write
// itself keeps retrying on every load (see IdentifyUser).
export const CONSENT_REPORTED_FLAG = "cognee_consent_failure_reported";

export const SESSION_SCOPED_STORAGE_KEYS = [
  "cognee_selected_tenant",
  "cognee_selected_tenant_name",
  "cognee_init_cache",
  "cognee_plan_type",
  "cognee_stripe_ensured",
  "cognee_pending_workspace_name",
  "cognee_pre_checkout_tenants",
  "cognee_user_email",
  "cognee_user_id",
  "cognee-awaiting-dataset",
  "cognee-graph-nodes",
  // Carries "this workspace is mid-wake" across a reload. Account-scoped for
  // the same reason as the tenant selection above: it relaxes the pod-readiness
  // budget and auth handling, which must not be inherited by whoever logs in next.
  "cognee_waking_tenant",
  // Held while a brand-new account's first workspace is being created (see
  // workspaceCreationLock.ts). It guards that account's creation only; left
  // behind, it keeps the next brand-new account on this browser from creating its own.
  "cognee_workspace_creation_lock",
] as const;

// Per-tenant keys — the tenant id suffix varies, so these are matched by prefix.
//
// cognee_last_graph_dataset: is keyed by pod instanceId, which IS the tenant
// id — so it is shared by every user of that tenant on this browser, not just
// by one account. Left unregistered, the next user to sign in here opens the
// graph on the previous one's dataset (see lastGraphDataset.ts).
//
// cognee-search-type- is keyed the same way and is not a cosmetic preference:
// it decides which retriever a new Search conversation runs on. Inherited by
// the next account on this browser, their questions would go to the code graph
// and come back as name lookups.
export const SESSION_SCOPED_STORAGE_PREFIXES = [
  "cognee-connected-integrations-",
  "cognee-search-active-conversation-",
  // The questions and answers themselves, so the next account must not see them.
  "cognee-search-code-conversations-",
  "cognee-search-type-",
  "cognee_last_graph_dataset:",
] as const;

// Cookies that must not outlive the account that set them. The domain has to
// match the one they were written with or the delete creates and expires a
// separate host-only cookie and leaves the real one in place.
//
// The two sign-up cookies are deliberately NOT here, and are cleared only by
// the sign-out route. An account switch cannot tell a previous visitor's
// abandoned consent answer from one this browser wrote seconds ago for the
// account now arriving, and clearing on the switch destroyed the second kind.
// Attribution is handled where the answer is written instead: the flush only
// writes it for an account created after the server timestamp inside the JWE,
// rather than deleting an answer whose owner may yet sign in (see
// @/modules/users/flushMarketingConsent).
export const SESSION_SCOPED_COOKIES = [
  { name: "cognee_selected_tenant", domain: null },
] as const;

export function clearSessionScopedStorage(): void {
  for (const key of SESSION_SCOPED_STORAGE_KEYS) {
    localStorage.removeItem(key);
    sessionStorage.removeItem(key);
  }
  for (const store of [localStorage, sessionStorage]) {
    for (let i = store.length - 1; i >= 0; i--) {
      const key = store.key(i);
      if (key && SESSION_SCOPED_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
        store.removeItem(key);
      }
    }
  }
  for (const { name, domain } of SESSION_SCOPED_COOKIES) {
    document.cookie = `${name}=;Max-Age=0;path=/${domain ? `;domain=${domain}` : ""}`;
  }
}

// sessionStorage, not localStorage: the fingerprint only needs to catch an
// account switch within the same tab's lifetime — a stale localStorage
// fingerprint from days ago would false-positive on legitimate re-logins.
const SESSION_FINGERPRINT_KEY = "cognee_session_user_id";

// Wipes session-scoped storage the moment a DIFFERENT account is detected in
// this tab, and is safe to call redundantly from every consumer of that
// storage (UserProvider, useTenantInit, ...) rather than only once in a
// single effect. Calling it inline, synchronously, at the point of use closes
// the ordering race a single top-level effect can't guarantee: React commits
// child effects before parent effects, so a child reading cognee_init_cache
// before the parent's own fingerprint effect fires would otherwise hydrate
// the previous account's apiKey first.
export function ensureSessionFingerprint(userId: string | null | undefined): void {
  if (!userId) return;
  const previousUserId = sessionStorage.getItem(SESSION_FINGERPRINT_KEY);
  if (previousUserId && previousUserId !== userId) {
    clearSessionScopedStorage();
  }
  sessionStorage.setItem(SESSION_FINGERPRINT_KEY, userId);
}
