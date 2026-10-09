// Cross-tab guard for first-workspace creation. An in-tab ref (see
// useTenantInit.ts's firstWorkspaceProvisioningStartedRef) stops React
// StrictMode / re-entrant calls from double-firing within one tab, but each
// tab has its own React tree — it can't see another tab at all. Two tabs of
// the same brand-new user can each independently decide "this user has zero
// tenants, create one", and the backend's own duplicate-name check is not
// atomic (see createTenant.ts), so both succeed and create two tenants.
//
// This is a soft mitigation, not a hard guarantee: it narrows the race to the
// gap between reading and writing localStorage, rather than eliminating it
// (a real fix needs a backend-side unique constraint or idempotency key).
const LOCK_KEY = "cognee_workspace_creation_lock";
// Long enough to cover the actual cross-tab race window (tabs opened within
// moments of each other), short enough to have expired well before the
// longest failure-and-retry cycle (waitForPodReady's 10-minute cold-start
// budget) — a stale lock from an abandoned attempt must not block a genuine
// retry from ever creating a tenant.
const LOCK_TTL_MS = 2 * 60 * 1000;

function isLockFresh(raw: string | null): boolean {
  if (!raw) return false;
  const writtenAt = Number(raw);
  return Number.isFinite(writtenAt) && Date.now() - writtenAt < LOCK_TTL_MS;
}

// Returns true if the caller acquired the lock and should proceed with
// creating the workspace; false if another attempt (another tab, or an
// earlier fallback in this same tab's chain) is already in flight, in which
// case the caller should only poll for the tenant, not also create one.
export function tryAcquireWorkspaceCreationLock(): boolean {
  try {
    if (isLockFresh(localStorage.getItem(LOCK_KEY))) return false;
    localStorage.setItem(LOCK_KEY, String(Date.now()));
    return true;
  } catch {
    // Storage unavailable (private browsing, quota, etc.) — fail open rather
    // than block workspace creation over a storage access error.
    return true;
  }
}

export function releaseWorkspaceCreationLock(): void {
  try {
    localStorage.removeItem(LOCK_KEY);
  } catch {
    // best-effort — TTL expiry is the authoritative fallback.
  }
}
