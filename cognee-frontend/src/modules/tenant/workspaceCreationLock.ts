// Cross-tab guard for workspace creation. An in-tab ref (see
// useTenantInit.ts's firstWorkspaceProvisioningStartedRef and
// useWorkspaceCreation.ts's submitInFlightRef) stops React StrictMode /
// re-entrant calls from double-firing within one tab, but each tab has its own
// React tree — it can't see another tab at all. Two tabs of the same brand-new
// user can each independently decide "this user has zero tenants, create one",
// and the backend's own duplicate-name check is not atomic (see
// createTenant.ts), so both succeed and create two tenants.
//
// This is a soft mitigation, not a hard guarantee: it narrows the race to the
// gap between reading and writing localStorage, rather than eliminating it
// (a real fix needs a backend-side unique constraint or idempotency key).
const LOCK_KEY = "cognee_workspace_creation_lock";
// Long enough to cover the actual cross-tab race window (tabs opened within
// moments of each other), short enough that a stale lock from an abandoned
// attempt doesn't block a genuine retry from ever creating a tenant.
const LOCK_TTL_MS = 2 * 60 * 1000;

// Stored as "<holder>:<writtenAtMs>". The holder id makes releasing a
// compare-and-delete: a poll loop that waits on this lock can outlive the TTL,
// so without it a holder finishing late would clear a lock a DIFFERENT tab had
// since acquired, reopening the duplicate-create window this exists to close.
function readFreshHolder(): string | null {
  let raw: string | null;
  try {
    raw = localStorage.getItem(LOCK_KEY);
  } catch {
    return null;
  }
  if (!raw) return null;

  const separator = raw.lastIndexOf(":");
  // A bare timestamp is the pre-holder-id format. Still honored while it is
  // fresh — during a rollout the other tab may be running the old build — but
  // it can never match a release, so it simply ages out.
  const holder = separator === -1 ? "" : raw.slice(0, separator);
  const writtenAt = Number(separator === -1 ? raw : raw.slice(separator + 1));

  if (!Number.isFinite(writtenAt) || Date.now() - writtenAt >= LOCK_TTL_MS) return null;
  return holder;
}

/** Whether some tab currently holds a non-expired creation lock. */
export function isWorkspaceCreationLockHeld(): boolean {
  return readFreshHolder() !== null;
}

/**
 * Returns a release token if the caller acquired the lock and should proceed
 * with creating the workspace, or null if another attempt (another tab, or an
 * earlier fallback in this same tab's chain) is already in flight — in which
 * case the caller should only poll for the tenant, not also create one.
 */
export function tryAcquireWorkspaceCreationLock(): string | null {
  try {
    if (readFreshHolder() !== null) return null;
    const holder = crypto.randomUUID();
    localStorage.setItem(LOCK_KEY, `${holder}:${Date.now()}`);
    return holder;
  } catch {
    // Storage unavailable (private browsing, quota, etc.) — fail open rather
    // than block workspace creation over a storage access error. The caller
    // still gets a token so its release path stays symmetric.
    return "unlocked";
  }
}

/** No-op unless this holder still owns the lock. */
export function releaseWorkspaceCreationLock(holder: string): void {
  try {
    if (readFreshHolder() !== holder) return;
    localStorage.removeItem(LOCK_KEY);
  } catch {
    // best-effort — TTL expiry is the authoritative fallback.
  }
}
