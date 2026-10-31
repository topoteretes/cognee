import { HttpError } from "@/services/http";

// How long the pod may keep rejecting the workspace's API key (401/403) before
// the readiness poll calls the key bad rather than not synced yet. The backend
// lists a new tenant from the first moment of POST /tenants and reports it
// "healthy" once its pod is Ready and routed, and only after that creates the
// tenant database and syncs the owner and their keys into the pod
// (cloud-backend create_tenant.py). A reload or a second tab can therefore
// poll a tenant that answers 401 while every status read says "healthy"
// (CLO-1276: healthy at 10:22:11, key synced at 10:22:15), so the window is
// counted from the pod's own first 401/403, the one clock the browser can see.
// Observed gaps are seconds, but the owner sync alone has taken over 30s in a
// kind cluster (user_sync_methods.py), hence a minute. Past it, a key that
// really is wrong still ends in an error state, a minute later than it used to.
export const AUTH_SYNC_GRACE_MS = 60_000;

// The poll's two grace clocks. Kept per poll (in waitForPodReady's schedule),
// never in storage, so a reload ("Try again") starts with both closed.
export interface AuthSyncClocks {
  // When the current unbroken run of 401/403s began; null outside one.
  authErrorRunStartedAt: number | null;
  // The last time a status read answered "provisioning"; null if none has.
  provisioningSeenAt: number | null;
}

export function isAuthError(err: unknown): boolean {
  return err instanceof HttpError && (err.status === 401 || err.status === 403);
}

// The start of the run of auth errors after the failure just seen. Any other
// failure ends the run (the pod restarting, the network dropping), so a 401, a
// stretch of connection errors and a fresh 401 open a new window rather than
// closing on the old one.
export function authErrorRunStart(runStartedAt: number | null, err: unknown, now: number = Date.now()): number | null {
  if (!isAuthError(err)) return null;
  return runStartedAt ?? now;
}

// A 401/403 still means "not synced yet" while either clock is open. The
// "provisioning" one keeps the window open for as long as the backend itself
// says the tenant is not done, which is all it needs if the status ever starts
// covering the key sync.
export function isWithinAuthSyncGrace(clocks: AuthSyncClocks, now: number = Date.now()): boolean {
  return isRecent(clocks.authErrorRunStartedAt, now) || isRecent(clocks.provisioningSeenAt, now);
}

function isRecent(at: number | null, now: number): boolean {
  return at !== null && now - at < AUTH_SYNC_GRACE_MS;
}
