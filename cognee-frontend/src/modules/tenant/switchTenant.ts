import { clearInitCache } from "./initCache";
import persistSelectedTenant from "./persistSelectedTenant";

// Read by useTenantInit.ts's connectToSelectedTenant on the other side of the
// reload this triggers. Value is the tenant id, not a bare "1" flag, so a
// stale leftover from an earlier switch can't be mistaken for freshness of a
// different tenant reached later in the same tab.
//
// This is a DURABLE marker, not a one-shot flag: it survives reloads and the
// PodUnreachableCard's "Try again" so a retry against a still-cold pod keeps
// the long cold-start budget + recoverable-auth handling instead of falling
// back to the short reconnect path. It is cleared only once the pod is
// confirmed ready (clearFreshlyCreatedTenant) — never merely on being read.
export const FRESHLY_CREATED_TENANT_KEY = "cognee_freshly_created_tenant";

export function markFreshlyCreatedTenant(tenantId: string): void {
  try {
    sessionStorage.setItem(FRESHLY_CREATED_TENANT_KEY, tenantId);
  } catch {
    // Best-effort — worst case this tenant just gets the normal optimistic
    // reconnect treatment, same as before this existed.
  }
}

export function isFreshlyCreatedTenant(tenantId: string): boolean {
  try {
    return sessionStorage.getItem(FRESHLY_CREATED_TENANT_KEY) === tenantId;
  } catch {
    return false;
  }
}

export function clearFreshlyCreatedTenant(): void {
  try {
    sessionStorage.removeItem(FRESHLY_CREATED_TENANT_KEY);
  } catch {
    // Best-effort.
  }
}

// Full-page navigation (not router.push) on purpose: switching tenants must
// tear down every piece of in-memory state (query caches, cogniInstance,
// pod polling) and re-run init from scratch against the new tenant.
//
// Onboarding is a once-per-user fact (see useOnboardingRedirect), not
// per-workspace, so switching or creating a workspace here never touches
// it — there is nothing workspace-scoped left to reset.
//
// isFreshlyCreated: set only when this tenant was JUST provisioned (the paid
// additional-workspace flow, right after Stripe checkout) — as opposed to
// switching to a workspace the user already had. connectToSelectedTenant
// optimistically assumes a reconnected tenant's pod is already warm (correct
// for genuinely established workspaces, including ones this device has never
// visited before — e.g. logging in fresh or opening a shared workspace for
// the first time), which is wrong for a tenant that came into existence
// seconds ago and whose pod/DNS may still be spinning up.
export default function switchTenant(
  tenantId: string,
  tenantName?: string,
  navigateTo?: string,
  isFreshlyCreated?: boolean,
): void {
  clearInitCache();
  persistSelectedTenant(tenantId, tenantName);
  if (isFreshlyCreated) {
    markFreshlyCreatedTenant(tenantId);
  }
  if (navigateTo) {
    window.location.href = navigateTo;
  } else {
    window.location.reload();
  }
}
