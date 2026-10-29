export interface Tenant {
  tenant_id: string;
  tenant_name: string;
}

// Per-tenant lifecycle status as reported by the control plane on the tenant
// LIST endpoint (GET /tenants/me/with-status) — distinct from
// TenantProvisioningStatus, which is the richer k8s view returned by
// GET /tenants/{id}/status for a single tenant.
export type TenantLifecycleStatus = "active" | "suspended";

// A suspended tenant is scaled to zero. It is resumed through the control
// plane before anything touches its pod — see wakeSuspendedTenant for why that
// is the right path even though the data-plane activator can also trigger a
// wake. Do not restate the gateway's behaviour here; it has changed twice
// already and the one description of it lives in that module.
// Annotated so the constant and the union above cannot drift apart silently.
export const TENANT_STATUS_SUSPENDED: TenantLifecycleStatus = "suspended";

// The correction applied when the list's row turns out to be stale. The row is
// only ever cleared by an accepted restore, and the backend's suspend sweep
// visits active rows alone, so a row stuck at "suspended" for a serving pod has
// nothing behind it that will repair it.
export const TENANT_STATUS_ACTIVE: TenantLifecycleStatus = "active";

// Where the app is in the wake-a-sleeping-workspace flow. "idle" covers both
// "never was asleep" and "already woken" — only a tenant actively being
// restored, or one that failed to come back, needs its own UI.
export type TenantWakeState = "idle" | "waking" | "failed";
