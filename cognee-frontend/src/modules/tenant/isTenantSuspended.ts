import { TENANT_STATUS_SUSPENDED } from "./types";

// Whether the control plane last reported this tenant as scaled to zero.
//
// Reads the status off the tenant list the caller already has, so an awake
// tenant costs no extra request on a normal page load.
//
// A tenant missing from the list, or one with no status field, is treated as
// awake. That is what makes this safe to ship ahead of the backend: the
// released control plane serves /tenants/me/with-status without a status key
// at all, so the wake flow is simply never entered until the suspend/restore
// work is tagged and promoted. A tenant that is asleep despite this returning
// false is still caught later, by the pod probe (see waitForPodReady's
// "suspended" outcome).
export default function isTenantSuspended(
  tenants: readonly { id: string; status?: string }[],
  tenantId: string,
): boolean {
  return tenants.find((t) => t.id === tenantId)?.status === TENANT_STATUS_SUSPENDED;
}
