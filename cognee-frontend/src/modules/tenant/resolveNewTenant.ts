import createTenant from "./createTenant";
import getMyTenants from "./getMyTenants";
import { PERSONAL_WORKSPACE_NAME } from "./TenantContext";
import { tryAcquireWorkspaceCreationLock } from "./workspaceCreationLock";
import type { Tenant } from "./types";

const POLL_ATTEMPTS = 36; // ~3 min at 5s intervals
const POLL_INTERVAL_MS = 5000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

// Create the user's first tenant and resolve its id.
//
// POST /tenants blocks until the tenant pod is fully provisioned (1-3 min), so
// we DON'T await it before doing anything else — that's what made onboarding sit
// with no visible activity. Instead we fire it and concurrently poll the tenant
// LIST every 5s: the tenant's DB record exists within seconds of the POST, so we
// pick it up fast (and the polling GETs are visible in the network tab). We use
// the LIST, not /tenants/current, because the latter 404s until the user's
// *active* tenant_id is set, which never happens for a freshly-created tenant.
// Returns null only if neither the poll nor the create call yields a tenant
// within the window.
export default async function resolveNewTenant(isCancelled: () => boolean): Promise<Tenant | null> {
  // Cross-tab guard, same lock as useTenantInit.ts's earlier attempt: if
  // that attempt (or another tab) is still holding it, this call is a
  // fallback for a create that may well still be in flight server-side —
  // firing a second createTenant() here would risk a duplicate tenant, so
  // this just polls instead.
  const creating = tryAcquireWorkspaceCreationLock()
    ? createTenant({ noRedirectOnAuth: true, tenantName: PERSONAL_WORKSPACE_NAME })
        .catch((err) => {
          console.warn("[API] create-tenant call failed, relying on poll instead:", err instanceof Error ? err.message : err);
          return null;
        })
    : Promise.resolve(null);

  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt++) {
    if (isCancelled()) return null;
    try {
      const mine = await getMyTenants();
      const found = mine.find((t) => t.isOwner) ?? mine[0];
      if (found) return { tenant_id: found.id, tenant_name: found.name };
    } catch {
      console.warn(`[API] tenant poll attempt ${attempt + 1} failed, retrying...`);
    }
    if (isCancelled()) return null;
    await sleep(POLL_INTERVAL_MS);
  }

  // The list never surfaced it — fall back to whatever the create call returned.
  return await creating;
}
