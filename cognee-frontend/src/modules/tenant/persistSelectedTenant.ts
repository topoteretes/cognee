const SELECTED_TENANT_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

// The selected tenant is persisted three ways on purpose:
//  - cookie: server-side API routes read the selection
//  - localStorage: survives tab close, lets a reload reconnect deterministically
//  - sessionStorage: marks that a selection was made THIS session (cleared on
//    tab close), which the auto-select logic in useTenantInit distinguishes from
//    a stale preference left over from a previous session
export default function persistSelectedTenant(tenantId: string, tenantName?: string): void {
  document.cookie = `cognee_selected_tenant=${tenantId};path=/;max-age=${SELECTED_TENANT_MAX_AGE_SECONDS};SameSite=Lax`;
  localStorage.setItem("cognee_selected_tenant", tenantId);
  if (tenantName) localStorage.setItem("cognee_selected_tenant_name", tenantName);
  sessionStorage.setItem("cognee_selected_tenant", tenantId);
}
