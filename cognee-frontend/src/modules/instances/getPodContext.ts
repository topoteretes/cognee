import { cookies } from "next/headers";
import managementFetch from "./managementFetch";

const TENANT_COOKIE = "cognee_selected_tenant";
const CACHE_TTL_MS = 60_000; // 1 minute

let cached: { data: { apiKey: string; base: string; tenant_id: string }; expiry: number } | null = null;

export default async function getPodContext() {
  if (cached && Date.now() < cached.expiry) {
    // Still check cookie in case tenant was switched
    const cookieStore = await cookies();
    const selectedTenant = cookieStore.get(TENANT_COOKIE)?.value;
    const cachedTenantFromUrl = cached.data.base.match(/tenant-([^.]+)/)?.[1];

    if (!selectedTenant || selectedTenant === cachedTenantFromUrl) {
      return cached.data;
    }
    // Tenant changed — invalidate cache
    cached = null;
  }

  const [svcResp, keysResp, tenantResp] = await Promise.all([
    managementFetch("/tenants/current/service-url"),
    managementFetch("/api-keys"),
    managementFetch("/tenants/current"),
  ]);

  const { service_url } = await svcResp.json();
  const keys = await keysResp.json();
  const { tenant_id } = await tenantResp.json();
  const apiKey = keys[0]?.key as string;

  let serviceUrl = (service_url as string).replace(/^http:\/\//, "https://");
  let activeTenantId = tenant_id as string;

  // Check if user has selected a different tenant
  const cookieStore = await cookies();
  const selectedTenant = cookieStore.get(TENANT_COOKIE)?.value;
  if (selectedTenant && selectedTenant !== tenant_id) {
    serviceUrl = serviceUrl.replace(/tenant-[^.]+/, `tenant-${selectedTenant}`);
    activeTenantId = selectedTenant;
  }

  const data = {
    apiKey,
    base: serviceUrl + "/api",
    tenant_id: activeTenantId,
  };

  cached = { data, expiry: Date.now() + CACHE_TTL_MS };
  return data;
}
