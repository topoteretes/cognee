import getTenantApiDomain from "./getTenantApiDomain";

// Construct tenant URL from the management API URL pattern.
// Falls back to explicit NEXT_PUBLIC_TENANT_API_DOMAIN if set.
export default function buildTenantUrl(tenantId: string): string {
  const domain = getTenantApiDomain();
  if (!domain) console.error("[INIT] Cannot build tenant URL: set NEXT_PUBLIC_TENANT_API_DOMAIN or NEXT_PUBLIC_MANAGEMENT_API_URL");
  return `https://tenant-${tenantId}.${domain}`;
}
