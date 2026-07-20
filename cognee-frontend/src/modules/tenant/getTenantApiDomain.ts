// Resolves the base domain tenant pods are served from (e.g. "127.0.0.1.nip.io"
// locally). Shared by buildTenantUrl.ts (per-tenant URL construction) and
// next.config.ts (CSP connect-src) so both stay in sync with a single source
// of truth instead of duplicating this env-parsing logic.
export default function getTenantApiDomain(): string {
  const explicit = process.env.NEXT_PUBLIC_TENANT_API_DOMAIN;
  const mgmtUrl = process.env.NEXT_PUBLIC_MANAGEMENT_API_URL ?? "";
  const domainMatch = mgmtUrl.match(/api\.(.*)/);
  return explicit || (domainMatch?.[1] ?? "");
}
