export default async function addUserToTenant(email: string, tenantId: string): Promise<void> {
  const params = new URLSearchParams({ email, tenant_id: tenantId });
  await fetch(`/api/tenants/users?${params}`, { method: "POST" });
}
