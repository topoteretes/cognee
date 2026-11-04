import { isRecord } from "@/modules/permissions/parseResponse";
import { MANAGE_USERS } from "@/modules/permissions/workspaceCapabilities";
import type { CogneeInstance } from "@/modules/instances/types";

/** What the signed-in user may do in the tenant (the owner holds everything). */
export async function getMyCapabilities(tenantId: string, instance: CogneeInstance): Promise<string[]> {
  const response = await instance.fetch(`/v1/permissions/tenants/${tenantId}/capabilities/me`);
  const body: unknown = await response.json();
  if (!isRecord(body) || !Array.isArray(body.capabilities)) return [];
  return body.capabilities.filter((entry): entry is string => typeof entry === "string");
}

/** Make a member a workspace admin or take it back. Owner only: it takes the
 *  grant and revoke capabilities, which an admin does not get. */
export async function setWorkspaceAdmin(
  userId: string,
  tenantId: string,
  isAdmin: boolean,
  instance: CogneeInstance,
): Promise<void> {
  const params = new URLSearchParams({ capability: MANAGE_USERS, tenant_id: tenantId });
  await instance.fetch(`/v1/permissions/capabilities/${userId}?${params.toString()}`, {
    method: isAdmin ? "POST" : "DELETE",
  });
}
