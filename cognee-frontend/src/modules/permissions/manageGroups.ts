import { asString, isRecord } from "@/modules/permissions/parseResponse";
import type { CogneeInstance } from "@/modules/instances/types";

// Groups are cognee roles. Creating and deleting one, and changing who is in
// it, needs manage_users in the tenant; the pod enforces it.

export async function createGroup(name: string, instance: CogneeInstance): Promise<string> {
  const params = new URLSearchParams({ role_name: name });
  const response = await instance.fetch(`/v1/permissions/roles?${params.toString()}`, { method: "POST" });
  const body: unknown = await response.json();
  const roleId = isRecord(body) ? asString(body.role_id) : null;
  if (!roleId) throw new Error("The group was created but the response did not say which one it is.");
  return roleId;
}

export async function deleteGroup(groupId: string, instance: CogneeInstance): Promise<void> {
  await instance.fetch(`/v1/permissions/roles/${groupId}`, { method: "DELETE" });
}

export async function addGroupMember(groupId: string, userId: string, instance: CogneeInstance): Promise<void> {
  const params = new URLSearchParams({ role_id: groupId });
  await instance.fetch(`/v1/permissions/users/${userId}/roles?${params.toString()}`, { method: "POST" });
}

export async function removeGroupMember(groupId: string, userId: string, instance: CogneeInstance): Promise<void> {
  const params = new URLSearchParams({ role_id: groupId });
  await instance.fetch(`/v1/permissions/users/${userId}/roles?${params.toString()}`, { method: "DELETE" });
}
