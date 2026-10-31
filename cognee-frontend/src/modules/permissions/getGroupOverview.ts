import { asPermissions, asRecords, asString } from "@/modules/permissions/parseResponse";
import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import { levelFor } from "@/modules/permissions/accessLevels";
import type { CogneeInstance } from "@/modules/instances/types";
import type { GroupDatasetGrant, GroupMember, GroupOverview } from "@/modules/permissions/types";

function toMembers(value: unknown): GroupMember[] {
  return asRecords(value).flatMap((member) => {
    const id = asString(member.id);
    const email = asString(member.email);
    return id && email ? [{ id, email }] : [];
  });
}

function toDatasets(value: unknown): GroupDatasetGrant[] {
  return asRecords(value).flatMap((dataset) => {
    const id = asString(dataset.id);
    const name = asString(dataset.name);
    if (!id || !name) return [];
    const permissions = asPermissions(dataset.permissions);
    return [{ id, name, permissions, level: levelFor(permissions) }];
  });
}

/**
 * The groups the signed-in user may see, each with its members and the datasets
 * it holds at the level it holds them. One request in place of a members call
 * and four per-permission dataset calls per group; the pod decides which groups
 * are visible (a workspace admin sees all, anyone else only their own).
 *
 * Pass `groupIds` to read only some groups, e.g. the one open in the detail view.
 */
export default async function getGroupOverview(
  instance: CogneeInstance,
  groupIds: readonly string[] = [],
): Promise<GroupOverview[]> {
  const params = new URLSearchParams();
  for (const groupId of groupIds) params.append("group_id", groupId);
  const query = params.size > 0 ? `?${params.toString()}` : "";

  // Built from several reads of the tenant database: the default read budget is for lookups.
  const response = await instance.fetch(`/v1/permissions/groups${query}`, { timeoutMs: SLOW_READ_TIMEOUT_MS });
  const body: unknown = await response.json();

  return asRecords(body).flatMap((group) => {
    const id = asString(group.id);
    const name = asString(group.name);
    if (!id || !name) return [];
    return [{ id, name, members: toMembers(group.members), datasets: toDatasets(group.datasets) }];
  });
}
