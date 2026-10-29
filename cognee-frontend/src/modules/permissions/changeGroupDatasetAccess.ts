import { permissionsFor, type AccessLevel } from "@/modules/permissions/accessLevels";
import type { CogneeInstance } from "@/modules/instances/types";

/**
 * Give a group a level on a dataset, change it, or (target null) take the
 * dataset away from the group.
 *
 * One request: the pod makes the group's permissions on the dataset exactly the
 * set of the level, in one transaction. cognee's own routes are one permission
 * per request (a dozen queries each, three for Full access, and a failure half
 * way left a set no level describes). The pod requires the caller to hold
 * `share` on the dataset.
 */
export default async function changeGroupDatasetAccess(
  groupId: string,
  datasetId: string,
  target: AccessLevel | null,
  instance: CogneeInstance,
): Promise<void> {
  const permissions = target ? permissionsFor(target) : [];
  await instance.fetch(`/v1/permissions/groups/${groupId}/datasets/${datasetId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ permissions }),
  });
}
