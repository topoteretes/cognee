import { permissionsFor, type AccessLevel } from "@/modules/permissions/accessLevels";
import type { CogneeInstance } from "@/modules/instances/types";

/**
 * Share a dataset with a group, a member of the workspace, or the whole workspace
 * (its id is the tenant id) at one level, change that level, or (target null)
 * take the access away.
 *
 * One request: the pod makes the principal's permissions on the dataset exactly
 * the set of the level, in one transaction, and refuses it unless the caller holds
 * `share` on the dataset (and would not lose their last `share`).
 */
export default async function changeDatasetAccess(
  principalId: string,
  datasetId: string,
  target: AccessLevel | null,
  instance: CogneeInstance,
): Promise<void> {
  const permissions = target ? permissionsFor(target) : [];
  await instance.fetch(`/v1/permissions/principals/${principalId}/datasets/${datasetId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ permissions }),
  });
}
