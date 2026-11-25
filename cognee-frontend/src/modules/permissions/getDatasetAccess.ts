import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import { asPermissions, asRecords, asString, type JsonRecord } from "@/modules/permissions/parseResponse";
import type { CogneeInstance } from "@/modules/instances/types";
import type { AccessGrant, DatasetAccess, PrincipalType } from "@/modules/permissions/types";

const PRINCIPAL_TYPES: readonly PrincipalType[] = ["user", "role", "tenant"];

function isPrincipalType(value: unknown): value is PrincipalType {
  return typeof value === "string" && (PRINCIPAL_TYPES as readonly string[]).includes(value);
}

function toGrant(entry: JsonRecord): AccessGrant | null {
  const principalId = asString(entry.principalId);
  if (!principalId || !isPrincipalType(entry.principalType)) return null;
  return {
    principalType: entry.principalType,
    principalId,
    name: asString(entry.name),
    permissions: asPermissions(entry.permissions),
    parentUserId: asString(entry.parentUserId),
  };
}

function toGrants(value: unknown): AccessGrant[] {
  return asRecords(value).flatMap((entry) => toGrant(entry) ?? []);
}

function toDatasetAccess(entry: JsonRecord): DatasetAccess | null {
  const datasetId = asString(entry.datasetId);
  const datasetName = asString(entry.datasetName);
  if (!datasetId || datasetName === null) return null;
  return {
    datasetId,
    datasetName,
    ownerId: asString(entry.ownerId),
    isOwner: entry.isOwner === true,
    permissions: asPermissions(entry.permissions),
    sources: toGrants(entry.sources),
    sharedWith: Array.isArray(entry.sharedWith) ? toGrants(entry.sharedWith) : null,
  };
}

/** Who holds what on each dataset the caller can read, and why. */
export default async function getDatasetAccess(
  instance: CogneeInstance,
  datasetIds?: readonly string[],
): Promise<DatasetAccess[]> {
  const params = new URLSearchParams();
  datasetIds?.forEach((id) => params.append("dataset_id", id));
  const query = params.toString();
  // Resolves access for every readable dataset: a build, not a lookup, so it gets the slow-read budget.
  const response = await instance.fetch(`/v1/datasets/access${query ? `?${query}` : ""}`, {
    timeoutMs: SLOW_READ_TIMEOUT_MS,
  });
  const body: unknown = await response.json();
  return asRecords(body).flatMap((entry) => toDatasetAccess(entry) ?? []);
}
