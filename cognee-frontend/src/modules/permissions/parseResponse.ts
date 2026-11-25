import { isDatasetPermission, type DatasetPermission } from "@/modules/permissions/accessLevels";

export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asRecords(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

export function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function asPermissions(value: unknown): DatasetPermission[] {
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is DatasetPermission => typeof entry === "string" && isDatasetPermission(entry));
}
