export const DATASET_PERMISSIONS = ["read", "write", "delete", "share"] as const;

export type DatasetPermission = (typeof DATASET_PERMISSIONS)[number];

export type AccessLevel = "view" | "edit" | "full";

export interface AccessLevelOption {
  value: AccessLevel;
  label: string;
  hint: string;
  permissions: readonly DatasetPermission[];
}

// A group's access is chosen per dataset and is the same for every member of
// the group: cognee stores it as ACL rows on the role, and a role has no
// per-member level. Each level is a fixed bundle of the pod's permissions.
export const ACCESS_LEVELS: readonly AccessLevelOption[] = [
  { value: "view", label: "Can view", hint: "Search and read it", permissions: ["read"] },
  { value: "edit", label: "Can edit", hint: "Also add and cognify data", permissions: ["read", "write"] },
  {
    value: "full",
    label: "Full access",
    hint: "Also share and delete it",
    permissions: ["read", "write", "delete", "share"],
  },
];

export function isDatasetPermission(value: string): value is DatasetPermission {
  return (DATASET_PERMISSIONS as readonly string[]).includes(value);
}

export function permissionsFor(level: AccessLevel): readonly DatasetPermission[] {
  return ACCESS_LEVELS.find((option) => option.value === level)?.permissions ?? [];
}

/**
 * The level whose bundle the held permissions exactly equal, or null when
 * they don't match any bundle. A grant made outside the UI can hold a set no
 * level matches exactly (read+share without write, say, or read+write+delete
 * without share); showing it as the closest covered level would hide the
 * extra permission and silently drop it the moment someone picks that level
 * in the UI, so it reads as "Custom" instead.
 */
export function levelFor(permissions: Iterable<string>): AccessLevel | null {
  const held = new Set(permissions);
  const exact = ACCESS_LEVELS.find(
    (option) => option.permissions.length === held.size && option.permissions.every((p) => held.has(p)),
  );
  return exact?.value ?? null;
}

export function accessLevelLabel(level: AccessLevel): string {
  return ACCESS_LEVELS.find((option) => option.value === level)?.label ?? level;
}
