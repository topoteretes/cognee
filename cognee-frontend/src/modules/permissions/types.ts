import type { AccessLevel, DatasetPermission } from "@/modules/permissions/accessLevels";

/** A group is a cognee role in the current tenant. */
export interface WorkspaceGroup {
  id: string;
  name: string;
  memberCount: number;
}

export interface GroupMember {
  id: string;
  email: string;
}

/** One dataset a group holds, with the access every member gets on it. */
export interface GroupDatasetGrant {
  id: string;
  name: string;
  permissions: DatasetPermission[];
  /** Null when the group holds something below "read" (made outside the UI). */
  level: AccessLevel | null;
}

/** A group with who is in it and what it holds, from GET /v1/permissions/groups. */
export interface GroupOverview {
  id: string;
  name: string;
  members: GroupMember[];
  datasets: GroupDatasetGrant[];
}

export type PrincipalType = "user" | "role" | "tenant";

/** What one principal holds on one dataset, from GET /v1/datasets/access. */
export interface AccessGrant {
  principalType: PrincipalType;
  principalId: string;
  name: string | null;
  permissions: DatasetPermission[];
  /** Set when the principal is an agent: the user it acts for. */
  parentUserId: string | null;
}

export interface DatasetAccess {
  datasetId: string;
  datasetName: string;
  ownerId: string | null;
  isOwner: boolean;
  /** The caller's effective permissions on the dataset. */
  permissions: DatasetPermission[];
  /** The grants the caller's access comes from. */
  sources: AccessGrant[];
  /** Every grant on the dataset; null unless the caller holds `share`. */
  sharedWith: AccessGrant[] | null;
}
