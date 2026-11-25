// Tenant-scoped capabilities from cognee's capability layer. A workspace admin
// is a member holding MANAGE_USERS; GRANT/REVOKE_CAPABILITIES are what it takes
// to make or unmake one, and only the owner holds them by default.
export const MANAGE_USERS = "manage_users";
export const GRANT_CAPABILITIES = "grant_capabilities";
export const REVOKE_CAPABILITIES = "revoke_capabilities";

export interface WorkspaceCapabilities {
  /** Invite and remove members, create groups and manage their members. */
  canManageUsers: boolean;
  /** Make a member an admin, and back. */
  canManageAdmins: boolean;
}

export const NO_CAPABILITIES: WorkspaceCapabilities = {
  canManageUsers: false,
  canManageAdmins: false,
};

export function toWorkspaceCapabilities(capabilities: readonly string[]): WorkspaceCapabilities {
  return {
    canManageUsers: capabilities.includes(MANAGE_USERS),
    canManageAdmins: capabilities.includes(GRANT_CAPABILITIES) && capabilities.includes(REVOKE_CAPABILITIES),
  };
}
