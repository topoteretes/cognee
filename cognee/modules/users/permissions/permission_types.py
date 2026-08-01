from typing import FrozenSet

PERMISSION_TYPES = ["read", "write", "delete", "share"]

# Capabilities are tenant-scoped actions, as opposed to PERMISSION_TYPES which are
# always about one dataset. They are stored in the same `permissions` table but
# granted through Tenant/Role/UserDefaultPermissions, which carry no dataset column.
#
# The catalog is system-defined because the code has to know what each name gates;
# only the assignment of a capability to a tenant, role or user is data the owner edits.
MANAGE_USERS = "manage_users"

CAPABILITY_TYPES: FrozenSet[str] = frozenset({MANAGE_USERS})

# Deprecated: superseded by the MANAGE_USERS capability. Kept as a fallback so
# tenants that today rely on a role literally named "admin" do not lose user
# management the moment they upgrade. Remove once those roles have been granted
# the capability explicitly.
USER_MANAGEMENT_ALLOWED_ROLE_NAMES: FrozenSet[str] = frozenset({"admin"})
