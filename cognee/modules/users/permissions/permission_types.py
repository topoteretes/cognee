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

# Deprecated in favour of the MANAGE_USERS capability. Kept because a tenant
# that has not been migrated yet has no capability rows at all, so resolution
# returns an empty set for everyone but the owner and this set is the only thing
# still granting its "admin" role user management.
#
# Remove it once existing "admin" roles have been granted MANAGE_USERS through
# the capability endpoints on the permissions router. Removing it before then
# locks those tenants out.
USER_MANAGEMENT_ALLOWED_ROLE_NAMES: FrozenSet[str] = frozenset({"admin"})
