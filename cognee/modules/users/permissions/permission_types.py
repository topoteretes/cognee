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

# Deprecated in favour of the MANAGE_USERS capability, but still the only path
# that grants user management to a non-owner today: nothing calls
# give_default_permission_to_{tenant,role,user}, so no capability can actually be
# assigned yet and resolution returns an empty set for everyone but the owner.
#
# Remove this set once granting is wired up (a capability-assignment endpoint,
# tracked separately) and existing "admin" roles have been migrated onto
# MANAGE_USERS. Removing it before then locks those tenants out.
USER_MANAGEMENT_ALLOWED_ROLE_NAMES: FrozenSet[str] = frozenset({"admin"})
