PERMISSION_TYPES = ["read", "write", "delete", "share"]

# Capabilities are tenant-scoped actions, as opposed to PERMISSION_TYPES which are
# always about one dataset. They live in their own table, principal_capabilities,
# keyed on (principal, tenant, capability).
#
# The catalog is system-defined because the code has to know what each name gates;
# only the assignment of a capability to a principal is data the owner edits.
#
# Names read as the action they allow, "manage_users" -> "manage users", which is
# what CapabilityDeniedError puts in its message.
MANAGE_USERS = "manage_users"

# Giving a capability to a principal and taking one away are separate
# capabilities, and neither comes with MANAGE_USERS: deciding who may do what in
# a tenant is a different job from adding and removing its members. Granting
# only passes on what the granter holds, directly or through a role, so
# GRANT_CAPABILITIES is not a way to reach the rest of the catalog. Revoking can
# take any capability away; the owner cannot lose theirs.
GRANT_CAPABILITIES = "grant_capabilities"
REVOKE_CAPABILITIES = "revoke_capabilities"

CAPABILITY_TYPES: frozenset[str] = frozenset(
    {MANAGE_USERS, GRANT_CAPABILITIES, REVOKE_CAPABILITIES}
)

# Deprecated in favour of granting capabilities. Kept because a tenant that has
# not been migrated yet has no capability rows at all, so resolution returns an
# empty set for everyone but the owner and this set is the only thing still
# letting its "admin" role act. A role named here passes every check that goes
# through has_grant_permission, whatever capability it asks for.
#
# Remove it once existing "admin" roles have been granted the capabilities they
# need through the capability endpoints on the permissions router. Removing it
# before then locks those tenants out.
USER_MANAGEMENT_ALLOWED_ROLE_NAMES: frozenset[str] = frozenset({"admin"})
