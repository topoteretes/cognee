from typing import FrozenSet

PERMISSION_TYPES = ["read", "write", "delete", "share"]

# Role names that grant permission to manage users in a tenant (list users, assign/remove
# roles, add/remove users, etc.). Tenant owner is always allowed regardless of this set.
# Add role names here when introducing tenant_admin, org_admin, etc.
USER_MANAGEMENT_ALLOWED_ROLE_NAMES: FrozenSet[str] = frozenset(
    {"tenant_admin"}  # extend with more role names as needed
)
