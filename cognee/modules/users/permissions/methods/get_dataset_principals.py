from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)

from ...models.ACL import ACL


def _display_name(principal) -> str:
    """Users are known by their email, roles and tenants by their name."""
    if principal.type == "user":
        return principal.email or ""
    return getattr(principal, "name", "") or ""


async def get_dataset_principals(dataset_id: UUID, user_id: UUID) -> list[dict]:
    """
        Return the principals holding a permission on a dataset, with the
        permissions each of them holds.

        The inverse of get_principal_datasets: that one answers "which datasets
        can this principal reach", this one "who can reach this dataset".

        Authorization matches the grant side — the caller must hold "share" on
        the dataset, which raises PermissionDeniedError otherwise, so a caller
        who cannot share it cannot enumerate who it is shared with either.

        A tenant is itself a principal: a dataset shared workspace-wide appears
        as one "tenant" entry rather than one entry per member.
    Args:
        dataset_id: Id of the dataset
        user_id: Id of the request owner

    Returns:
        list[dict]: [{"principal_id", "kind", "name", "permissions": [str]}],
        one entry per principal, permissions sorted for a stable response.
    """
    # Raises PermissionDeniedError when the caller cannot share this dataset.
    await get_specific_user_permission_datasets(user_id, "share", [dataset_id])

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(ACL)
            .options(joinedload(ACL.principal), joinedload(ACL.permission))
            .where(ACL.dataset_id == dataset_id)
        )
        acls = result.unique().scalars().all()

    # One row per principal, its permissions collected — the table stores a row
    # per (principal, permission), which would otherwise repeat the principal.
    principals: dict[UUID, dict] = {}

    for acl in acls:
        if acl.principal is None or acl.permission is None:
            continue

        entry = principals.get(acl.principal_id)
        if entry is None:
            entry = {
                "principal_id": str(acl.principal_id),
                # The polymorphic discriminator: "user", "role" or "tenant".
                "kind": acl.principal.type,
                "name": _display_name(acl.principal),
                "permissions": set(),
            }
            principals[acl.principal_id] = entry

        entry["permissions"].add(acl.permission.name)

    return [{**entry, "permissions": sorted(entry["permissions"])} for entry in principals.values()]
