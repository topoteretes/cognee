from sqlalchemy import literal, select, union_all

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Dataset import Dataset
from cognee.shared.logging_utils import get_logger

from ...models.ACL import ACL
from ...models.Permission import Permission
from ...models.User import User
from ...models.UserRole import UserRole
from ...models.UserTenant import UserTenant

logger = get_logger()


async def get_all_user_permission_datasets(user: User, permission_type: str) -> list[Dataset]:
    """
        Return the datasets the user effectively holds a permission on, within
        the tenant they are currently in.

        Effective means the union of what these principals hold directly:

        * the user;
        * the agents the user created — the user holds the agents' keys, so this
          grants nothing the user could not already do; only the agents' own
          grants count, their tenants and roles are the user's;
        * every tenant the user is a member of;
        * the user's roles, which exist only while the user is in a tenant.

        Two round trips, whatever the number of principals: one UNION collects
        the membership ids, one ACL query reads the grants of all of them and
        narrows to the user's current tenant in SQL.
    Args:
        user
        permission_type

    Returns:
        list[Dataset]: The user's effective datasets in their current tenant,
        each once.
    """
    memberships = union_all(
        select(User.id.label("principal_id"), literal("agent").label("kind")).where(
            User.parent_user_id == user.id
        ),
        select(UserTenant.tenant_id, literal("tenant")).where(UserTenant.user_id == user.id),
        select(UserRole.role_id, literal("role")).where(UserRole.user_id == user.id),
    )
    tenant_filter = (
        Dataset.tenant_id == user.tenant_id if user.tenant_id else Dataset.tenant_id.is_(None)
    )

    async with get_relational_engine().get_async_session() as session:
        rows = (await session.execute(memberships)).all()
        by_kind: dict[str, set] = {"agent": set(), "tenant": set(), "role": set()}
        for principal_id, kind in rows:
            by_kind[kind].add(principal_id)
        principal_ids = {user.id} | by_kind["agent"] | by_kind["tenant"]
        if by_kind["tenant"]:
            principal_ids |= by_kind["role"]

        result = await session.execute(
            select(Dataset)
            .join(ACL, ACL.dataset_id == Dataset.id)
            .join(Permission, Permission.id == ACL.permission_id)
            .where(
                ACL.principal_id.in_(principal_ids),
                Permission.name == permission_type,
                tenant_filter,
            )
        )
        # One dataset can be reached through several principals; keep it once.
        unique: dict = {}
        for dataset in result.scalars().all():
            unique.setdefault(dataset.id, dataset)
        return list(unique.values())
