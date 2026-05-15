from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions import PERMISSION_TYPES
from cognee.modules.users.exceptions import PermissionNotFoundError

from ...models import Principal, ACL, Permission


async def revoke_permission_on_dataset(
    principal: Principal,
    dataset_id: UUID,
    permission_name: str,
):
    """
    Revoke a specific permission on a dataset from a principal.

    Args:
        principal: Principal whose permission is being revoked
        dataset_id: Id of the dataset
        permission_name: Name of permission to revoke
    """
    if permission_name not in PERMISSION_TYPES:
        raise PermissionNotFoundError(
            message=f"{permission_name} not found or not in allowed permission types"
        )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        permission = (
            (await session.execute(select(Permission).filter(Permission.name == permission_name)))
            .scalars()
            .first()
        )

        if permission is None:
            return  # Permission type doesn't exist yet, nothing to revoke

        await session.execute(
            delete(ACL).where(
                ACL.principal_id == principal.id,
                ACL.dataset_id == dataset_id,
                ACL.permission_id == permission.id,
            )
        )
        await session.commit()
