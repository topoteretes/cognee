from sqlalchemy import select
from sqlalchemy.orm import joinedload

from cognee.infrastructure.databases.relational import get_relational_engine

from ...models.Principal import Principal
from cognee.modules.data.models.Dataset import Dataset
from ...models.ACL import ACL


async def get_principal_datasets(principal: Principal, permission_type: str) -> list[Dataset]:
    """
        Return a list of datasets for which the user (principal) has a certain permission.
    Args:
        principal: Information about the user
        permission_type: Type of permission

    Returns:
        list[Dataset]: List of datasets for which the user (principal)
        has the permission (permission_type).

    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # If dataset id was returned it means the principal has permission to access it
        result = await session.execute(
            select(ACL)
            .join(ACL.permission)
            .options(joinedload(ACL.dataset))
            .where(ACL.principal_id == principal.id)
            .where(ACL.permission.has(name=permission_type))
        )
        acls = result.unique().scalars().all()
        return [acl.dataset for acl in acls]
