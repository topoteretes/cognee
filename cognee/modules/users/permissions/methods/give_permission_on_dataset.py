from sqlalchemy.future import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ...models import Principal, ACL, Permission
from uuid import UUID
from cognee.modules.users.permissions import PERMISSION_TYPES
from cognee.modules.users.exceptions import PermissionNotFoundError


async def give_permission_on_dataset(
    principal: Principal,
    dataset_id: UUID,
    permission_name: str,
):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        permission = (
            (await session.execute(select(Permission).filter(Permission.name == permission_name)))
            .scalars()
            .first()
        )

        if permission_name not in PERMISSION_TYPES:
            # If permission is not in allowed permission types
            raise PermissionNotFoundError(
                message=f"{permission_name} not found or not in allowed permission types"
            )
        elif permission is None:
            permission = Permission(name=permission_name)
            existing_acl = None
        else:
            # Check if the ACL entry already exists to avoid duplicates
            existing_acl = await session.execute(
                select(ACL).filter(
                    ACL.principal_id == principal.id,
                    ACL.dataset_id == dataset_id,
                    ACL.permission_id == permission.id,
                )
            )
            existing_acl = existing_acl.scalars().first()

        # If no existing ACL entry is found, proceed to add a new one
        if existing_acl is None:
            acl = ACL(principal_id=principal.id, dataset_id=dataset_id, permission=permission)
            session.add(acl)
            await session.commit()
