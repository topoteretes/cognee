from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.infrastructure.databases.relational import get_relational_engine

from ...models.User import User
from ...models.ACL import ACL

logger = get_logger()


async def check_permission_on_dataset(user: User, permission_type: str, dataset_id: UUID):
    if user is None:
        user = await get_default_user()

    # # TODO: Enable user role permissions again. Temporarily disabled during rework.
    # user_roles_ids = [role.id for role in user.roles]
    user_roles_ids = []

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # If dataset id was returned it means the user has permission to access it
        result = await session.execute(
            select(ACL)
            .join(ACL.permission)
            .options(joinedload(ACL.dataset))
            .where(ACL.principal_id.in_([user.id, *user_roles_ids]))
            .where(ACL.permission.has(name=permission_type))
        )
        acls = result.unique().scalars().all()
        has_permission = dataset_id in [acl.dataset.id for acl in acls]

        if not has_permission:
            raise PermissionDeniedError(
                message=f"User {user.id} does not have {permission_type} permission on documents"
            )
