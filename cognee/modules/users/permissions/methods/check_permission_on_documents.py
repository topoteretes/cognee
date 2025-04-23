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


async def check_permission_on_documents(user: User, permission_type: str, document_ids: list[UUID]):
    if user is None:
        user = await get_default_user()

    # TODO: Enable user role permissions again. Temporarily disabled during rework.
    # user_roles_ids = [role.id for role in user.roles]
    user_roles_ids = []

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(ACL)
            .join(ACL.permission)
            .options(joinedload(ACL.data))
            .where(ACL.principal_id.in_([user.id, *user_roles_ids]))
            .where(ACL.permission.has(name=permission_type))
        )
        acls = result.unique().scalars().all()
        data_ids = [acl.data.id for acl in acls]
        has_permissions = all(document_id in data_ids for document_id in document_ids)

        if not has_permissions:
            raise PermissionDeniedError(
                message=f"User {user.id} does not have {permission_type} permission on documents"
            )
