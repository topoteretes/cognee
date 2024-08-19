from sqlalchemy.future import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ...models import User, ACL, Resource, Permission

async def give_permission_on_document(
    user: User,
    document_id: str,
    permission_name: str,
):
    db_engine = get_relational_engine()

    document_resource = Resource(resource_id = document_id)

    async with db_engine.get_async_session() as session:
        permission = (await session.execute(select(Permission).filter(Permission.name == permission_name))).scalars().first()

        if permission is None:
            permission = Permission(name = permission_name)

        acl = ACL(principal_id = user.id)
        acl.permission = permission
        acl.resources.append(document_resource)

        session.add(acl)

        await session.commit()
