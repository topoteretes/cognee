from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ...models import ACL, Resource, Permission

async def get_document_ids_for_user(user_id: UUID) -> list[str]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        async with session.begin():
            document_ids = (await session.scalars(
                select(Resource.resource_id)
                .join(ACL.resources)
                .join(ACL.permission)
                .where(
                    ACL.principal_id == user_id,
                    Permission.name == "read",
                )
            )).all()

            return document_ids
