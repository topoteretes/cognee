from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ...models import ACL

async def get_document_ids_for_user(user_id: UUID) -> list[str]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        async with session.begin():
            result = await session.execute(
                select(ACL.resources.resource_id) \
                    .join(ACL.resources) \
                    .filter_by(
                        ACL.principal_id == user_id,
                        ACL.permission.name == "read",
                    )
            )
            document_ids = [row[0] for row in result.scalars().all()]
            return document_ids




