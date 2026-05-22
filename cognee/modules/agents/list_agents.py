from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import select, outerjoin

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.User import User
from cognee.modules.users.models.UserApiKey import UserApiKey


@dataclass
class AgentInfo:
    user: User
    api_key_label: Optional[str]


async def list_agents(owner_id: UUID) -> list[AgentInfo]:
    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        stmt = (
            select(User, UserApiKey.label)
            .select_from(outerjoin(User, UserApiKey, User.id == UserApiKey.user_id))
            .where(User.parent_user_id == owner_id)
        )

        results = (await session.execute(stmt)).all()
        return [AgentInfo(user=row[0], api_key_label=row[1]) for row in results]
