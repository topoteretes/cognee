from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine

async def get_default_user() -> User:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        return session.query(User).filter(User.email == "default_user@example.com").first()
