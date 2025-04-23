from types import SimpleNamespace
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select
from cognee.modules.users.models import User
from cognee.base_config import get_base_config
from cognee.modules.users.exceptions.exceptions import UserNotFoundError
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.create_default_user import create_default_user


async def get_default_user() -> SimpleNamespace:
    db_engine = get_relational_engine()
    base_config = get_base_config()
    default_email = base_config.default_user_email or "default_user@example.com"

    try:
        async with db_engine.get_async_session() as session:
            query = (
                select(User).options(selectinload(User.roles)).where(User.email == default_email)
            )

            result = await session.execute(query)
            user = result.scalars().first()

            if user is None:
                return await create_default_user()

            # We return a SimpleNamespace to have the same user type as our SaaS
            # SimpleNamespace is just a dictionary which can be accessed through attributes
            auth_data = SimpleNamespace(id=user.id, tenant_id=user.tenant_id, roles=[])
            return auth_data
    except Exception as error:
        if "principals" in str(error.args):
            raise DatabaseNotCreatedError() from error

        raise UserNotFoundError(f"Failed to retrieve default user: {default_email}") from error
