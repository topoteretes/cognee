from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User


async def get_search_history(user: User = None) -> list:
    if not user:
        user = await get_default_user()

    return await get_history(user.id)
