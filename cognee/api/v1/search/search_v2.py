from typing import Union

from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import UserNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.search.methods import search as search_function


async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: User = None,
    datasets: Union[list[str], str, None] = None,
) -> list:
    # We use lists from now on for datasets
    if isinstance(datasets, str):
        datasets = [datasets]

    if user is None:
        user = await get_default_user()

    if user is None:
        raise UserNotFoundError

    filtered_search_results = await search_function(query_text, query_type, datasets, user)

    return filtered_search_results
