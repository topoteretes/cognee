from uuid import UUID
from typing import Union, Optional, List, Type

from cognee.modules.users.models import User
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.modules.search.methods import search as search_function
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.exceptions import DatasetNotFoundError


async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: User = None,
    datasets: Optional[Union[list[str], str]] = None,
    dataset_ids: Optional[Union[list[UUID], UUID]] = None,
    system_prompt_path: str = "answer_simple_question.txt",
    top_k: int = 10,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> list:
    # We use lists from now on for datasets
    if isinstance(datasets, UUID) or isinstance(datasets, str):
        datasets = [datasets]

    if user is None:
        user = await get_default_user()

    # Transform string based datasets to UUID - String based datasets can only be found for current user
    if datasets is not None and [all(isinstance(dataset, str) for dataset in datasets)]:
        datasets = await get_authorized_existing_datasets(datasets, "read", user)
        datasets = [dataset.id for dataset in datasets]
        if not datasets:
            raise DatasetNotFoundError(message="No datasets found.")

    filtered_search_results = await search_function(
        query_text=query_text,
        query_type=query_type,
        dataset_ids=dataset_ids if dataset_ids else datasets,
        user=user,
        system_prompt_path=system_prompt_path,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
    )

    return filtered_search_results
