import json
import asyncio
from uuid import UUID
from fastapi.encoders import jsonable_encoder
from typing import Any, List, Optional, Tuple, Type, Union

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee.context_global_variables import set_database_global_context_variables
from cognee.context_global_variables import backend_access_control_enabled

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.types import (
    SearchResult,
    SearchType,
)
from cognee.modules.search.operations import log_query, log_result
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee import __version__ as cognee_version
from cognee.modules.search.methods.get_retriever_output import get_retriever_output

logger = get_logger()


async def search(
    query_text: str,
    query_type: SearchType,
    dataset_ids: Union[list[UUID], None],
    user: User,
    system_prompt_path="answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    verbose=False,
    retriever_specific_config: Optional[dict] = None,
) -> List[SearchResult]:
    """

    Args:
        query_text:
        query_type:
        datasets:
        user:
        system_prompt_path:
        top_k:

    Returns:

    Notes:
        Searching by dataset is only available in ENABLE_BACKEND_ACCESS_CONTROL mode
    """
    query = await log_query(query_text, query_type.value, user.id)
    send_telemetry(
        "cognee.search EXECUTION STARTED",
        user.id,
        additional_properties={
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
        },
    )

    search_results = await authorized_search(
        query_type=query_type,
        query_text=query_text,
        user=user,
        dataset_ids=dataset_ids,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        save_interaction=save_interaction,
        last_k=last_k,
        only_context=only_context,
        session_id=session_id,
        wide_search_top_k=wide_search_top_k,
        triplet_distance_penalty=triplet_distance_penalty,
        retriever_specific_config=retriever_specific_config,
    )

    send_telemetry(
        "cognee.search EXECUTION COMPLETED",
        user.id,
        additional_properties={
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
        },
    )

    await log_result(
        query.id,
        json.dumps(jsonable_encoder(search_results)),
        user.id,
    )

    return _backwards_compatible_search_results(search_results, verbose)


async def authorized_search(
    query_type: SearchType,
    query_text: str,
    user: User,
    dataset_ids: Optional[list[UUID]] = None,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    retriever_specific_config: Optional[dict] = None,
) -> List[Tuple[Any, Union[List[Edge], str], List[Dataset]]]:
    """
    Verifies access for provided datasets or uses all datasets user has read access for and performs search per dataset.
    Not to be used outside of active access control mode.
    """
    # Find datasets user has read access for (if datasets are provided only return them. Provided user has read access)
    search_datasets = await get_authorized_existing_datasets(
        datasets=dataset_ids, permission_type="read", user=user
    )

    # Searches all provided datasets and handles setting up of appropriate database context based on permissions
    search_results = await search_in_datasets_context(
        search_datasets=search_datasets,
        query_type=query_type,
        query_text=query_text,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        save_interaction=save_interaction,
        last_k=last_k,
        only_context=only_context,
        session_id=session_id,
        wide_search_top_k=wide_search_top_k,
        triplet_distance_penalty=triplet_distance_penalty,
        retriever_specific_config=retriever_specific_config,
    )

    return search_results


async def search_in_datasets_context(
    search_datasets: list[Dataset],
    query_type: SearchType,
    query_text: str,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    retriever_specific_config: Optional[dict] = None,
) -> List[Tuple[Any, Union[str, List[Edge]], List[Dataset]]]:
    """
    Searches all provided datasets and handles setting up of appropriate database context based on permissions.
    Not to be used outside of active access control mode.
    """

    async def _search_in_dataset_context(
        dataset: Dataset,
        query_type: SearchType,
        query_text: str,
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: int = 10,
        node_type: Optional[Type] = NodeSet,
        node_name: Optional[List[str]] = None,
        save_interaction: bool = False,
        last_k: Optional[int] = None,
        only_context: bool = False,
        session_id: Optional[str] = None,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        retriever_specific_config: Optional[dict] = None,
    ) -> Tuple[Any, Union[str, List[Edge]], List[Dataset]]:
        # Set database configuration in async context for each dataset user has access for
        await set_database_global_context_variables(dataset.id, dataset.owner_id)

        # Check if graph for dataset is empty and log warnings if necessary
        graph_engine = await get_graph_engine()
        is_empty = await graph_engine.is_empty()
        if is_empty:
            # TODO: we can log here, but not all search types use graph. Still keeping this here for reviewer input
            from cognee.modules.data.methods import get_dataset_data

            dataset_data = await get_dataset_data(dataset.id)

            if len(dataset_data) > 0:
                logger.warning(
                    f"Dataset '{dataset.name}' has {len(dataset_data)} data item(s) but the knowledge graph is empty. "
                    "Please run cognify to process the data before searching."
                )
            else:
                logger.warning(
                    f"Search attempt on an empty knowledge graph - no data has been added to this dataset: {dataset.name}"
                )

        # Get retriever output in the context of the current dataset
        return await get_retriever_output(
            query_type=query_type,
            query_text=query_text,
            dataset=dataset,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            only_context=only_context,
            session_id=session_id,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            retriever_specific_config=retriever_specific_config,
        )

    # Search every dataset async based on query and appropriate database configuration
    tasks = []
    for dataset in search_datasets:
        tasks.append(
            _search_in_dataset_context(
                dataset=dataset,
                query_type=query_type,
                query_text=query_text,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                last_k=last_k,
                only_context=only_context,
                session_id=session_id,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
                retriever_specific_config=retriever_specific_config,
            )
        )

    return await asyncio.gather(*tasks)


def _backwards_compatible_search_results(search_results, verbose: bool):
    """
    Prepares search results in a format compatible with previous versions of the API.
    """
    # This is for maintaining backwards compatibility
    if backend_access_control_enabled():
        return_value = []
        for search_result in search_results:
            # Dataset info needs to be always included
            search_result_dict = {
                "dataset_id": search_result.dataset_id,
                "dataset_name": search_result.dataset_name,
                "dataset_tenant_id": search_result.dataset_tenant_id,
            }
            if verbose:
                # Include all different types of results only in verbose mode
                search_result_dict["text_result"] = search_result.completion
                search_result_dict["context_result"] = search_result.context
                search_result_dict["objects_result"] = search_result.result_object
            else:
                # Result attribute handles returning appropriate result based on set flags and outputs
                search_result_dict["search_result"] = search_result.result

            return_value.append(search_result_dict)
        return return_value
    else:
        return_value = []
        if verbose:
            for search_result in search_results:
                # Include all different types of results only in verbose mode
                search_result_dict = {
                    "text_result": search_result.completion,
                    "context_result": search_result.context,
                    "objects_result": search_result.result_object,
                }
                return_value.append(search_result_dict)
            return return_value
        else:
            for search_result in search_results:
                # Result attribute handles returning appropriate result based on set flags and outputs
                return_value.append(search_result.result)

            # For maintaining backwards compatibility
            if len(return_value) == 1 and isinstance(return_value[0], list):
                # If a single element list return the element directly
                return return_value[0]
            else:
                # Otherwise return the list of results
                return return_value
