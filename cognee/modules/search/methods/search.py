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
   SearchResultDataset,
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
from .get_search_type_tools import get_search_type_tools
from .no_access_control_search import no_access_control_search
from ..utils.prepare_search_result import prepare_search_result
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps # Import your function

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
    use_combined_context: bool = False,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
    verbose: bool = False,
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

    actual_accessed_items = [] # Collect all accessed items here

    # Use search function filtered by permissions if access control is enabled
    if backend_access_control_enabled():
        raw_search_results = await authorized_search(
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
        )
        if use_combined_context:
            # raw_search_results is (completion, context, datasets)
            _, context_data, _ = raw_search_results
            if isinstance(context_data, list): # Expecting a list of Edge or similar
                actual_accessed_items.extend(context_data)
            # If context_data is a string, it's already textual and might not map to specific nodes for timestamp updates
        else:
            for result_tuple in raw_search_results:
                _, context_data, _ = result_tuple
                if isinstance(context_data, list): # Expecting a list of Edge or similar
                    actual_accessed_items.extend(context_data)
    else:
        raw_search_results = [
            await no_access_control_search(
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
            )
        ]
        # In this case, raw_search_results is a list containing a single tuple
        if raw_search_results:
            _, context_data, _ = raw_search_results[0]
            if isinstance(context_data, list): # Expecting a list of Edge or similar
                actual_accessed_items.extend(context_data)

    # Call the update_node_access_timestamps function here
    # Pass the collected actual_accessed_items
    await update_node_access_timestamps(actual_accessed_items)

    send_telemetry(
        "cognee.search EXECUTION COMPLETED",
        user.id,
        additional_properties={
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
        },
    )

    search_results = raw_search_results

    await log_result(
        query.id,
        json.dumps(
            jsonable_encoder(
                [await prepare_search_result(search_result) for search_result in search_results]
            )
        ),
        user.id,
    )

    if use_combined_context:
        # Note: combined context search must always be verbose and return a CombinedSearchResult with graphs info
        prepared_search_results = await prepare_search_result(
            search_results[0] if isinstance(search_results, list) else search_results
        )
        result = prepared_search_results["result"]
        graphs = prepared_search_results["graphs"]
        context = prepared_search_results["context"]
        datasets = prepared_search_results["datasets"]

        return CombinedSearchResult(
            result=result,
            graphs=graphs,
            context=context,
            datasets=[
                SearchResultDataset(
                    id=dataset.id,
                    name=dataset.name,
                )
                for dataset in datasets
            ],
        )
    else:
        return_value = []
        if only_context:
            for search_result in search_results:
                prepared_search_results = await prepare_search_result(search_result)

                result = prepared_search_results["result"]
                graphs = prepared_search_results["graphs"]
                context = prepared_search_results["context"]
                datasets = prepared_search_results["datasets"]

                if only_context:
                    search_result_dict = {
                        "search_result": [context] if context else None,
                        "dataset_id": datasets[0].id,
                        "dataset_name": datasets[0].name,
                        "dataset_tenant_id": datasets[0].tenant_id,
                    }
                    if verbose:
                        # Include graphs only in verbose mode
                        search_result_dict["graphs"] = graphs

                    return_value.append(search_result_dict)
                else:
                    search_result_dict = {
                        "search_result": [result] if result else None,
                        "dataset_id": datasets[0].id,
                        "dataset_name": datasets[0].name,
                        "dataset_tenant_id": datasets[0].tenant_id,
                    }
                    if verbose:
                        # Include graphs only in verbose mode
                        search_result_dict["graphs"] = graphs

                    return_value.append(search_result_dict)

            return return_value
        else:
            for search_result in search_results:
                result, context, datasets = search_result
                return_value.append(result)
        # For maintaining backwards compatibility
        if len(return_value) == 1 and isinstance(return_value[0], list):
            return return_value[0]
        else:
            return return_value


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
    context: Optional[Any] = None,
    session_id: Optional[str] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
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
        context: Optional[Any] = None,
        session_id: Optional[str] = None,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
    ) -> Tuple[Any, Union[str, List[Edge]], List[Dataset]]:
        # Set database configuration in async context for each dataset user has access for
        await set_database_global_context_variables(dataset.id, dataset.owner_id)

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
                    "Search attempt on an empty knowledge graph - no data has been added to this dataset"
                )

        specific_search_tools = await get_search_type_tools(
            query_type=query_type,
            query_text=query_text,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            last_k=last_k,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
        )
        search_tools = specific_search_tools
        if len(search_tools) == 2:
            [get_completion, get_context] = search_tools

            if only_context:
                return None, await get_context(query_text), [dataset]

            search_context = context or await get_context(query_text)
            search_result = await get_completion(query_text, search_context, session_id=session_id)

            return search_result, search_context, [dataset]
        else:
            unknown_tool = search_tools[0]

            return await unknown_tool(query_text), "", [dataset]

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
                context=context,
                session_id=session_id,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            )
        )

    return await asyncio.gather(*tasks)
