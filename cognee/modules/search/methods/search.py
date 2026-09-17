import asyncio
from typing import Any
from uuid import UUID

from cognee import __version__ as cognee_version
from cognee.base_config import get_base_config
from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)
from cognee.modules.data.models import Dataset
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.observability import (
    COGNEE_SEARCH_QUERY,
    COGNEE_SEARCH_TYPE,
    new_span,
)
from cognee.modules.retrieval.context_preview import SharedSessionHistory
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.operations import log_search_history
from cognee.modules.search.types import (
    ContextFormat,
    SearchResult,
    SearchType,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()


def _single_dataset_id(dataset_ids: list[UUID] | UUID | None) -> UUID | None:
    """Return the dataset a search is scoped to, when it is exactly one.

    Searches fan out across every dataset they are given, and ``None`` means
    "every dataset the user can read". Neither case has a single dataset to
    attribute the logged query to, so both record ``None``.
    """
    if dataset_ids is None:
        return None
    if isinstance(dataset_ids, UUID):
        return dataset_ids
    return dataset_ids[0] if len(dataset_ids) == 1 else None


async def search(
    query_text: str,
    query_type: SearchType,
    dataset_ids: list[UUID] | None,
    user: User,
    system_prompt_path="answer_simple_question.txt",
    system_prompt: str | None = None,
    top_k: int = 15,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    context_format: ContextFormat | str = ContextFormat.CONTEXT,
    session_id: str | None = None,
    wide_search_top_k: int | None = None,
    triplet_distance_penalty: float | None = None,
    feedback_influence: float = get_base_config().default_feedback_influence,
    verbose=False,
    retriever_specific_config: dict | None = None,
    neighborhood_depth: int | None = None,
    neighborhood_seed_top_k: int | None = None,
    include_references: bool = False,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[SearchResult]:
    """Run one search type over the datasets a user may read and return per-dataset results.

    This is the internal entry point behind ``cognee.search()`` (which resolves
    dataset names to ids and validates argument combinations first) and, through
    it, ``cognee.recall()``. It:

    1. resolves ``dataset_ids`` to the datasets ``user`` has ``read`` permission on
       (``None`` means every readable dataset);
    2. fans the query out to each dataset concurrently, each under that dataset's
       database context when ``ENABLE_BACKEND_ACCESS_CONTROL`` is on (with it off,
       all datasets share one context and the search runs once);
    3. picks the retriever for ``query_type`` (see
       ``get_search_type_retriever_instance``), resolving ``FEELING_LUCKY`` to a
       concrete type and letting ``HYBRID_COMPLETION`` defer to
       ``GRAPH_COMPLETION`` when the request needs graph-only features or the
       chunk collection is missing;
    4. logs the query and completion text to search history (never the raw
       retrieved objects).

    Args:
        query_text: The user's question or search string.
        query_type: Which retriever to run. ``FEELING_LUCKY`` selects one via an
            LLM; ``HYBRID_COMPLETION`` may be deferred as described above.
        dataset_ids: Datasets to search, or ``None`` for all readable datasets.
        user: The requesting user; drives permission filtering and history.
        system_prompt_path: Prompt template file for completion-style types.
        system_prompt: Inline system prompt; overrides ``system_prompt_path``.
        top_k: Maximum retrieved objects per dataset (default 15). Hybrid caps
            its per-lane ``top_k`` unless overridden via ``retriever_specific_config``.
        node_type: ``DataPoint`` subclass used to filter nodes (default ``NodeSet``).
            A custom type forces hybrid to defer to graph completion.
        node_name: Restrict retrieval to these node names (e.g. node-set tags),
            combined with ``node_name_filter_operator`` (``"OR"``/``"AND"``).
        only_context: Return the retrieval context instead of an LLM answer.
        context_format: With ``only_context``, ``"context"`` (bare string, default)
            or ``"prompt"`` (question, context, session_context, user/system prompt).
        session_id: Session whose history is added to the completion context and
            that receives the QA entry. Does not search the session cache; that is
            ``recall()``-only.
        wide_search_top_k, triplet_distance_penalty: Graph-completion tuning knobs;
            rejected when the type is hybrid.
        feedback_influence: Weight of learned feedback in graph ranking (default
            from ``BaseConfig.default_feedback_influence``).
        verbose: Return the raw per-dataset payload shape instead of the
            backwards-compatible result list.
        retriever_specific_config: Extra constructor kwargs for the chosen
            retriever (e.g. ``response_model``, hybrid lane ``top_k`` values,
            ``skills``/``tools``/``max_iter`` for ``AGENTIC_COMPLETION``,
            ``code_query`` for ``CODE``).
        neighborhood_depth, neighborhood_seed_top_k: Graph neighbourhood
            expansion controls; validated as positive integers by the caller.
        include_references: Attach structured ``EvidenceReference`` objects
            (edge evidence, source chunks) to completions that support them.
        llm_config, embedding_config: Per-call provider overrides applied inside
            each dataset context.

    Returns:
        One ``SearchResult`` per dataset that produced output. ``search_result``
        holds the retriever's completion (a string or dict for completion types,
        the context list for retrieval-only types, or a ``{"seed_not_found":
        True, ...}`` marker for a per-dataset ``CODE`` seed miss);
        ``dataset_id``/``dataset_name`` identify the dataset. With
        ``verbose=True`` the list carries the full ``SearchResultPayload`` shape.

    Notes:
        Scoping to specific datasets requires ``ENABLE_BACKEND_ACCESS_CONTROL``
        (the default). Permission failures yield an empty list, not an error.
    """
    send_telemetry(
        "cognee.search EXECUTION STARTED",
        user,
        additional_properties={
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
            "include_references": include_references,
        },
    )

    with new_span("cognee.search.authorize") as span:
        span.set_attribute(COGNEE_SEARCH_TYPE, query_type.value)
        span.set_attribute(COGNEE_SEARCH_QUERY, query_text[:500])
        span.set_attribute("cognee.search.top_k", top_k)
        span.set_attribute(
            "cognee.search.dataset_count",
            len(dataset_ids) if dataset_ids else 0,
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
            node_name_filter_operator=node_name_filter_operator,
            only_context=only_context,
            context_format=context_format,
            session_id=session_id,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            feedback_influence=feedback_influence,
            retriever_specific_config=retriever_specific_config,
            neighborhood_depth=neighborhood_depth,
            neighborhood_seed_top_k=neighborhood_seed_top_k,
            include_references=include_references,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )

        span.set_attribute("cognee.search.result_count", len(search_results))

    send_telemetry(
        "cognee.search EXECUTION COMPLETED",
        user,
        additional_properties={
            "cognee_version": cognee_version,
            "tenant_id": str(user.tenant_id) if user.tenant_id else "Single User Tenant",
        },
    )

    # Logged after the search because only the results say which datasets were
    # actually searched — dataset_ids=None means "every dataset the user can
    # read". Only the completion text is stored, never the raw result_objects,
    # which run 50-100 KB each and would grow the DB without bound.
    await log_search_history(query_text, query_type.value, user.id, search_results)

    return _backwards_compatible_search_results(search_results, verbose)


async def authorized_search(
    query_type: SearchType,
    query_text: str,
    user: User,
    dataset_ids: list[UUID] | None = None,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: str | None = None,
    top_k: int = 15,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    context_format: ContextFormat | str = ContextFormat.CONTEXT,
    session_id: str | None = None,
    wide_search_top_k: int | None = None,
    triplet_distance_penalty: float | None = None,
    feedback_influence: float = get_base_config().default_feedback_influence,
    retriever_specific_config: dict | None = None,
    neighborhood_depth: int | None = None,
    neighborhood_seed_top_k: int | None = None,
    include_references: bool = False,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[SearchResultPayload]:
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
        user=user,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        node_name_filter_operator=node_name_filter_operator,
        only_context=only_context,
        context_format=context_format,
        session_id=session_id,
        wide_search_top_k=wide_search_top_k,
        triplet_distance_penalty=triplet_distance_penalty,
        feedback_influence=feedback_influence,
        retriever_specific_config=retriever_specific_config,
        neighborhood_depth=neighborhood_depth,
        neighborhood_seed_top_k=neighborhood_seed_top_k,
        include_references=include_references,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )

    return search_results


async def search_in_datasets_context(
    search_datasets: list[Dataset],
    query_type: SearchType,
    query_text: str,
    user: User,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: str | None = None,
    top_k: int = 15,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    node_name_filter_operator: str = "OR",
    only_context: bool = False,
    context_format: ContextFormat | str = ContextFormat.CONTEXT,
    session_id: str | None = None,
    wide_search_top_k: int | None = None,
    triplet_distance_penalty: float | None = None,
    feedback_influence: float = get_base_config().default_feedback_influence,
    retriever_specific_config: dict | None = None,
    neighborhood_depth: int | None = None,
    neighborhood_seed_top_k: int | None = None,
    include_references: bool = False,
    llm_config: LLMConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[tuple[Any, str | list[Edge], list[Dataset]]]:
    """
    Searches all provided datasets and handles setting up of appropriate database context based on permissions.
    Not to be used outside of active access control mode.
    """

    async def _search_in_dataset_context(
        dataset: Dataset,
        query_type: SearchType,
        query_text: str,
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: str | None = None,
        top_k: int = 15,
        node_type: type | None = NodeSet,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
        only_context: bool = False,
        context_format: ContextFormat | str = ContextFormat.CONTEXT,
        session_id: str | None = None,
        wide_search_top_k: int | None = None,
        triplet_distance_penalty: float | None = None,
        feedback_influence: float = get_base_config().default_feedback_influence,
        retriever_specific_config: dict | None = None,
        neighborhood_depth: int | None = None,
        neighborhood_seed_top_k: int | None = None,
        include_references: bool = False,
    ) -> SearchResultPayload:
        with new_span("cognee.search.dataset") as span:
            span.set_attribute("cognee.search.dataset_name", dataset.name or "")
            span.set_attribute("cognee.search.dataset_id", str(dataset.id))

            async with set_database_global_context_variables(
                dataset.id,
                dataset.owner_id,
                llm_config=llm_config,
                embedding_config=embedding_config,
            ):
                # Check if graph for dataset is empty and log warnings if necessary
                graph_engine = await get_graph_engine()
                is_empty = await graph_engine.is_empty()
                data_item_count = None
                if is_empty:
                    # TODO: we can log here, but not all search types use graph. Still keeping this here for reviewer input
                    from cognee.modules.data.methods import get_dataset_data

                    dataset_data = await get_dataset_data(dataset.id)
                    data_item_count = len(dataset_data)

                    if len(dataset_data) > 0:
                        logger.warning(
                            f"Dataset '{dataset.name}' has {len(dataset_data)} data item(s) but the knowledge graph is empty. "
                            "Please run cognify to process the data before searching."
                        )
                    else:
                        logger.warning(
                            f"Search attempt on an empty knowledge graph - no data has been added to this dataset: {dataset.name}"
                        )

                    span.set_attribute("cognee.search.graph_empty", True)

                # Get retriever output in the context of the current dataset
                try:
                    return await get_retriever_output(
                        query_type=query_type,
                        query_text=query_text,
                        user=user,
                        dataset=dataset,
                        system_prompt_path=system_prompt_path,
                        system_prompt=system_prompt,
                        top_k=top_k,
                        node_type=node_type,
                        node_name=node_name,
                        node_name_filter_operator=node_name_filter_operator,
                        only_context=only_context,
                        context_format=context_format,
                        shared_history=shared_history,
                        session_id=session_id,
                        wide_search_top_k=wide_search_top_k,
                        triplet_distance_penalty=triplet_distance_penalty,
                        feedback_influence=feedback_influence,
                        retriever_specific_config=retriever_specific_config,
                        neighborhood_depth=neighborhood_depth,
                        neighborhood_seed_top_k=neighborhood_seed_top_k,
                        include_references=include_references,
                    )
                except NoDataError as error:
                    # The retriever knows its graph (or collection) is empty but not
                    # which dataset it was searching. This frame knows both, so tag
                    # the error with the dataset and what it is missing; the fan-out
                    # below decides whether that fails the whole search.
                    raise DatasetNoDataError(
                        dataset, error, graph_is_empty=is_empty, data_item_count=data_item_count
                    ) from error

    async def _report_code_seed_miss(dataset_search, dataset: Dataset) -> SearchResultPayload:
        """Report a per-dataset CODE seed miss instead of failing the whole request.

        A name/id seed that one dataset cannot resolve says nothing about the
        other datasets being searched, so surface it as that dataset's result.
        """
        from cognee.modules.retrieval.code_retriever import CodeSeedNotFoundError

        try:
            return await dataset_search
        except CodeSeedNotFoundError as error:
            # ``error`` is the one place a caller reads a dataset's failure; the
            # completion marker is the shape this case shipped with and stays
            # for callers that already parse it.
            return SearchResultPayload(
                result_object=None,
                context=None,
                completion={"seed_not_found": True, "error": str(error)},
                error=str(error),
                search_type=query_type,
                only_context=False,
                dataset_name=dataset.name,
                dataset_id=dataset.id,
                dataset_tenant_id=dataset.tenant_id,
            )

    # One conversation-history read — the only billed step of the session layer (it
    # embeds the query for vector recall) — for the whole fan-out. The guidance block
    # still renders per dataset because preferences are dataset-scoped.
    shared_history = None
    if only_context and ContextFormat.parse(context_format) is ContextFormat.PROMPT:
        shared_history = SharedSessionHistory(query=query_text, session_id=session_id)

    # Search every dataset async based on query and appropriate database configuration
    tasks = []
    soften_code_seed_misses = query_type is SearchType.CODE and len(search_datasets) > 1
    if backend_access_control_enabled():
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
                    node_name_filter_operator=node_name_filter_operator,
                    only_context=only_context,
                    context_format=context_format,
                    session_id=session_id,
                    wide_search_top_k=wide_search_top_k,
                    triplet_distance_penalty=triplet_distance_penalty,
                    feedback_influence=feedback_influence,
                    retriever_specific_config=retriever_specific_config,
                    neighborhood_depth=neighborhood_depth,
                    neighborhood_seed_top_k=neighborhood_seed_top_k,
                    include_references=include_references,
                )
            )
            if soften_code_seed_misses:
                tasks[-1] = _report_code_seed_miss(tasks[-1], dataset)
    else:
        # Run search without setting database context in case access control is disabled
        # Needed for low level pipelines that need to run search without dataset context.
        dataset = search_datasets[0] if len(search_datasets) == 1 else None
        retriever_kwargs = {
            "query_type": query_type,
            "query_text": query_text,
            "user": user,
            "dataset": dataset,
            "system_prompt_path": system_prompt_path,
            "system_prompt": system_prompt,
            "top_k": top_k,
            "node_type": node_type,
            "node_name": node_name,
            "node_name_filter_operator": node_name_filter_operator,
            "only_context": only_context,
            "context_format": context_format,
            "shared_history": shared_history,
            "session_id": session_id,
            "wide_search_top_k": wide_search_top_k,
            "triplet_distance_penalty": triplet_distance_penalty,
            "feedback_influence": feedback_influence,
            "retriever_specific_config": retriever_specific_config,
            "neighborhood_depth": neighborhood_depth,
            "neighborhood_seed_top_k": neighborhood_seed_top_k,
            "include_references": include_references,
        }

        async def _search_without_context() -> SearchResultPayload:
            # No dataset DB context to set when access control is disabled, but
            # still forward any per-call LLM/embedding overrides onto the async
            # context (set_database_global_context_variables applies these even
            # in single-tenant mode and ignores the dataset argument).
            async with set_database_global_context_variables(
                dataset.id if dataset else None,
                user.id,
                llm_config=llm_config,
                embedding_config=embedding_config,
            ):
                try:
                    return await get_retriever_output(**retriever_kwargs)
                except NoDataError as error:
                    if dataset is None:
                        # Shared single-tenant graph, no dataset to name.
                        raise
                    raise DatasetNoDataError(dataset, error) from error

        tasks.append(_search_without_context())

    return _collect_dataset_results(
        await asyncio.gather(*tasks, return_exceptions=True),
        query_type=query_type,
        query_text=query_text,
        only_context=only_context,
        context_format=context_format,
    )


def _prompt_preview_fields(search_result) -> dict:
    """Prompt-preview keys for verbose output, present exactly when the prompt shape was asked for.

    Gated on ``context_format``, not on whether the values happen to be set: a verbose
    caller that requested the prompt always gets the three keys (possibly ``None``), and an
    ordinary search never sees them — the key set depends on the request, not on session
    state.
    """
    if search_result.context_format != ContextFormat.PROMPT:
        return {}
    return {
        "session_context_result": search_result.session_context,
        "user_prompt_result": search_result.user_prompt,
        "system_prompt_result": search_result.system_prompt,
    }


class DatasetNoDataError(NoDataError):
    """A retriever's NoDataError, tagged with the dataset it was searching.

    Raised only inside the per-dataset fan-out and consumed by
    ``_collect_dataset_results``; it never leaves ``search_in_datasets_context``.
    """

    def __init__(
        self,
        dataset: Dataset,
        error: NoDataError,
        *,
        graph_is_empty: bool = False,
        data_item_count: int | None = None,
    ):
        self.dataset = dataset
        self.reason = _no_data_reason(error, graph_is_empty, data_item_count)
        super().__init__(
            message=f"Dataset '{dataset.name}' (id: {dataset.id}): {self.reason}",
            status_code=error.status_code,
        )


def _no_data_reason(error: NoDataError, graph_is_empty: bool, data_item_count: int | None) -> str:
    """What a dataset is missing, phrased as the fix the caller has to apply."""
    if graph_is_empty and data_item_count:
        return (
            f"holds {data_item_count} data item(s) but its knowledge graph is empty; "
            "run cognify on this dataset before searching."
        )
    if graph_is_empty:
        return "no data has been added; add data and run cognify before searching."
    # A populated graph whose retriever still found nothing to search, e.g. RAG's
    # missing chunk collection: keep the retriever's own reason.
    return error.message


def _collect_dataset_results(
    outcomes: list,
    *,
    query_type: SearchType,
    query_text: str,
    only_context: bool,
    context_format: str | ContextFormat,
) -> list[SearchResultPayload]:
    """Turn the fan-out's per-dataset outcomes into one entry per dataset, or one error.

    A dataset without searchable memory fails the search only when *every* searched
    dataset is in that state; then one NoDataError (404) names each of them and what
    it is missing. Otherwise the list keeps one entry per dataset, in request order:
    a dataset that could not be searched comes back with empty results and its reason
    on ``error``, so the caller is told what happened without losing the siblings'
    answers (``datasets=None`` means "every dataset the user can read", so a single
    freshly created dataset must not take down unscoped search). Any other exception
    propagates unchanged and fails the whole search.
    """
    for outcome in outcomes:
        if isinstance(outcome, BaseException) and not isinstance(outcome, DatasetNoDataError):
            raise outcome

    no_data = [outcome for outcome in outcomes if isinstance(outcome, DatasetNoDataError)]
    if no_data and len(no_data) == len(outcomes):
        if len(no_data) == 1:
            only = no_data[0]
            raise NoDataError(
                message=(
                    f"No searchable memory in dataset '{only.dataset.name}' "
                    f"(id: {only.dataset.id}): {only.reason}"
                ),
                status_code=only.status_code,
            )
        lines = "\n".join(
            f"- '{error.dataset.name}' (id: {error.dataset.id}): {error.reason}"
            for error in no_data
        )
        raise NoDataError(
            message=(
                f"No searchable memory in any of the {len(no_data)} searched datasets:\n{lines}"
            ),
            status_code=no_data[0].status_code,
        )

    payloads: list[SearchResultPayload] = []
    for outcome in outcomes:
        if isinstance(outcome, DatasetNoDataError):
            logger.warning("Dataset without searchable memory: %s", outcome.message)
            payloads.append(
                _no_data_payload(
                    outcome,
                    query_type=query_type,
                    query_text=query_text,
                    only_context=only_context,
                    context_format=context_format,
                )
            )
        else:
            payloads.append(outcome)
    return payloads


def _no_data_payload(
    error: DatasetNoDataError,
    *,
    query_type: SearchType,
    query_text: str,
    only_context: bool,
    context_format: str | ContextFormat,
) -> SearchResultPayload:
    """The entry a dataset without searchable memory gets: empty results plus the reason.

    Shaped like a query miss for the same request (``[]`` under ``search_result``,
    an empty context for ``only_context``) so existing callers see what they always
    saw; only ``error`` is new.
    """
    return SearchResultPayload(
        result_object=[],
        context=[] if only_context else None,
        search_type=query_type,
        only_context=only_context,
        question=query_text,
        context_format=ContextFormat.parse(context_format),
        dataset_name=error.dataset.name,
        dataset_id=error.dataset.id,
        dataset_tenant_id=error.dataset.tenant_id,
        error=error.reason,
    )


def _backwards_compatible_search_results(search_results, verbose: bool):
    """
    Prepares search results in a format compatible with previous versions of the API.

    Does not include SearchResultPayload.search_type. only_context / verbose
    callers that parse retriever-specific shapes must pin query_type; hybrid
    may have deferred to GRAPH_COMPLETION.
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
                search_result_dict.update(_prompt_preview_fields(search_result))
                search_result_dict["evidence"] = [
                    reference.model_dump(mode="json") for reference in search_result.evidence
                ]
            else:
                # Result attribute handles returning appropriate result based on set flags and outputs
                search_result_dict["search_result"] = search_result.result

            if search_result.error is not None:
                # Only on entries that carry one: the dict stays byte-identical for
                # every dataset that was searched normally.
                search_result_dict["error"] = search_result.error

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
                    **_prompt_preview_fields(search_result),
                    "evidence": [
                        reference.model_dump(mode="json") for reference in search_result.evidence
                    ],
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
