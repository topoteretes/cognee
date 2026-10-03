from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.only_context_prompt import build_only_context_prompt
from cognee.modules.retrieval.session_aware_completion import run_session_aware_completion
from cognee.modules.retrieval.utils.evidence import (
    append_source_evidence_text,
    graph_source_evidence,
)
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.methods.hybrid_deferral import (
    hybrid_deferral_reason,
    reject_hybrid_graph_only_knobs,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.operations.select_search_type import select_search_type
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def _effective_search_type(
    query_type: SearchType, query_text: str, kwargs: dict, graph_is_empty: bool
) -> SearchType:
    """Resolve FEELING_LUCKY and hybrid deferral to the retriever type that will actually run."""
    if query_type is SearchType.FEELING_LUCKY:
        if graph_is_empty:
            resolved = SearchType.HYBRID_COMPLETION
        else:
            resolved = await select_search_type(query_text)
    else:
        resolved = query_type

    if resolved is SearchType.HYBRID_COMPLETION:
        reject_hybrid_graph_only_knobs(kwargs)
        reason = await hybrid_deferral_reason(kwargs, graph_is_empty=graph_is_empty)
        if reason:
            logger.info("Deferring HYBRID_COMPLETION to GRAPH_COMPLETION: %s", reason)
            # Payload.search_type records this. search() strips it; only_context
            # and verbose callers that parse shape must pin query_type.
            return SearchType.GRAPH_COMPLETION
    return resolved


def _dataset_fields(kwargs: dict) -> dict:
    """The dataset identity every SearchResultPayload carries, absent when unscoped."""
    dataset = kwargs.get("dataset")
    return {
        "dataset_name": dataset.name if dataset else None,
        "dataset_id": dataset.id if dataset else None,
        "dataset_tenant_id": dataset.tenant_id if dataset else None,
    }


def _context_evidence(retriever_instance, retrieved_objects, kwargs: dict) -> list:
    """Structured evidence for the artifacts the retriever placed into context."""
    evidence_method = getattr(retriever_instance, "get_context_evidence", None)
    if not callable(evidence_method):
        return []
    dataset = kwargs.get("dataset")
    try:
        return evidence_method(retrieved_objects, dataset_id=getattr(dataset, "id", None))
    except Exception as error:
        logger.warning("Unable to build structured context evidence: %s", error, exc_info=True)
        return []


async def get_retriever_output(
    query_type: SearchType, query_text: str, **kwargs
) -> SearchResultPayload:
    graph_engine = await get_graph_engine()
    graph_is_empty = await graph_engine.is_empty()
    if graph_is_empty:
        logger.warning("Search attempt on an empty knowledge graph")
        return SearchResultPayload(
            result_object=[],
            context=None,
            completion=[],
            search_type=query_type,
            only_context=kwargs.get("only_context", False),
            dataset_name=kwargs.get("dataset").name if kwargs.get("dataset") else None,
            dataset_id=kwargs.get("dataset").id if kwargs.get("dataset") else None,
            dataset_tenant_id=kwargs.get("dataset").tenant_id if kwargs.get("dataset") else None,
        )

    effective_query_type = await _effective_search_type(
        query_type, query_text, kwargs, graph_is_empty
    )

    retriever_instance = await get_search_type_retriever_instance(
        query_type=effective_query_type, query_text=query_text, **kwargs
    )

    only_context = kwargs.get("only_context", False)
    retrieved_objects, context, completion = await run_session_aware_completion(
        retriever_instance,
        raw_query=query_text,
        original_search_type=query_type,
        only_context=only_context,
        search_type_for_spans=effective_query_type,
    )

    prompts = None
    if only_context:
        # The (user_prompt, system_prompt) pair the completion would have received, or
        # None when the retriever sends no single templated prompt (bare context is
        # returned then). The caller's session_id is passed explicitly: non-generative
        # retrievers do not keep one, and the system prompt must describe the session
        # that was asked about. shared_history is the fan-out's single
        # conversation-history read.
        prompts = await build_only_context_prompt(
            retriever_instance,
            query=query_text,
            context=context,
            session_id=kwargs.get("session_id"),
            shared_history=kwargs.get("shared_history"),
        )

    evidence = []
    if kwargs.get("include_references", False):
        evidence = _context_evidence(retriever_instance, retrieved_objects, kwargs)
        dataset = kwargs.get("dataset")
        if dataset is not None and any(reference.kind == "graph_edge" for reference in evidence):
            # Indexed relational sidecar lookup; cheap next to the LLM
            # completion that already ran inside the session-aware door.
            try:
                evidence.extend(await graph_source_evidence(evidence, getattr(dataset, "id", None)))
                completion = append_source_evidence_text(completion, evidence)
            except Exception as error:
                logger.warning("Unable to resolve graph source evidence: %s", error, exc_info=True)

    return SearchResultPayload(
        result_object=retrieved_objects,
        context=context,
        completion=completion,
        evidence=evidence,
        search_type=effective_query_type,
        only_context=only_context,
        user_prompt=prompts[0] if prompts else None,
        system_prompt=prompts[1] if prompts else None,
        **_dataset_fields(kwargs),
    )
