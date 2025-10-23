from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid5, NAMESPACE_OID

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine

from .models import FeedbackEnrichment


logger = get_logger("extract_feedback_interactions")


def _filter_negative_feedback(feedback_nodes):
    """Filter for negative sentiment feedback using precise sentiment classification."""
    return [
        (node_id, props)
        for node_id, props in feedback_nodes
        if (props.get("sentiment", "").casefold() == "negative" or props.get("score", 0) < 0)
    ]


def _get_normalized_id(node_id, props) -> str:
    """Return Cognee node id preference: props.id → props.node_id → raw node_id."""
    return str(props.get("id") or props.get("node_id") or node_id)


async def _fetch_feedback_and_interaction_graph_data() -> Tuple[List, List]:
    """Fetch feedback and interaction nodes with edges from graph engine."""
    try:
        graph_engine = await get_graph_engine()
        attribute_filters = [{"type": ["CogneeUserFeedback", "CogneeUserInteraction"]}]
        return await graph_engine.get_filtered_graph_data(attribute_filters)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch filtered graph data", error=str(exc))
        return [], []


def _separate_feedback_and_interaction_nodes(graph_nodes: List) -> Tuple[List, List]:
    """Split nodes into feedback and interaction groups by type field."""
    feedback_nodes = [
        (_get_normalized_id(node_id, props), props)
        for node_id, props in graph_nodes
        if props.get("type") == "CogneeUserFeedback"
    ]
    interaction_nodes = [
        (_get_normalized_id(node_id, props), props)
        for node_id, props in graph_nodes
        if props.get("type") == "CogneeUserInteraction"
    ]
    return feedback_nodes, interaction_nodes


def _match_feedback_nodes_to_interactions_by_edges(
    feedback_nodes: List, interaction_nodes: List, graph_edges: List
) -> List[Tuple[Tuple, Tuple]]:
    """Match feedback to interactions using gives_feedback_to edges."""
    interaction_by_id = {node_id: (node_id, props) for node_id, props in interaction_nodes}
    feedback_by_id = {node_id: (node_id, props) for node_id, props in feedback_nodes}
    feedback_edges = [
        (source_id, target_id)
        for source_id, target_id, rel, _ in graph_edges
        if rel == "gives_feedback_to"
    ]

    feedback_interaction_pairs: List[Tuple[Tuple, Tuple]] = []
    for source_id, target_id in feedback_edges:
        source_id_str, target_id_str = str(source_id), str(target_id)

        feedback_node = feedback_by_id.get(source_id_str)
        interaction_node = interaction_by_id.get(target_id_str)

        if feedback_node and interaction_node:
            feedback_interaction_pairs.append((feedback_node, interaction_node))

    return feedback_interaction_pairs


def _sort_pairs_by_recency_and_limit(
    feedback_interaction_pairs: List[Tuple[Tuple, Tuple]], last_n_limit: Optional[int]
) -> List[Tuple[Tuple, Tuple]]:
    """Sort by interaction created_at desc with updated_at fallback, then limit."""

    def _recency_key(pair):
        _, (_, interaction_props) = pair
        created_at = interaction_props.get("created_at") or ""
        updated_at = interaction_props.get("updated_at") or ""
        return (created_at, updated_at)

    sorted_pairs = sorted(feedback_interaction_pairs, key=_recency_key, reverse=True)
    return sorted_pairs[: last_n_limit or len(sorted_pairs)]


async def _generate_human_readable_context_summary(
    question_text: str, raw_context_text: str
) -> str:
    """Generate a concise human-readable summary for given context."""
    try:
        prompt = read_query_prompt("feedback_user_context_prompt.txt")
        rendered = prompt.format(question=question_text, context=raw_context_text)
        return await LLMGateway.acreate_structured_output(
            text_input=rendered, system_prompt="", response_model=str
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to summarize context", error=str(exc))
        return raw_context_text or ""


def _has_required_feedback_fields(enrichment: FeedbackEnrichment) -> bool:
    """Validate required fields exist in the FeedbackEnrichment DataPoint."""
    return (
        enrichment.question is not None
        and enrichment.original_answer is not None
        and enrichment.context is not None
        and enrichment.feedback_text is not None
        and enrichment.feedback_id is not None
        and enrichment.interaction_id is not None
    )


async def _build_feedback_interaction_record(
    feedback_node_id: str, feedback_props: Dict, interaction_node_id: str, interaction_props: Dict
) -> Optional[FeedbackEnrichment]:
    """Build a single FeedbackEnrichment DataPoint with context summary."""
    try:
        question_text = interaction_props.get("question")
        original_answer_text = interaction_props.get("answer")
        raw_context_text = interaction_props.get("context", "")
        feedback_text = feedback_props.get("feedback") or feedback_props.get("text") or ""

        context_summary_text = await _generate_human_readable_context_summary(
            question_text or "", raw_context_text
        )

        enrichment = FeedbackEnrichment(
            id=str(uuid5(NAMESPACE_OID, f"{question_text}_{interaction_node_id}")),
            text="",
            question=question_text,
            original_answer=original_answer_text,
            improved_answer="",
            feedback_id=UUID(str(feedback_node_id)),
            interaction_id=UUID(str(interaction_node_id)),
            belongs_to_set=None,
            context=context_summary_text,
            feedback_text=feedback_text,
            new_context="",
            explanation="",
        )

        if _has_required_feedback_fields(enrichment):
            return enrichment
        else:
            logger.warning("Skipping invalid feedback item", interaction=str(interaction_node_id))
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to process feedback pair", error=str(exc))
        return None


async def _build_feedback_interaction_records(
    matched_feedback_interaction_pairs: List[Tuple[Tuple, Tuple]],
) -> List[FeedbackEnrichment]:
    """Build all FeedbackEnrichment DataPoints from matched pairs."""
    feedback_interaction_records: List[FeedbackEnrichment] = []
    for (feedback_node_id, feedback_props), (
        interaction_node_id,
        interaction_props,
    ) in matched_feedback_interaction_pairs:
        record = await _build_feedback_interaction_record(
            feedback_node_id, feedback_props, interaction_node_id, interaction_props
        )
        if record:
            feedback_interaction_records.append(record)
    return feedback_interaction_records


async def extract_feedback_interactions(
    data: Any, last_n: Optional[int] = None
) -> List[FeedbackEnrichment]:
    """Extract negative feedback-interaction pairs and create FeedbackEnrichment DataPoints."""
    if not data or data == [{}]:
        logger.info(
            "No data passed to the extraction task (extraction task fetches data from graph directly)",
            data=data,
        )

    graph_nodes, graph_edges = await _fetch_feedback_and_interaction_graph_data()
    if not graph_nodes:
        logger.warning("No graph nodes retrieved from database")
        return []

    feedback_nodes, interaction_nodes = _separate_feedback_and_interaction_nodes(graph_nodes)
    logger.info(
        "Retrieved nodes from graph",
        total_nodes=len(graph_nodes),
        feedback_nodes=len(feedback_nodes),
        interaction_nodes=len(interaction_nodes),
    )

    negative_feedback_nodes = _filter_negative_feedback(feedback_nodes)
    logger.info(
        "Filtered feedback nodes",
        total_feedback=len(feedback_nodes),
        negative_feedback=len(negative_feedback_nodes),
    )

    if not negative_feedback_nodes:
        logger.info("No negative feedback found; returning empty list")
        return []

    matched_feedback_interaction_pairs = _match_feedback_nodes_to_interactions_by_edges(
        negative_feedback_nodes, interaction_nodes, graph_edges
    )
    if not matched_feedback_interaction_pairs:
        logger.info("No feedback-to-interaction matches found; returning empty list")
        return []

    matched_feedback_interaction_pairs = _sort_pairs_by_recency_and_limit(
        matched_feedback_interaction_pairs, last_n
    )

    feedback_interaction_records = await _build_feedback_interaction_records(
        matched_feedback_interaction_pairs
    )

    logger.info("Extracted feedback pairs", count=len(feedback_interaction_records))
    return feedback_interaction_records
