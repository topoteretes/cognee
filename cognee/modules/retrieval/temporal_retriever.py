import os
from typing import Any, Dict, List, Optional, Tuple, Type
from datetime import datetime, timezone

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.used_graph_elements import extract_from_temporal_dict
from cognee.shared.logging_utils import get_logger

from cognee.modules.engine.utils.generate_timestamp_datapoint import date_to_int
from cognee.tasks.temporal_graph.models import QueryInterval, Timestamp
from cognee.tasks.temporal_graph.time_precision import expand_to_period_end

logger = get_logger()

__all__ = ["TemporalRetriever", "expand_to_period_end"]


def _event_time_key(event: Dict[str, Any]) -> Tuple[int, int]:
    """Chronological sort key; events without any time anchor sort last."""
    anchor = event.get("time_at", event.get("time_from", event.get("time_to")))
    return (0, int(anchor)) if anchor is not None else (1, 0)


def _format_ms(value: Any) -> str:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (ValueError, TypeError, OverflowError, OSError):
        return str(value)


def _event_time_label(event: Dict[str, Any]) -> Optional[str]:
    if event.get("time_at") is not None:
        return _format_ms(event["time_at"])
    time_from, time_to = event.get("time_from"), event.get("time_to")
    if time_from is not None and time_to is not None:
        return f"{_format_ms(time_from)} to {_format_ms(time_to)}"
    if time_from is not None:
        return f"from {_format_ms(time_from)}"
    if time_to is not None:
        return f"until {_format_ms(time_to)}"
    return None


class TemporalRetriever(GraphCompletionRetriever):
    """
    Time-aware graph completion.

    A query is first parsed into a time window (one structured LLM call). The window
    selects ``Timestamp`` nodes, the events anchored to them are collected from the
    graph, and those events are ranked by vector similarity to the question. The
    top-k events, rendered chronologically, form the LLM context. With no usable
    window (or on a backend without temporal queries) it falls back to the parent's
    triplet search. The public methods are:

    - get_completion
    - get_retrieved_objects
    - get_context_from_objects

    Instance variables include:
    - time_extraction_prompt_path
    - wide_search_top_k (size of the vector candidate pool used for ranking)
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        time_extraction_prompt_path: str = "extract_query_time.txt",
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 6.5,
        feedback_influence: float = get_base_config().default_feedback_influence,
        session_id: Optional[str] = None,
        response_model: Type = str,
        include_references: bool = False,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            feedback_influence=feedback_influence,
            session_id=session_id,
            response_model=response_model,
            include_references=include_references,
        )
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.time_extraction_prompt_path = time_extraction_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.node_type = node_type
        self.node_name = node_name

    def extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        """Extract node_ids/edge_ids from temporal dict (triplets or relevant_events)."""
        if isinstance(retrieved_objects, dict):
            return extract_from_temporal_dict(retrieved_objects)
        return None

    def descriptions_to_string(self, results: List[Dict[str, Any]]) -> str:
        """Render events chronologically, one block each, prefixed with their time."""
        blocks = []
        for event in sorted(results, key=_event_time_key):
            text = (event.get("description") or event.get("name") or "").strip()
            if not text:
                continue
            label = _event_time_label(event)
            if label:
                text = f"[{label}] {text}"
            if event.get("location"):
                text = f"{text} (location: {event['location']})"
            blocks.append(text)
        return "\n#####################\n".join(blocks)

    async def extract_time_from_query(self, query: str):
        prompt_path = self.time_extraction_prompt_path

        if os.path.isabs(prompt_path):
            base_directory = os.path.dirname(prompt_path)
            prompt_path = os.path.basename(prompt_path)
        else:
            base_directory = None

        time_now = datetime.now().strftime("%d-%m-%Y")

        system_prompt = render_prompt(
            prompt_path, {"time_now": time_now}, base_directory=base_directory
        )

        interval = await LLMGateway.acreate_structured_output(query, system_prompt, QueryInterval)

        time_from = interval.starts_at
        # A year-only "1996" parses as 1996-01-01; the user meant the whole year.
        time_to = expand_to_period_end(interval.ends_at)

        return time_from, time_to

    async def filter_top_k_events(
        self, relevant_events: List[Dict[str, Any]], scored_results
    ) -> List[Dict[str, Any]]:
        """Rank the time-window events by vector similarity and keep the top k.

        Events that the vector search scored come first, best (lowest distance)
        first. Events the candidate pool did not reach are kept after them in
        chronological order rather than dropped, so a sparse pool still yields a
        full, deterministic top-k.
        """
        # ScoredResult.id is a UUID while event["id"] arrives from the graph as a
        # string, so both sides are normalized to str or every lookup misses.
        score_lookup = {str(res.id): res.score for res in scored_results}

        scored, unscored = [], []
        for event in relevant_events:
            score = score_lookup.get(str(event["id"]))
            if score is None:
                unscored.append({**event, "score": float("inf")})
            else:
                scored.append({**event, "score": score})

        scored.sort(key=lambda event: event["score"])
        unscored.sort(key=_event_time_key)

        return (scored + unscored)[: self.top_k]

    async def get_retrieved_objects(self, query: str) -> dict:
        time_from, time_to = await self.extract_time_from_query(query)

        if not time_from and not time_to:
            logger.info(
                "No timestamps identified based on the query, performing retrieval using triplet search on events and entities."
            )
            return {"triplets": await self.get_triplets(query)}

        unified = await get_unified_engine()
        graph_engine = unified.graph

        # When the triplet fallback runs for a question about a period, a fact
        # closed *after* that period began was still true then: judge validity
        # as of the window start, not as of now.
        as_of_ms = date_to_int(time_from) if isinstance(time_from, Timestamp) else None

        try:
            ids = await graph_engine.collect_time_ids(time_from=time_from, time_to=time_to)
            relevant_events = await graph_engine.collect_events(ids) if ids else []
        except NotImplementedError as error:
            logger.warning(
                "%s. Falling back to triplet search for this TEMPORAL query.", str(error)
            )
            return {"triplets": await self.get_triplets(query, as_of_ms=as_of_ms)}

        if not relevant_events:
            logger.info(
                "No events identified based on timestamp filtering, performing retrieval using triplet search on events and entities."
            )
            return {"triplets": await self.get_triplets(query, as_of_ms=as_of_ms)}

        vector_engine = unified.vector
        query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]

        # The candidate pool must be wider than top_k: the search runs over every
        # event in the collection, and only the hits that fall inside the time
        # window can be ranked. A pool of top_k would leave most window events
        # unscored and the ranking would collapse to graph order.
        candidate_pool = max(self.wide_search_top_k, self.top_k, len(relevant_events))
        vector_search_results = await vector_engine.search(
            collection_name="Event_name", query_vector=query_vector, limit=candidate_pool
        )

        return {"relevant_events": relevant_events, "vector_search_results": vector_search_results}

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> Any:
        """Retrieves context based on the query."""
        relevant_events = retrieved_objects.get("relevant_events")
        if relevant_events:
            top_k_events = await self.filter_top_k_events(
                relevant_events, retrieved_objects.get("vector_search_results") or []
            )
            # Record what actually reached the prompt so provenance / feedback
            # attribute the answer to these events, not the whole candidate pool.
            retrieved_objects["selected_events"] = top_k_events
            return self.descriptions_to_string(top_k_events)

        # In case no events were found, fall back to triplet context
        triplets = retrieved_objects.get("triplets", [])
        return await self.resolve_edges_to_text(triplets)
