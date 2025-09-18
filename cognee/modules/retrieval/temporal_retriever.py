import os
from typing import Any, Optional, List, Type


from operator import itemgetter
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.shared.logging_utils import get_logger


from cognee.tasks.temporal_graph.models import QueryInterval

logger = get_logger()


class TemporalRetriever(GraphCompletionRetriever):
    """
    Handles graph completion by generating responses based on a series of interactions with
    a language model. This class extends from GraphCompletionRetriever and is designed to
    manage the retrieval and validation process for user queries, integrating follow-up
    questions based on reasoning. The public methods are:

    - get_completion

    Instance variables include:
    - validation_system_prompt_path
    - validation_user_prompt_path
    - followup_system_prompt_path
    - followup_user_prompt_path
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        time_extraction_prompt_path: str = "extract_query_time.txt",
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
        )
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.time_extraction_prompt_path = time_extraction_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.node_type = node_type
        self.node_name = node_name

    def descriptions_to_string(self, results):
        descs = []
        for entry in results:
            d = entry.get("description")
            if d:
                descs.append(d.strip())
        return "\n#####################\n".join(descs)

    async def extract_time_from_query(self, query: str):
        prompt_path = self.time_extraction_prompt_path

        if os.path.isabs(prompt_path):
            base_directory = os.path.dirname(prompt_path)
            prompt_path = os.path.basename(prompt_path)
        else:
            base_directory = None

        system_prompt = LLMGateway.render_prompt(prompt_path, {}, base_directory=base_directory)

        interval = await LLMGateway.acreate_structured_output(query, system_prompt, QueryInterval)

        time_from = interval.starts_at
        time_to = interval.ends_at

        return time_from, time_to

    async def filter_top_k_events(self, relevant_events, scored_results):
        # Build a score lookup from vector search results
        score_lookup = {res.payload["id"]: res.score for res in scored_results}

        events_with_scores = []
        for event in relevant_events[0]["events"]:
            score = score_lookup.get(event["id"], float("inf"))
            events_with_scores.append({**event, "score": score})

        events_with_scores.sort(key=itemgetter("score"))

        return events_with_scores[: self.top_k]

    async def get_context(self, query: str) -> Any:
        """Retrieves context based on the query."""

        time_from, time_to = await self.extract_time_from_query(query)

        graph_engine = await get_graph_engine()

        triplets = []

        if time_from and time_to:
            ids = await graph_engine.collect_time_ids(time_from=time_from, time_to=time_to)
        elif time_from:
            ids = await graph_engine.collect_time_ids(time_from=time_from)
        elif time_to:
            ids = await graph_engine.collect_time_ids(time_to=time_to)
        else:
            logger.info(
                "No timestamps identified based on the query, performing retrieval using triplet search on events and entities."
            )
            triplets = await self.get_triplets(query)
            return await self.resolve_edges_to_text(triplets)

        if ids:
            relevant_events = await graph_engine.collect_events(ids=ids)
        else:
            logger.info(
                "No events identified based on timestamp filtering, performing retrieval using triplet search on events and entities."
            )
            triplets = await self.get_triplets(query)
            return await self.resolve_edges_to_text(triplets)

        vector_engine = get_vector_engine()
        query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]

        vector_search_results = await vector_engine.search(
            collection_name="Event_name", query_vector=query_vector, limit=0
        )

        top_k_events = await self.filter_top_k_events(relevant_events, vector_search_results)

        return self.descriptions_to_string(top_k_events)

    async def get_completion(self, query: str, context: Optional[str] = None) -> List[str]:
        """Generates a response using the query and optional context."""
        if not context:
            context = await self.get_context(query=query)

        if context:
            completion = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )

        return [completion]
