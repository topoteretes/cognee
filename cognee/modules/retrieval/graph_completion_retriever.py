import asyncio
from typing import Any, Optional, Type, List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.tasks.storage import add_data_points
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion, summarize_text
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.utils.extract_uuid_from_node import extract_uuid_from_node
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.modules.engine.models.node_set import NodeSet
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.modules.graph.utils import get_entity_nodes_from_triplets

logger = get_logger("GraphCompletionRetriever")


class GraphCompletionRetriever(BaseRetriever):
    """
    Retriever for handling graph-based completion searches.

    This class implements the retrieval pipeline by searching for graph triplets (get_retrieved_objects function),
    resolving those triplets into human-readable text context (get_context_from_objects function), and generating
    LLM completions using the retrieved graph data (get_completion_from_context function).
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        save_interaction: bool = False,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.save_interaction = save_interaction
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.top_k = top_k if top_k is not None else 5
        self.wide_search_top_k = wide_search_top_k
        self.node_type = node_type
        self.node_name = node_name
        self.triplet_distance_penalty = triplet_distance_penalty
        # session_id (Optional[str]): Identifier for managing conversation history.
        self.session_id = session_id
        # response_model (Type): The Pydantic model or type for the expected response.
        self.response_model = response_model

    async def get_retrieved_objects(self, query: str) -> List[Edge]:
        """
        Performs a brute-force triplet search on the graph and updates access timestamps.

        Args:
            query (str): The search query to find relevant graph triplets.

        Returns:
            List[Edge]: A list of retrieved Edge objects (triplets).
                       Returns an empty list if the graph is empty or no results are found.
        """
        graph_engine = await get_graph_engine()
        is_empty = await graph_engine.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            return []

        triplets = await self.get_triplets(query)

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return []
        # TODO: Remove when refactor of timestamps tracking is merged
        entity_nodes = get_entity_nodes_from_triplets(triplets)
        await update_node_access_timestamps(entity_nodes)

        return triplets

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """
        Converts retrieved graph edges into a human-readable string format.

        Parameters:
        -----------

            - retrieved_edges (list): A list of edges retrieved from the graph.

        Returns:
        --------

            - str: A formatted string representation of the nodes and their connections.
        """
        return await resolve_edges_to_text(retrieved_edges)

    async def get_triplets(self, query: str) -> List[Edge]:
        """
        Retrieves relevant graph triplets based on a query string.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant triplets in the graph.

        Returns:
        --------

            - list: A list of found triplets that match the query.
        """
        subclasses = get_all_subclasses(DataPoint)
        vector_index_collections: List[str] = []

        for subclass in subclasses:
            if "metadata" in subclass.model_fields:
                metadata_field = subclass.model_fields["metadata"]
                if hasattr(metadata_field, "default") and metadata_field.default is not None:
                    if isinstance(metadata_field.default, dict):
                        index_fields = metadata_field.default.get("index_fields", [])
                        for field_name in index_fields:
                            vector_index_collections.append(f"{subclass.__name__}_{field_name}")

        found_triplets = await brute_force_triplet_search(
            query,
            top_k=self.top_k,
            collections=vector_index_collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
            wide_search_top_k=self.wide_search_top_k,
            triplet_distance_penalty=self.triplet_distance_penalty,
        )

        return found_triplets

    async def get_context_from_objects(self, query, retrieved_objects) -> str:
        """
        Transforms raw retrieved graph triplets into a textual context string.

        Args:
            query (str): The original search query.
            retrieved_objects (List[Edge]): The raw triplets returned from the search.
                                            Output of the get_retrieved_objects method.

        Returns:
            str: A string representing the resolved graph context.
                 Returns an empty list (as string) if no triplets are provided.

        Note: To avoid duplicate retrievals, ensure that retrieved_objects
              are provided from get_retrieved_objects method call.
        """

        triplets = retrieved_objects

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return ""

        return await self.resolve_edges_to_text(triplets)

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Optional[List[Edge]],
        context: str,
    ) -> List[Any]:
        """
        Generates an LLM response based on the query, context, and conversation history.
        Optionally saves the interaction and updates the session cache.

        Args:
            query (str): The user's question or prompt.
            retrieved_objects (Optional[List[Edge]]): Raw triplets used for interaction mapping.
                                                     Output of get_retrieved_objects method.
            context (str): The text-resolved graph context.
                           Output of the get_context_from_objects method.

        Returns:
            List[Any]: A list containing the generated response (completion).

        Note: To avoid duplicate retrievals, ensure that retrieved_objects and context
              are provided from previous method calls.
        """

        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        if session_save:
            conversation_history = await get_conversation_history(session_id=self.session_id)

            context_summary, completion = await asyncio.gather(
                summarize_text(context),
                generate_completion(
                    query=query,
                    context=context,
                    user_prompt_path=self.user_prompt_path,
                    system_prompt_path=self.system_prompt_path,
                    system_prompt=self.system_prompt,
                    conversation_history=conversation_history,
                    response_model=self.response_model,
                ),
            )
        else:
            completion = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
            )

        if self.save_interaction and retrieved_objects and completion:
            await self.save_qa(
                question=query, answer=completion, context=context, triplets=retrieved_objects
            )

        if session_save:
            await save_conversation_history(
                query=query,
                context_summary=context_summary,
                answer=completion,
                session_id=self.session_id,
            )

        return [completion]

    async def save_qa(self, question: str, answer: str, context: str, triplets: List) -> None:
        """
        Saves a question and answer pair for later analysis or storage.
        Parameters:
        -----------
            - question (str): The question text.
            - answer (str): The answer text.
            - context (str): The context text.
            - triplets (List): A list of triples retrieved from the graph.
        """
        nodeset_name = "Interactions"
        interactions_node_set = NodeSet(
            id=uuid5(NAMESPACE_OID, name=nodeset_name), name=nodeset_name
        )
        source_id = uuid5(NAMESPACE_OID, name=(question + answer + context))

        cognee_user_interaction = CogneeUserInteraction(
            id=source_id,
            question=question,
            answer=answer,
            context=context,
            belongs_to_set=interactions_node_set,
        )

        await add_data_points(data_points=[cognee_user_interaction])

        relationships = []
        relationship_name = "used_graph_element_to_answer"
        for triplet in triplets:
            target_id_1 = extract_uuid_from_node(triplet.node1)
            target_id_2 = extract_uuid_from_node(triplet.node2)
            if target_id_1 and target_id_2:
                relationships.append(
                    (
                        source_id,
                        target_id_1,
                        relationship_name,
                        {
                            "relationship_name": relationship_name,
                            "source_node_id": source_id,
                            "target_node_id": target_id_1,
                            "ontology_valid": False,
                            "feedback_weight": 0,
                        },
                    )
                )

                relationships.append(
                    (
                        source_id,
                        target_id_2,
                        relationship_name,
                        {
                            "relationship_name": relationship_name,
                            "source_node_id": source_id,
                            "target_node_id": target_id_2,
                            "ontology_valid": False,
                            "feedback_weight": 0,
                        },
                    )
                )

            if len(relationships) > 0:
                graph_engine = await get_graph_engine()
                await graph_engine.add_edges(relationships)
