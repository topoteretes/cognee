from typing import Any, Optional, Type, List, Coroutine
from collections import Counter
from uuid import NAMESPACE_OID, uuid5
import string

from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.utils.extract_uuid_from_node import extract_uuid_from_node
from cognee.modules.retrieval.utils.models import CogneeUserInteraction
from cognee.modules.engine.models.node_set import NodeSet
from cognee.infrastructure.databases.graph import get_graph_engine

logger = get_logger("GraphCompletionRetriever")


class GraphCompletionRetriever(BaseRetriever):
    """
    Retriever for handling graph-based completion searches.

    This class provides methods to retrieve graph nodes and edges, resolve them into a
    human-readable format, and generate completions based on graph context. Public methods
    include:
    - resolve_edges_to_text
    - get_triplets
    - get_context
    - get_completion
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        save_interaction: bool = False,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.save_interaction = save_interaction
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.node_type = node_type
        self.node_name = node_name

    def _get_nodes(self, retrieved_edges: list) -> dict:
        """Creates a dictionary of nodes with their names and content."""
        nodes = {}
        for edge in retrieved_edges:
            for node in (edge.node1, edge.node2):
                if node.id not in nodes:
                    text = node.attributes.get("text")
                    if text:
                        name = self._get_title(text)
                        content = text
                    else:
                        name = node.attributes.get("name", "Unnamed Node")
                        content = node.attributes.get("description", name)
                    nodes[node.id] = {"node": node, "name": name, "content": content}
        return nodes

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
        nodes = self._get_nodes(retrieved_edges)
        node_section = "\n".join(
            f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
            for info in nodes.values()
        )
        connection_section = "\n".join(
            f"{nodes[edge.node1.id]['name']} --[{edge.attributes['relationship_type']}]--> {nodes[edge.node2.id]['name']}"
            for edge in retrieved_edges
        )
        return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"

    async def get_triplets(self, query: str) -> list:
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
        vector_index_collections = []

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
        )

        return found_triplets

    async def get_context(self, query: str) -> str | tuple[str, list]:
        """
        Retrieves and resolves graph triplets into context based on a query.

        Parameters:
        -----------

            - query (str): The query string used to retrieve context from the graph triplets.

        Returns:
        --------

            - str: A string representing the resolved context from the retrieved triplets, or an
              empty string if no triplets are found.
        """
        triplets = await self.get_triplets(query)

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return "", triplets

        context = await self.resolve_edges_to_text(triplets)

        return context, triplets

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates a completion using graph connections context based on a query.

        Parameters:
        -----------

            - query (str): The query string for which a completion is generated.
            - context (Optional[Any]): Optional context to use for generating the completion; if
              not provided, context is retrieved based on the query. (default None)

        Returns:
        --------

            - Any: A generated completion based on the query and context provided.
        """
        triplets = None

        if context is None:
            context, triplets = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )

        if self.save_interaction and context and triplets and completion:
            await self.save_qa(
                question=query, answer=completion, context=context, triplets=triplets
            )

        return [completion]

    def _top_n_words(self, text, stop_words=None, top_n=3, separator=", "):
        """Concatenates the top N frequent words in text."""
        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS

        words = [word.lower().strip(string.punctuation) for word in text.split()]

        if stop_words:
            words = [word for word in words if word and word not in stop_words]

        top_words = [word for word, freq in Counter(words).most_common(top_n)]

        return separator.join(top_words)

    def _get_title(self, text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
        """Creates a title, by combining first words with most frequent words from the text."""
        first_n_words = text.split()[:first_n_words]
        top_n_words = self._top_n_words(text, top_n=top_n_words)
        return f"{' '.join(first_n_words)}... [{top_n_words}]"

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

        await add_data_points(data_points=[cognee_user_interaction], update_edge_collection=False)

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
