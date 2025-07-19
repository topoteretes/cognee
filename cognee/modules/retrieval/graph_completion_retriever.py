from typing import Any, Optional, Type, List
from collections import Counter
import string

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS
from cognee.shared.logging_utils import get_logger

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
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.node_type = node_type
        self.node_name = node_name
        logger.info(
            f"Initialized GraphCompletionRetriever with top_k={self.top_k}, node_type={self.node_type}, node_name={self.node_name}"
        )

    def resolve_edges_to_text(self, edges) -> str:
        """
        Transform nodes and relationships within edges to a human-readable text format.

        Takes a list of edges and converts them into natural text by extracting name, text,
        and description attributes of each relationship. Suitable formatting is applied if
        attributes are missing.

        Parameters:
        -----------

            - edges: The edges from the graph to transform into text. Each edge should be a
              list/tuple with three elements (subject, relationship, object) or Edge objects.

        Returns:
        --------

            - str: A formatted text representation of the edges.
        """
        logger.debug(f"Resolving {len(edges)} edges to text")

        relationships_text = []
        for edge in edges:
            # Handle Edge objects from CogneeGraphElements
            if hasattr(edge, "node1") and hasattr(edge, "node2") and hasattr(edge, "attributes"):
                subject = edge.node1.attributes
                obj = edge.node2.attributes
                relationship = edge.attributes
            else:
                # Handle tuple format (subject, relationship, object)
                subject, relationship, obj = edge

            def get_name_and_content(node_data, name_field="name"):
                name = node_data.get(name_field)
                text = node_data.get("text")
                description = node_data.get("description")

                if description:
                    return f"{name} ({description})"
                if text:
                    return f"{name} ({text})"
                if name:
                    return name
                else:
                    return "Unknown"

            subject_name = get_name_and_content(subject)
            relationship_name = relationship.get("relationship_name", "UNKNOWN")
            object_name = get_name_and_content(obj)

            relationships_text.append(f"{subject_name} {relationship_name} {object_name}")

        result_text = "\n".join(relationships_text)
        logger.debug(f"Generated {len(result_text)} characters of relationship text")
        return result_text

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
        logger.info(
            f"Starting triplet search for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

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

        logger.debug(f"Found {len(vector_index_collections)} vector index collections to search")

        found_triplets = await brute_force_triplet_search(
            query,
            top_k=self.top_k,
            collections=vector_index_collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
        )

        logger.info(f"Retrieved {len(found_triplets)} triplets from graph search")
        return found_triplets

    async def get_context(self, query: str) -> str:
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
        logger.info(
            f"Starting context retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        triplets = await self.get_triplets(query)

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return ""

        context = self.resolve_edges_to_text(triplets)
        logger.info(
            f"Generated context with {len(context)} characters from {len(triplets)} triplets"
        )
        return context

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
        logger.info(
            f"Starting completion generation for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        if context is None:
            logger.debug("No context provided, retrieving context from graph")
            context = await self.get_context(query)
        else:
            logger.debug("Using provided context")

        logger.info(
            f"Generating completion with context length: {len(str(context)) if context else 0} characters"
        )

        try:
            completion = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )
            logger.info("Graph completion generation successful")
            return [completion]
        except Exception as e:
            logger.error(f"Error during graph completion generation: {str(e)}")
            raise

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
