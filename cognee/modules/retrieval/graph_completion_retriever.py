from typing import Any, Optional, List
from collections import Counter
import string
import logging

from pydantic import BaseModel, Field
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS
from cognee.modules.search.models.Query import Query

t_logger = logging.getLogger(__name__)
t_logger.setLevel(logging.INFO)


class AnswerValidation(BaseModel):
    is_valid: bool = Field(
        ...,
        description="Indicates whether the answer to the question is fully supported by the context",
    )
    reasoning: str = Field("", description="Detailed reasoning of what is missing from the context")


class GraphCompletionRetriever(BaseRetriever):
    """Retriever for handling graph-based completion searches."""

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: Optional[int] = 5,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5

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
                        content = name
                    nodes[node.id] = {"node": node, "name": name, "content": content}
        return nodes

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """Converts retrieved graph edges into a human-readable string format."""
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
        """Retrieves relevant graph triplets."""
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
            query, top_k=self.top_k, collections=vector_index_collections or None
        )

        return found_triplets

    async def get_context(self, query: str) -> str:
        """Retrieves and resolves graph triplets into context."""
        triplets = await self.get_triplets(query)

        if len(triplets) == 0:
            return ""

        return await self.resolve_edges_to_text(triplets)

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates a completion using graph connections context."""
        """
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )

        """

        completion = await self.get_chain_of_thought(query=query)
        return [completion]

    async def get_chain_of_thought(self, query, max_iter=4):
        llm_client = get_llm_client()
        followup_question = ""
        triplets = []

        for round_idx in range(max_iter + 1):
            if round_idx == 0:
                triplets = await self.get_triplets(query)
                context = await self.resolve_edges_to_text(triplets)
            else:
                triplets += await self.get_triplets(followup_question)
                context = await self.resolve_edges_to_text(list(set(triplets)))
                t_logger.info(f"Round {round_idx} - context: {context}")

            # Generate answer
            answer = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )
            t_logger.info(f"Round {round_idx} - answer: {answer}")

            # Prepare validation prompt
            valid_user_prompt = (
                f"""\n\n--Question--\n{query}\n\n--ANSWER--\n{answer}\n\n--CONTEXT--\n{context}\n"""
            )

            valid_system_prompt = "You are a helpful agent who are allowed to use only the provided question answer and context. I want to you find reasoning what is missing from the context or why the answer is not answering the question or not correct strictly based on the context"

            reasoning = await llm_client.acreate_structured_output(
                text_input=valid_user_prompt,
                system_prompt=valid_system_prompt,
                response_model=str,
            )

            # Ask follow-up question to fill gaps
            followup_system = (
                "You are a helpful assistant whose job is to ask exactly one clarifying follow-up question,"
                " to collect the missing piece of information needed to fully answer the user’s original query."
                " Respond with the question only (no extra text, no punctuation beyond what’s needed)."
            )

            followup_prompt = (
                "Based on the following, ask exactly one question that would directly resolve the gap identified in the validation reasoning and allow a valid answer."
                "Think in a way that with the followup question you are exploring a knowledge graph which contains entities, entity types and document chunks\n\n"
                f"Query: {query}\n"
                f"Answer: {answer}\n"
                f"Reasoning:\n{reasoning}\n"
            )
            followup_question = await llm_client.acreate_structured_output(
                text_input=followup_prompt, system_prompt=followup_system, response_model=str
            )
            t_logger.info(f"Round {round_idx} - follow-up question: {followup_question}")

        # Fallback if no iteration passed validation
        return answer

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
