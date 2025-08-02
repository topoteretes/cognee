from typing import Any, Optional, Type, List
from collections import Counter
import string

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS
from cognee.shared.logging_utils import get_logger
from cognee.temporal_poc.models.models import QueryInterval
from cognee.temporal_poc.temporal_cognify import date_to_int

logger = get_logger("TemporalRetriever")


class TemporalRetriever(BaseRetriever):
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

    async def extract_time_from_query(self, query: str):
        llm_client = get_llm_client()

        system_prompt = """
                For the purposes of identifying timestamps in a query, you are tasked with extracting relevant timestamps from the query.
                ## Timestamp requirements
                - If the query contains interval extrack both starts_at and ends_at  properties
                - If the query contains an instantaneous timestamp, starts_at and ends_at should be the same
                - If the query its open ended (before 2009 or after 2009), the corresponding non defined end of the time should be none
                    -For example: "before 2009" -- starts_at: None, ends_at: 2009 or  "after 2009" -- starts_at: 2009, ends_at: None
                - Put always the data that comes first in time as starts_at and the timestamps that comes second in time as ends_at
                ## Output Format
                Your reply should be a JSON: list of dictionaries with the following structure:
                ```python
                class QueryInterval(BaseModel):
                    starts_at: Optional[Timestamp] = None
                    ends_at: Optional[Timestamp] = None
                ```
        """

        interval = await llm_client.acreate_structured_output(query, system_prompt, QueryInterval)

        return interval

    def descriptions_to_string(self, results):
        descs = []
        for entry in results:
            events = entry.get("events", [])
            for ev in events:
                d = ev.get("description")
                if d:
                    descs.append(d.strip())
        return "\n-".join(descs)

    async def get_context(self, query: str) -> str:
        # :TODO: This is a POC and yes this method is far far far far from nice :D

        graph_engine = await get_graph_engine()
        interval = await self.extract_time_from_query(query=query)

        time_from = interval.starts_at
        time_to = interval.ends_at

        event_collection_cypher = """UNWIND [{quoted}] AS uid
            MATCH (start {{id: uid}})
            MATCH (start)-[*1..2]-(event)
            WHERE event.type = 'Event'
            WITH DISTINCT event
            RETURN collect(event) AS events;
            """

        if time_from and time_to:
            time_from = date_to_int(time_from)
            time_to = date_to_int(time_to)

            cypher = """
                    MATCH (n)
                    WHERE n.type = 'Timestamp'
                      AND n.time_at >= $time_from
                      AND n.time_at <= $time_to
                    RETURN n.id AS id
                    """
            params = {"time_from": time_from, "time_to": time_to}
            time_nodes = await graph_engine.query(cypher, params)

            time_ids_list = [item["id"] for item in time_nodes if "id" in item]

            ids = ", ".join("'{0}'".format(uid) for uid in time_ids_list)

            event_collection_cypher = event_collection_cypher.format(quoted=ids)
            relevant_events = await graph_engine.query(event_collection_cypher)

            context = self.descriptions_to_string(relevant_events)

            return context
        elif time_from:
            time_from = date_to_int(time_from)

            cypher = """
                    MATCH (n)
                    WHERE n.type = 'Timestamp'
                      AND n.time_at >= $time_from
                    RETURN n.id AS id
                    """
            params = {"time_from": time_from}
            time_nodes = await graph_engine.query(cypher, params)

            time_ids_list = [item["id"] for item in time_nodes if "id" in item]

            ids = ", ".join("'{0}'".format(uid) for uid in time_ids_list)

            event_collection_cypher = event_collection_cypher.format(quoted=ids)
            relevant_events = await graph_engine.query(event_collection_cypher)

            context = self.descriptions_to_string(relevant_events)

            return context

        elif time_to:
            time_to = date_to_int(time_to)

            cypher = """
                    MATCH (n)
                    WHERE n.type = 'Timestamp'
                      AND n.time_at <= $time_to
                    RETURN n.id AS id
                    """
            params = {"time_to": time_to}

            time_nodes = await graph_engine.query(cypher, params)

            time_ids_list = [item["id"] for item in time_nodes if "id" in item]

            ids = ", ".join("'{0}'".format(uid) for uid in time_ids_list)

            event_collection_cypher = event_collection_cypher.format(quoted=ids)
            relevant_events = await graph_engine.query(event_collection_cypher)

            context = self.descriptions_to_string(relevant_events)

            return context
        else:
            logger.info(
                "We couldn't find any timestamps in this query therefore we return empty context"
            )
            return ""

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]

    def _top_n_words(self, text, stop_words=None, top_n=3, separator=", "):
        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS

        words = [word.lower().strip(string.punctuation) for word in text.split()]

        if stop_words:
            words = [word for word in words if word and word not in stop_words]

        top_words = [word for word, freq in Counter(words).most_common(top_n)]

        return separator.join(top_words)

    def _get_title(self, text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
        first_n_words = text.split()[:first_n_words]
        top_n_words = self._top_n_words(text, top_n=top_n_words)
        return f"{' '.join(first_n_words)}... [{top_n_words}]"
