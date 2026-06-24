from collections import Counter
from typing import Any, Optional
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.infrastructure.databases.unified import get_unified_engine


class GraphAggregationRetriever(BaseRetriever):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def get_retrieved_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
    ) -> Any:
        unified_engine = await get_unified_engine()
        nodes, edges = await unified_engine.graph.get_graph_data()

        return {
            "query": query,
            "nodes": nodes,
            "edges": edges,
        }

    def _detect_operation(self, query: str):
        query = query.lower()

        if "how many" in query or "count" in query:
            if "by type" in query or "by_type" in query:
                return "group_by_count"
            return "count"

        if (
            "most connected" in query
            or "top connected" in query
            or "most relationships" in query
            or "highest degree" in query
        ):
            return "top_by_degree"

        return "count"

    def _count_nodes(self, nodes):
        return len(nodes)

    def _count_by_type(self, nodes):
        counts = Counter()
        for _, props in nodes:
            node_type = props.get("type", "UNKNOWN")
            counts[node_type] += 1
        return dict(counts)

    def _top_by_degree(self, nodes, edges):
        degree = Counter()
        for source, target, _, _ in edges:
            degree[source] += 1
            degree[target] += 1

        if not degree:
            return None

        top_node_id = degree.most_common(1)[0][0]

        for node_id, props in nodes:
            if node_id == top_node_id:
                return {
                    "node_id": node_id,
                    "name": props.get("name"),
                    "degree": degree[top_node_id],
                }
        return None
    
    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> Any:
        nodes = retrieved_objects["nodes"]
        edges = retrieved_objects["edges"]

        # Safely compute structural components
        return {
            "total_nodes": self._count_nodes(nodes),
            "node_types": self._count_by_type(nodes),
            "top_connected_node": self._top_by_degree(nodes, edges),
            "nodes": nodes,
            "edges": edges,
        }

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[list] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ):
        nodes = context["nodes"]
        edges = context["edges"]
        operation = self._detect_operation(query or "")

        # Execute analytical fallbacks directly for speed / structure accuracy
        if operation == "count":
            count = self._count_nodes(nodes)
            analytics_data = f"There are {count} entities in the graph."
        elif operation == "group_by_count":
            analytics_data = self._count_by_type(nodes)
        elif operation == "top_by_degree":
            analytics_data = self._top_by_degree(nodes, edges)
        else:
            analytics_data = "Unsupported aggregation operation"

        system_prompt = f"""
You are a graph analytics assistant.
Use the provided exact calculated graph statistics to clearly answer the user's natural language question.

Calculated Statistics:
{analytics_data}
"""

        response = await LLMGateway.acreate_structured_output(
            text_input=query or "",
            system_prompt=system_prompt,
            response_model=str,
        )

        return [response]