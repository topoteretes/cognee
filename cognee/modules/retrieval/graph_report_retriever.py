from typing import Any, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.graph_report import (
    build_graph_report_with_suggested_questions,
    render_graph_report_markdown,
)
from cognee.modules.retrieval.base_retriever import BaseRetriever


class GraphReportRetriever(BaseRetriever):
    """Build a graph insight report from the current dataset graph."""

    def __init__(
        self,
        top_n: int = 10,
        node_name: Optional[list[str]] = None,
        node_name_filter_operator: str = "OR",
        use_llm_questions: bool = True,
    ):
        self.top_n = top_n
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.use_llm_questions = use_llm_questions

    async def get_retrieved_objects(self, query: Optional[str] = None) -> Any:
        graph_engine = await get_graph_engine()
        graph_data = await graph_engine.get_graph_data()

        return await build_graph_report_with_suggested_questions(
            graph_data,
            top_n=self.top_n,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
            use_llm_questions=self.use_llm_questions,
        )

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> str:
        if not retrieved_objects:
            return ""
        return render_graph_report_markdown(retrieved_objects)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[list[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> dict:
        return retrieved_objects or {}
