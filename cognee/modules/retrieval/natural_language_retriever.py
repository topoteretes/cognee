from typing import Any, Optional
import logging
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions import SearchTypeNotSupported
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger("NaturalLanguageRetriever")


class NaturalLanguageRetriever(BaseRetriever):
    """Retriever for handling natural language search"""

    def __init__(
        self,
        system_prompt_path: str = "natural_language_retriever_system.txt",
        max_attempts: int = 3,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.system_prompt_path = system_prompt_path
        self.max_attempts = max_attempts

    async def _get_graph_schema(self, graph_engine) -> tuple:
        """Retrieve the node and edge schemas from the graph database."""
        node_schemas = await graph_engine.query(
            """
            MATCH (n)
            UNWIND keys(n) AS prop
            RETURN DISTINCT labels(n) AS NodeLabels, collect(DISTINCT prop) AS Properties;
            """
        )
        edge_schemas = await graph_engine.query(
            """
            MATCH ()-[r]->()
            UNWIND keys(r) AS key
            RETURN DISTINCT key;
            """
        )
        return node_schemas, edge_schemas

    async def _generate_cypher_query(self, query: str, edge_schemas, previous_attempts=None) -> str:
        """Generate a Cypher query using LLM based on natural language query and schema information."""
        llm_client = get_llm_client()
        system_prompt = render_prompt(
            self.system_prompt_path,
            context={
                "edge_schemas": edge_schemas,
                "previous_attempts": previous_attempts or "No attempts yet",
            },
        )

        return await llm_client.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=str,
        )

    async def _execute_cypher_query(self, query: str, graph_engine: GraphDBInterface) -> Any:
        """Execute the natural language query against Neo4j with multiple attempts."""
        node_schemas, edge_schemas = await self._get_graph_schema(graph_engine)
        previous_attempts = ""
        cypher_query = ""

        for attempt in range(self.max_attempts):
            logger.info(f"Starting attempt {attempt + 1}/{self.max_attempts} for query generation")
            try:
                cypher_query = await self._generate_cypher_query(
                    query, edge_schemas, previous_attempts
                )

                logger.info(
                    f"Executing generated Cypher query (attempt {attempt + 1}): {cypher_query[:100]}..."
                    if len(cypher_query) > 100
                    else cypher_query
                )
                context = await graph_engine.query(cypher_query)

                if context:
                    result_count = len(context) if isinstance(context, list) else 1
                    logger.info(
                        f"Successfully executed query (attempt {attempt + 1}): returned {result_count} result(s)"
                    )
                    return context

                previous_attempts += f"Query: {cypher_query} -> Result: None\n"

            except Exception as e:
                previous_attempts += f"Query: {cypher_query if 'cypher_query' in locals() else 'Not generated'} -> Executed with error: {e}\n"
                logger.error(f"Error executing query: {str(e)}")

        logger.warning(
            f"Failed to get results after {self.max_attempts} attempts for query: '{query[:50]}...'"
        )
        return []

    async def get_context(self, query: str) -> Optional[Any]:
        """Retrieves relevant context using a natural language query converted to Cypher."""
        try:
            graph_engine = await get_graph_engine()

            if isinstance(graph_engine, (NetworkXAdapter)):
                raise SearchTypeNotSupported("Natural language search type not supported.")

            return await self._execute_cypher_query(query, graph_engine)
        except Exception as e:
            logger.error("Failed to execute natural language search retrieval: %s", str(e))
            raise e

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Returns a completion based on the query and context."""
        if context is None:
            context = await self.get_context(query)

        return context
