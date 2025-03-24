from typing import Any, Optional
import logging
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.exceptions import SearchTypeNotSupported
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger("NaturalLanguageRetriever")


class NaturalLanguageRetriever(BaseRetriever):
    """Retriever for handling natural language search"""

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        max_attempts: int = 3,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
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

    async def _generate_cypher_query(
        self, query: str, node_schemas, edge_schemas, previous_attempts=None
    ) -> str:
        """Generate a Cypher query using LLM based on natural language query and schema information."""
        llm_client = get_llm_client()

        text_input = f"""
User query:
{query}
        """

        system_prompt = f"""
You are an expert Neo4j Cypher query generator tasked with translating natural language questions into precise, optimized Cypher queries.

TASK:
Generate a valid, executable Cypher query that accurately answers the user's question based on the provided graph schema.

GRAPH SCHEMA INFORMATION:
- You will be given node labels and their properties in format: NodeLabels [list of properties]
- You will be given relationship types between nodes
- ONLY use node labels, properties, and relationship types that exist in the provided schema
- Respect relationship directions (source→target) exactly as specified in the schema
- Properties may have specific formats (e.g., dates, codes) - infer these from examples when possible

QUERY REQUIREMENTS:
1. Return ONLY the exact Cypher query with NO explanations, comments, or markdown
2. Generate syntactically correct Neo4j Cypher code (Neo4j 4.4+ compatible)
3. Be precise - match the exact property names and relationship types from the schema
4. Handle complex queries by breaking them into logical pattern matching parts
5. Use parameters (e.g., $name) for literal values when appropriate
6. Use appropriate data types for parameters (strings, numbers, booleans)

PERFORMANCE OPTIMIZATION:
1. Use indexes and constraints when available (assume they exist on ID properties)
2. Include LIMIT clauses for queries that could return large result sets
3. Use efficient patterns - avoid unnecessary pattern complexity
4. Consider using OPTIONAL MATCH for parts that might not exist
5. For aggregation, use efficient aggregation functions (count, sum, avg)
6. For pathfinding, consider using shortestPath() or apoc.algo.* procedures

ERROR PREVENTION:
1. Validate your query steps mentally before finalizing
2. Ensure relationship directions match schema
3. Check property names match exactly what's in the schema
4. Use pattern variables consistently throughout the query
5. If previous attempts failed, analyze the failures and adjust your approach

Node schemas:
- EntityType
Properties: description, ontology_valid, name, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents the categories or classifications for entities in the database.

- Entity
Properties: description, ontology_valid, name, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents individual entities that belong to a specific type or classification.

- TextDocument
Properties: raw_data_location, name, mime_type, external_metadata, created_at, type, version, topological_rank, updated_at, metadata, id
Purpose: Represents documents containing text data, along with metadata about their storage and format.

- DocumentChunk
Properties: version, created_at, type, topological_rank, cut_type, text, metadata, chunk_index, chunk_size, updated_at, id
Purpose: Represents segmented portions of larger documents, useful for processing or analysis at a more granular level.

- TextSummary
Properties: topological_rank, metadata, id, type, updated_at, created_at, text, version
Purpose: Represents summarized content generated from larger text documents, retaining essential information and metadata.

Edge schema (relationship properties):
{edge_schemas}

This queries doesn't work. Do NOT use them:
{previous_attempts or "None"}

Example 1:
Get all nodes connected to John
MATCH (n:Entity {{'name': 'John'}})--(neighbor)
RETURN n, neighbor
        """

        return await llm_client.acreate_structured_output(
            text_input=text_input,
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
                    query, node_schemas, edge_schemas, previous_attempts
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
