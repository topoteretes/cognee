from typing import Any, Optional

from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client

import logging

logger = logging.getLogger(__name__)


class CypherQueryRetriever(BaseRetriever):
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.llm_client = get_llm_client()

    async def get_context(self, query: str) -> Optional[Any]:
        graph_engine = await get_graph_engine()
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
        previous_attempt_context = None
        for attempt in range(self.max_attempts):
            llm_client = get_llm_client()
            system_prompt = f"""
    You are an advanced Cypher query generator for Neo4j. Your task:
    - Based on the following graph schema (nodes and edges) and a natural language query, generate a valid Cypher query that retrieves the requested information.
    - Provide only the Cypher query, no explanations or additional formatting.
    - Take into account the previous failed attempts and adjust the query accordingly.
    
    Return only a valid, optimized Cypher query.
    """

            cypher_query = await llm_client.acreate_structured_output(
                text_input=f"""
    User query:
    {query}

    Node schema (labels and properties):
    {node_schemas}

    Edge schema (relationship properties):
    {edge_schemas}

    Previous attempts:
    {previous_attempt_context}
                """,
                system_prompt=system_prompt,
                response_model=str,
            )
            try:
                context = await graph_engine.query(cypher_query)
                if context:
                    return context
                else:
                    previous_attempt_context += f"Query: {cypher_query} -> Result: None\n"
                    continue
            except Exception as e:
                previous_attempt_context += f"Query: {cypher_query} -> Executed with error: {e}\n"
        return None

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        if context is None:
            context = await self.get_context(query)

        return context
