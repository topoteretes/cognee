"""Tool for searching and querying the knowledge graph."""

import sys
import json
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import JSONEncoder

from src.shared import context
from .utils import retrieved_edges_to_string

logger = get_logger()


async def search(search_query: str, search_type: str) -> list:
    """
    Search and query the knowledge graph for insights, information, and connections.

    This is the final step in the Cognee workflow that retrieves information from the
    processed knowledge graph. It supports multiple search modes optimized for different
    use cases - from simple fact retrieval to complex reasoning and code analysis.

    Search Prerequisites:
        - **LLM_API_KEY**: Required for GRAPH_COMPLETION and RAG_COMPLETION search types
        - **Data Added**: Must have data previously added via `cognee.add()`
        - **Knowledge Graph Built**: Must have processed data via `cognee.cognify()`
        - **Vector Database**: Must be accessible for semantic search functionality

    Search Types & Use Cases:

        **GRAPH_COMPLETION** (Recommended):
            Natural language Q&A using full graph context and LLM reasoning.
            Best for: Complex questions, analysis, summaries, insights.
            Returns: Conversational AI responses with graph-backed context.

        **RAG_COMPLETION**:
            Traditional RAG using document chunks without graph structure.
            Best for: Direct document retrieval, specific fact-finding.
            Returns: LLM responses based on relevant text chunks.

        **CHUNKS**:
            Raw text segments that match the query semantically.
            Best for: Finding specific passages, citations, exact content.
            Returns: Ranked list of relevant text chunks with metadata.

        **SUMMARIES**:
            Pre-generated hierarchical summaries of content.
            Best for: Quick overviews, document abstracts, topic summaries.
            Returns: Multi-level summaries from detailed to high-level.

        **CODE**:
            Code-specific search with syntax and semantic understanding.
            Best for: Finding functions, classes, implementation patterns.
            Returns: Structured code information with context and relationships.

        **CYPHER**:
            Direct graph database queries using Cypher syntax.
            Best for: Advanced users, specific graph traversals, debugging.
            Returns: Raw graph query results.

        **FEELING_LUCKY**:
            Intelligently selects and runs the most appropriate search type.
            Best for: General-purpose queries or when you're unsure which search type is best.
            Returns: The results from the automatically selected search type.

    Parameters
    ----------
    search_query : str
        Your question or search query in natural language.
        Examples:
        - "What are the main themes in this research?"
        - "How do these concepts relate to each other?"
        - "Find information about machine learning algorithms"
        - "What functions handle user authentication?"

    search_type : str
        The type of search to perform. Valid options include:
        - "GRAPH_COMPLETION": Returns an LLM response based on the search query and Cognee's memory
        - "RAG_COMPLETION": Returns an LLM response based on the search query and standard RAG data
        - "CODE": Returns code-related knowledge in JSON format
        - "CHUNKS": Returns raw text chunks from the knowledge graph
        - "SUMMARIES": Returns pre-generated hierarchical summaries
        - "CYPHER": Direct graph database queries
        - "FEELING_LUCKY": Automatically selects best search type

        The search_type is case-insensitive and will be converted to uppercase.

    Returns
    -------
    list
        A list containing a single TextContent object with the search results.
        The format of the result depends on the search_type:
        - **GRAPH_COMPLETION/RAG_COMPLETION**: Conversational AI response strings
        - **CHUNKS**: Relevant text passages with source metadata
        - **SUMMARIES**: Hierarchical summaries from general to specific
        - **CODE**: Structured code information with context
        - **FEELING_LUCKY**: Results in format of automatically selected search type
        - **CYPHER**: Raw graph query results

    Performance & Optimization:
        - **GRAPH_COMPLETION**: Slower but most intelligent, uses LLM + graph context
        - **RAG_COMPLETION**: Medium speed, uses LLM + document chunks (no graph traversal)
        - **CHUNKS**: Fastest, pure vector similarity search without LLM
        - **SUMMARIES**: Fast, returns pre-computed summaries
        - **CODE**: Medium speed, specialized for code understanding
        - **FEELING_LUCKY**: Variable speed, uses LLM + search type selection intelligently

    Environment Variables:
        Required for LLM-based search types (GRAPH_COMPLETION, RAG_COMPLETION):
        - LLM_API_KEY: API key for your LLM provider

        Optional:
        - LLM_PROVIDER, LLM_MODEL: Configure LLM for search responses
        - VECTOR_DB_PROVIDER: Must match what was used during cognify
        - GRAPH_DATABASE_PROVIDER: Must match what was used during cognify

    Notes
    -----
    - Different search types produce different output formats
    - The function handles the conversion between Cognee's internal result format and MCP's output format

    """

    async def search_task(search_query: str, search_type: str) -> str:
        """Search the knowledge graph"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            search_results = await context.cognee_client.search(
                query_text=search_query, query_type=search_type
            )

            # Handle different result formats based on API vs direct mode
            if context.cognee_client.use_api:
                # API mode returns JSON-serialized results
                if isinstance(search_results, str):
                    return search_results
                elif isinstance(search_results, list):
                    if (
                        search_type.upper() in ["GRAPH_COMPLETION", "RAG_COMPLETION"]
                        and len(search_results) > 0
                    ):
                        return str(search_results[0])
                    return str(search_results)
                else:
                    return json.dumps(search_results, cls=JSONEncoder)
            else:
                # Direct mode processing
                if search_type.upper() == "CODE":
                    return json.dumps(search_results, cls=JSONEncoder)
                elif (
                    search_type.upper() == "GRAPH_COMPLETION"
                    or search_type.upper() == "RAG_COMPLETION"
                ):
                    return str(search_results[0])
                elif search_type.upper() == "CHUNKS":
                    return str(search_results)
                elif search_type.upper() == "INSIGHTS":
                    results = retrieved_edges_to_string(search_results)
                    return results
                else:
                    return str(search_results)

    search_results = await search_task(search_query, search_type)
    return [types.TextContent(type="text", text=search_results)]
