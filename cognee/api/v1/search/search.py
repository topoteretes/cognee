from uuid import UUID
from typing import Union, Optional, List, Type

from cognee.modules.users.models import User
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.modules.search.methods import search as search_function
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.exceptions import DatasetNotFoundError


async def search(
    query_text: str,
    query_type: SearchType = SearchType.GRAPH_COMPLETION,
    user: User = None,
    datasets: Optional[Union[list[str], str]] = None,
    dataset_ids: Optional[Union[list[UUID], UUID]] = None,
    system_prompt_path: str = "answer_simple_question.txt",
    top_k: int = 10,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> list:
    """
    Search and query the knowledge graph for insights, information, and connections.

    This is the final step in the Cognee workflow that retrieves information from the
    processed knowledge graph. It supports multiple search modes optimized for different
    use cases - from simple fact retrieval to complex reasoning and code analysis.

    Search Prerequisites:
        - **LLM_API_KEY**: Required for GRAPH_COMPLETION and RAG_COMPLETION search types
        - **Data Added**: Must have data previously added via `cognee.add()`
        - **Knowledge Graph Built**: Must have processed data via `cognee.cognify()`
        - **Dataset Permissions**: User must have 'read' permission on target datasets
        - **Vector Database**: Must be accessible for semantic search functionality

    Search Types & Use Cases:

        **GRAPH_COMPLETION** (Default - Recommended):
            Natural language Q&A using full graph context and LLM reasoning.
            Best for: Complex questions, analysis, summaries, insights.
            Returns: Conversational AI responses with graph-backed context.

        **RAG_COMPLETION**:
            Traditional RAG using document chunks without graph structure.
            Best for: Direct document retrieval, specific fact-finding.
            Returns: LLM responses based on relevant text chunks.

        **INSIGHTS**:
            Structured entity relationships and semantic connections.
            Best for: Understanding concept relationships, knowledge mapping.
            Returns: Formatted relationship data and entity connections.

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

    Args:
        query_text: Your question or search query in natural language.
            Examples:
            - "What are the main themes in this research?"
            - "How do these concepts relate to each other?"
            - "Find information about machine learning algorithms"
            - "What functions handle user authentication?"

        query_type: SearchType enum specifying the search mode.
                   Defaults to GRAPH_COMPLETION for conversational AI responses.

        user: User context for data access permissions. Uses default if None.

        datasets: Dataset name(s) to search within. Searches all accessible if None.
            - Single dataset: "research_papers"
            - Multiple datasets: ["docs", "reports", "analysis"]
            - None: Search across all user datasets

        dataset_ids: Alternative to datasets - use specific UUID identifiers.

        system_prompt_path: Custom system prompt file for LLM-based search types.
                          Defaults to "answer_simple_question.txt".

        top_k: Maximum number of results to return (1-N)
              Higher values provide more comprehensive but potentially noisy results.

        node_type: Filter results to specific entity types (for advanced filtering).

        node_name: Filter results to specific named entities (for targeted search).

    Returns:
        list: Search results in format determined by query_type:

            **GRAPH_COMPLETION/RAG_COMPLETION**:
                [List of conversational AI response strings]

            **INSIGHTS**:
                [List of formatted relationship descriptions and entity connections]

            **CHUNKS**:
                [List of relevant text passages with source metadata]

            **SUMMARIES**:
                [List of hierarchical summaries from general to specific]

            **CODE**:
                [List of structured code information with context]





    Performance & Optimization:
        - **GRAPH_COMPLETION**: Slower but most intelligent, uses LLM + graph context
        - **RAG_COMPLETION**: Medium speed, uses LLM + document chunks (no graph traversal)
        - **INSIGHTS**: Fast, returns structured relationships without LLM processing
        - **CHUNKS**: Fastest, pure vector similarity search without LLM
        - **SUMMARIES**: Fast, returns pre-computed summaries
        - **CODE**: Medium speed, specialized for code understanding
        - **top_k**: Start with 10, increase for comprehensive analysis (max 100)
        - **datasets**: Specify datasets to improve speed and relevance

    Next Steps After Search:
        - Use results for further analysis or application integration
        - Combine different search types for comprehensive understanding
        - Export insights for reporting or downstream processing
        - Iterate with refined queries based on initial results

    Environment Variables:
        Required for LLM-based search types (GRAPH_COMPLETION, RAG_COMPLETION):
        - LLM_API_KEY: API key for your LLM provider

        Optional:
        - LLM_PROVIDER, LLM_MODEL: Configure LLM for search responses
        - VECTOR_DB_PROVIDER: Must match what was used during cognify
        - GRAPH_DATABASE_PROVIDER: Must match what was used during cognify

    Raises:
        DatasetNotFoundError: If specified datasets don't exist or aren't accessible
        PermissionDeniedError: If user lacks read access to requested datasets
        NoDataError: If no relevant data found for the search query
        InvalidValueError: If LLM_API_KEY is not set (for LLM-based search types)
        ValueError: If query_text is empty or search parameters are invalid
        CollectionNotFoundError: If vector collection not found (data not processed)
    """
    # We use lists from now on for datasets
    if isinstance(datasets, UUID) or isinstance(datasets, str):
        datasets = [datasets]

    if user is None:
        user = await get_default_user()

    # Transform string based datasets to UUID - String based datasets can only be found for current user
    if datasets is not None and [all(isinstance(dataset, str) for dataset in datasets)]:
        datasets = await get_authorized_existing_datasets(datasets, "read", user)
        datasets = [dataset.id for dataset in datasets]
        if not datasets:
            raise DatasetNotFoundError(message="No datasets found.")

    filtered_search_results = await search_function(
        query_text=query_text,
        query_type=query_type,
        dataset_ids=dataset_ids if dataset_ids else datasets,
        user=user,
        system_prompt_path=system_prompt_path,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
    )

    return filtered_search_results
