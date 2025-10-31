"""Tool for transforming data into a structured knowledge graph."""

import sys
import asyncio
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger, get_log_file_location

from src.shared import context
from .utils import load_class

logger = get_logger()


async def cognify(
    data: str, graph_model_file: str = None, graph_model_name: str = None, custom_prompt: str = None
) -> list:
    """
    Transform ingested data into a structured knowledge graph.

    This is the core processing step in Cognee that converts raw text and documents
    into an intelligent knowledge graph. It analyzes content, extracts entities and
    relationships, and creates semantic connections for enhanced search and reasoning.

    Prerequisites:
        - **LLM_API_KEY**: Must be configured (required for entity extraction and graph generation)
        - **Data Added**: Must have data previously added via `cognee.add()`
        - **Vector Database**: Must be accessible for embeddings storage
        - **Graph Database**: Must be accessible for relationship storage

    Input Requirements:
        - **Content Types**: Works with any text-extractable content including:
            * Natural language documents
            * Structured data (CSV, JSON)
            * Code repositories
            * Academic papers and technical documentation
            * Mixed multimedia content (with text extraction)

    Processing Pipeline:
        1. **Document Classification**: Identifies document types and structures
        2. **Permission Validation**: Ensures user has processing rights
        3. **Text Chunking**: Breaks content into semantically meaningful segments
        4. **Entity Extraction**: Identifies key concepts, people, places, organizations
        5. **Relationship Detection**: Discovers connections between entities
        6. **Graph Construction**: Builds semantic knowledge graph with embeddings
        7. **Content Summarization**: Creates hierarchical summaries for navigation

    Parameters
    ----------
    data : str
        The data to be processed and transformed into structured knowledge.
        This can include natural language, file location, or any text-based information
        that should become part of the agent's memory.

    graph_model_file : str, optional
        Path to a custom schema file that defines the structure of the generated knowledge graph.
        If provided, this file will be loaded using importlib to create a custom graph model.
        Default is None, which uses Cognee's built-in KnowledgeGraph model.

    graph_model_name : str, optional
        Name of the class within the graph_model_file to instantiate as the graph model.
        Required if graph_model_file is specified.
        Default is None, which uses the default KnowledgeGraph class.

    custom_prompt : str, optional
        Custom prompt string to use for entity extraction and graph generation.
        If provided, this prompt will be used instead of the default prompts for
        knowledge graph extraction. The prompt should guide the LLM on how to
        extract entities and relationships from the text content.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.

    Next Steps:
        After successful cognify processing, use search functions to query the knowledge:

        ```python
        import cognee
        from cognee import SearchType

        # Process your data into knowledge graph
        await cognee.cognify()

        # Query for insights using different search types:

        # 1. Natural language completion with graph context
        insights = await cognee.search(
            "What are the main themes?",
            query_type=SearchType.GRAPH_COMPLETION
        )

        # 2. Get entity relationships and connections
        relationships = await cognee.search(
            "connections between concepts",
            query_type=SearchType.GRAPH_COMPLETION
        )

        # 3. Find relevant document chunks
        chunks = await cognee.search(
            "specific topic",
            query_type=SearchType.CHUNKS
        )
        ```

    Environment Variables:
        Required:
        - LLM_API_KEY: API key for your LLM provider

        Optional:
        - LLM_PROVIDER, LLM_MODEL, VECTOR_DB_PROVIDER, GRAPH_DATABASE_PROVIDER
        - LLM_RATE_LIMIT_ENABLED: Enable rate limiting (default: False)
        - LLM_RATE_LIMIT_REQUESTS: Max requests per interval (default: 60)

    Notes
    -----
    - The function launches a background task and returns immediately
    - The actual cognify process may take significant time depending on text length
    - Use the cognify_status tool to check the progress of the operation

    """

    async def cognify_task(
        data: str,
        graph_model_file: str = None,
        graph_model_name: str = None,
        custom_prompt: str = None,
    ) -> str:
        """Build knowledge graph from the input text"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Cognify process starting.")

            graph_model = None
            if graph_model_file and graph_model_name:
                if context.cognee_client.use_api:
                    logger.warning("Custom graph models are not supported in API mode, ignoring.")
                else:
                    from cognee.shared.data_models import KnowledgeGraph

                    graph_model = load_class(graph_model_file, graph_model_name)

            await context.cognee_client.add(data)

            try:
                await context.cognee_client.cognify(
                    custom_prompt=custom_prompt, graph_model=graph_model
                )
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    asyncio.create_task(
        cognify_task(
            data=data,
            graph_model_file=graph_model_file,
            graph_model_name=graph_model_name,
            custom_prompt=custom_prompt,
        )
    )

    log_file = get_log_file_location()
    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current cognify status use the cognify_status tool\n"
        f"or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]
