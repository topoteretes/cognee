import json
import os
import sys
import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from cognee.shared.logging_utils import get_logger, setup_logging, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.storage.utils import JSONEncoder
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .cognee_client import CogneeClient
except ImportError:
    from cognee_client import CogneeClient


try:
    from cognee.tasks.codingagents.coding_rule_associations import (
        add_rule_associations,
        get_existing_rules,
    )
except ModuleNotFoundError:
    from .codingagents.coding_rule_associations import (
        add_rule_associations,
        get_existing_rules,
    )


mcp = FastMCP("Cognee")

logger = get_logger()

cognee_client: Optional[CogneeClient] = None


async def run_sse_with_cors():
    """Custom SSE transport with CORS middleware."""
    sse_app = mcp.sse_app()
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        sse_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http_with_cors():
    """Custom HTTP transport with CORS middleware."""
    http_app = mcp.streamable_http_app()
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        http_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


@mcp.tool()
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
                if cognee_client.use_api:
                    logger.warning("Custom graph models are not supported in API mode, ignoring.")
                else:
                    from cognee.shared.data_models import KnowledgeGraph

                    graph_model = load_class(graph_model_file, graph_model_name)

            await cognee_client.add(data)

            try:
                await cognee_client.cognify(custom_prompt=custom_prompt, graph_model=graph_model)
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


@mcp.tool(
    name="save_interaction", description="Logs user-agent interactions and query-answer pairs"
)
async def save_interaction(data: str) -> list:
    """
    Transform and save a user-agent interaction into structured knowledge.

    Parameters
    ----------
    data : str
        The input string containing user queries and corresponding agent answers.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the background task launch.
    """

    async def save_user_agent_interaction(data: str) -> None:
        """Build knowledge graph from the interaction data"""
        with redirect_stdout(sys.stderr):
            logger.info("Save interaction process starting.")

            await cognee_client.add(data, node_set=["user_agent_interaction"])

            try:
                await cognee_client.cognify()
                logger.info("Save interaction process finished.")

                # Rule associations only work in direct mode
                if not cognee_client.use_api:
                    logger.info("Generating associated rules from interaction data.")
                    await add_rule_associations(data=data, rules_nodeset_name="coding_agent_rules")
                    logger.info("Associated rules generated from interaction data.")
                else:
                    logger.warning("Rule associations are not available in API mode, skipping.")

            except Exception as e:
                logger.error("Save interaction process failed.")
                raise ValueError(f"Failed to Save interaction: {str(e)}")

    asyncio.create_task(
        save_user_agent_interaction(
            data=data,
        )
    )

    log_file = get_log_file_location()
    text = (
        f"Background process launched to process the user-agent interaction.\n"
        f"To check the current status, use the cognify_status tool or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def search(search_query: str, search_type: str, top_k: int = 5) -> list:
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

    top_k : int, optional
        Maximum number of results to return (default: 5).
        Controls the amount of context retrieved from the knowledge graph.
        - Lower values (3-5): Faster, more focused results
        - Higher values (10-20): More comprehensive, but slower and more context-heavy
        Helps manage response size and context window usage in MCP clients.

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

    async def search_task(search_query: str, search_type: str, top_k: int) -> str:
        """
        Internal task to execute knowledge graph search with result formatting.

        Handles the actual search execution and formats results appropriately
        for MCP clients based on the search type and execution mode (API vs direct).

        Parameters
        ----------
        search_query : str
            The search query in natural language
        search_type : str
            Type of search to perform (GRAPH_COMPLETION, CHUNKS, etc.)
        top_k : int
            Maximum number of results to return

        Returns
        -------
        str
            Formatted search results as a string, with format depending on search_type
        """
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            search_results = await cognee_client.search(
                query_text=search_query, query_type=search_type, top_k=top_k
            )

            # Handle different result formats based on API vs direct mode
            if cognee_client.use_api:
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

    search_results = await search_task(search_query, search_type, top_k)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
async def list_data(dataset_id: str = None) -> list:
    """
    List all datasets and their data items with IDs for deletion operations.

    This function helps users identify data IDs and dataset IDs that can be used
    with the delete tool. It provides a comprehensive view of available data.

    Parameters
    ----------
    dataset_id : str, optional
        If provided, only list data items from this specific dataset.
        If None, lists all datasets and their data items.
        Should be a valid UUID string.

    Returns
    -------
    list
        A list containing a single TextContent object with formatted information
        about datasets and data items, including their IDs for deletion.

    Notes
    -----
    - Use this tool to identify data_id and dataset_id values for the delete tool
    - The output includes both dataset information and individual data items
    - UUIDs are displayed in a format ready for use with other tools
    """
    from uuid import UUID

    with redirect_stdout(sys.stderr):
        try:
            output_lines = []

            if dataset_id:
                # Detailed data listing for specific dataset is only available in direct mode
                if cognee_client.use_api:
                    return [
                        types.TextContent(
                            type="text",
                            text="❌ Detailed data listing for specific datasets is not available in API mode.\nPlease use the API directly or use direct mode.",
                        )
                    ]

                from cognee.modules.users.methods import get_default_user
                from cognee.modules.data.methods import get_dataset, get_dataset_data

                logger.info(f"Listing data for dataset: {dataset_id}")
                dataset_uuid = UUID(dataset_id)
                user = await get_default_user()

                dataset = await get_dataset(user.id, dataset_uuid)

                if not dataset:
                    return [
                        types.TextContent(type="text", text=f"❌ Dataset not found: {dataset_id}")
                    ]

                # Get data items in the dataset
                data_items = await get_dataset_data(dataset.id)

                output_lines.append(f"📁 Dataset: {dataset.name}")
                output_lines.append(f"   ID: {dataset.id}")
                output_lines.append(f"   Created: {dataset.created_at}")
                output_lines.append(f"   Data items: {len(data_items)}")
                output_lines.append("")

                if data_items:
                    for i, data_item in enumerate(data_items, 1):
                        output_lines.append(f"   📄 Data item #{i}:")
                        output_lines.append(f"      Data ID: {data_item.id}")
                        output_lines.append(f"      Name: {data_item.name or 'Unnamed'}")
                        output_lines.append(f"      Created: {data_item.created_at}")
                        output_lines.append("")
                else:
                    output_lines.append("   (No data items in this dataset)")

            else:
                # List all datasets - works in both modes
                logger.info("Listing all datasets")
                datasets = await cognee_client.list_datasets()

                if not datasets:
                    return [
                        types.TextContent(
                            type="text",
                            text="📂 No datasets found.\nUse the cognify tool to create your first dataset!",
                        )
                    ]

                output_lines.append("📂 Available Datasets:")
                output_lines.append("=" * 50)
                output_lines.append("")

                for i, dataset in enumerate(datasets, 1):
                    # In API mode, dataset is a dict; in direct mode, it's formatted as dict
                    if isinstance(dataset, dict):
                        output_lines.append(f"{i}. 📁 {dataset.get('name', 'Unnamed')}")
                        output_lines.append(f"   Dataset ID: {dataset.get('id')}")
                        output_lines.append(f"   Created: {dataset.get('created_at', 'N/A')}")
                    else:
                        output_lines.append(f"{i}. 📁 {dataset.name}")
                        output_lines.append(f"   Dataset ID: {dataset.id}")
                        output_lines.append(f"   Created: {dataset.created_at}")
                    output_lines.append("")

                if not cognee_client.use_api:
                    output_lines.append("💡 To see data items in a specific dataset, use:")
                    output_lines.append('   list_data(dataset_id="your-dataset-id-here")')
                    output_lines.append("")
                output_lines.append("🗑️  To delete specific data, use:")
                output_lines.append('   delete(data_id="data-id", dataset_id="dataset-id")')

            result_text = "\n".join(output_lines)
            logger.info("List data operation completed successfully")

            return [types.TextContent(type="text", text=result_text)]

        except ValueError as e:
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            error_msg = f"❌ Failed to list data: {str(e)}"
            logger.error(f"List data error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def delete(data_id: str, dataset_id: str, mode: str = "soft") -> list:
    """
    Delete specific data from a dataset in the Cognee knowledge graph.

    This function removes a specific data item from a dataset while keeping the
    dataset itself intact. It supports both soft and hard deletion modes.

    Parameters
    ----------
    data_id : str
        The UUID of the data item to delete from the knowledge graph.
        This should be a valid UUID string identifying the specific data item.

    dataset_id : str
        The UUID of the dataset containing the data to be deleted.
        This should be a valid UUID string identifying the dataset.

    mode : str, optional
        The deletion mode to use. Options are:
        - "soft" (default): Removes the data but keeps related entities that might be shared
        - "hard": Also removes degree-one entity nodes that become orphaned after deletion
        Default is "soft" for safer deletion that preserves shared knowledge.

    Returns
    -------
    list
        A list containing a single TextContent object with the deletion results,
        including status, deleted node counts, and confirmation details.

    Notes
    -----
    - This operation cannot be undone. The specified data will be permanently removed.
    - Hard mode may remove additional entity nodes that become orphaned
    - The function provides detailed feedback about what was deleted
    - Use this for targeted deletion instead of the prune tool which removes everything
    """
    from uuid import UUID

    with redirect_stdout(sys.stderr):
        try:
            logger.info(
                f"Starting delete operation for data_id: {data_id}, dataset_id: {dataset_id}, mode: {mode}"
            )

            # Convert string UUIDs to UUID objects
            data_uuid = UUID(data_id)
            dataset_uuid = UUID(dataset_id)

            # Call the cognee delete function via client
            result = await cognee_client.delete(
                data_id=data_uuid, dataset_id=dataset_uuid, mode=mode
            )

            logger.info(f"Delete operation completed successfully: {result}")

            # Format the result for MCP response
            formatted_result = json.dumps(result, indent=2, cls=JSONEncoder)

            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Delete operation completed successfully!\n\n{formatted_result}",
                )
            ]

        except ValueError as e:
            # Handle UUID parsing errors
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            # Handle all other errors (DocumentNotFoundError, DatasetNotFoundError, etc.)
            error_msg = f"❌ Delete operation failed: {str(e)}"
            logger.error(f"Delete operation error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def prune():
    """
    Reset the Cognee knowledge graph by removing all stored information.

    This function performs a complete reset of both the data layer and system layer
    of the Cognee knowledge graph, removing all nodes, edges, and associated metadata.
    It is typically used during development or when needing to start fresh with a new
    knowledge base.

    Returns
    -------
    list
        A list containing a single TextContent object with confirmation of the prune operation.

    Notes
    -----
    - This operation cannot be undone. All memory data will be permanently deleted.
    - The function prunes both data content (using prune_data) and system metadata (using prune_system)
    - This operation is not available in API mode
    """
    with redirect_stdout(sys.stderr):
        try:
            await cognee_client.prune_data()
            await cognee_client.prune_system(metadata=True)
            return [types.TextContent(type="text", text="Pruned")]
        except NotImplementedError:
            error_msg = "❌ Prune operation is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Prune operation failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def cognify_status():
    """
    Get the current status of the cognify pipeline.

    This function retrieves information about current and recently completed cognify operations
    in the main_dataset. It provides details on progress, success/failure status, and statistics
    about the processed data.

    Returns
    -------
    list
        A list containing a single TextContent object with the status information as a string.
        The status includes information about active and completed jobs for the cognify_pipeline.

    Notes
    -----
    - The function retrieves pipeline status specifically for the "cognify_pipeline" on the "main_dataset"
    - Status information includes job progress, execution time, and completion status
    - The status is returned in string format for easy reading
    - This operation is not available in API mode
    """
    with redirect_stdout(sys.stderr):
        try:
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()
            status = await cognee_client.get_pipeline_status(
                [await get_unique_dataset_id("main_dataset", user)], "cognify_pipeline"
            )
            return [types.TextContent(type="text", text=str(status))]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get cognify status: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


def node_to_string(node):
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )

    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    edge_strings = []
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship_type = edge["relationship_name"]
        edge_str = f"{node_to_string(node1)} {relationship_type} {node_to_string(node2)}"
        edge_strings.append(edge_str)

    return "\n".join(edge_strings)


def load_class(model_file, model_name):
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, model_name)

    return model_class


async def main():
    global cognee_client

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    # HTTP transport options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (default: 8000)",
    )

    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for the MCP HTTP endpoint (default: /mcp)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the HTTP server (default: info)",
    )

    parser.add_argument(
        "--no-migration",
        default=False,
        action="store_true",
        help="Argument stops database migration from being attempted",
    )

    # Cognee API connection options
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of a running Cognee FastAPI server (e.g., http://localhost:8000). "
        "If provided, the MCP server will connect to the API instead of using cognee directly.",
    )

    parser.add_argument(
        "--api-token",
        default=None,
        help="Authentication token for the API (optional, required if API has authentication enabled).",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Skip migrations when in API mode (the API server handles its own database)
    if not args.no_migration and not args.api_url:
        from cognee.modules.engine.operations.setup import setup

        await setup()

        # Run Alembic migrations from the main cognee directory where alembic.ini is located
        logger.info("Running database migrations...")
        migration_result = subprocess.run(
            ["python", "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )

        if migration_result.returncode != 0:
            migration_output = migration_result.stderr + migration_result.stdout
            # Check for the expected UserAlreadyExists error (which is not critical)
            if (
                "UserAlreadyExists" in migration_output
                or "User default_user@example.com already exists" in migration_output
            ):
                logger.warning("Warning: Default user already exists, continuing startup...")
            else:
                logger.error(f"Migration failed with unexpected error: {migration_output}")
                sys.exit(1)

        logger.info("Database migrations done.")
    elif args.api_url:
        logger.info("Skipping database migrations (using API mode)")

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(f"Running MCP server with SSE transport on {args.host}:{args.port}")
        await run_sse_with_cors()
    elif args.transport == "http":
        logger.info(
            f"Running MCP server with Streamable HTTP transport on {args.host}:{args.port}{args.path}"
        )
        await run_http_with_cors()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
