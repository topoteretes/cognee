import json
import os
import re
import sys
import argparse
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from cognee.modules.data.methods.get_datasets_by_name import get_datasets_by_name
from cognee.modules.data.methods.get_last_added_data import get_last_added_data
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger, setup_logging, get_log_file_location
from cognee.shared.usage_logger import log_usage
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
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
    from .strip_vectors import strip_vectors
except ImportError:
    from strip_vectors import strip_vectors


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

# Stores background task errors keyed by dataset_name so cognify_status can report them
_task_errors: dict[str, list[tuple[str, str]]] = {}


def _configure_transport_security(host: str) -> None:
    """Configure MCP transport security based on env vars and bind host.

    Must be called before run_sse_with_cors() or run_http_with_cors(), since
    the SDK reads mcp.settings.transport_security lazily when creating the app.

    Env vars:
        MCP_DISABLE_DNS_REBINDING_PROTECTION: Set to "true" to disable all
            Host/Origin header validation. Useful for LAN or Docker deployments.
        MCP_ALLOWED_HOSTS: Comma-separated additional Host header patterns
            (e.g. "192.168.1.50:*,myserver.local:*"). Appended to the
            localhost defaults. Requires the ":*" port glob suffix.
    """
    disable = os.getenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", "false").lower() == "true"

    if disable:
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
        logger.info("MCP transport security: DNS rebinding protection disabled")
        return

    extra_hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]

    # The SDK only auto-populates localhost defaults when transport_security is
    # None AND host is a loopback address. When the user binds to 0.0.0.0 or a
    # LAN IP, we must provide the full allowed list ourselves.
    localhost_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    localhost_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]

    allowed_hosts = localhost_hosts + extra_hosts
    # Derive origins from extra hosts so users don't need to set both.
    allowed_origins = localhost_origins + [f"http://{h}" for h in extra_hosts]

    if host not in ("127.0.0.1", "localhost", "::1") or extra_hosts:
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        logger.info(
            "MCP transport security: allowed_hosts=%s",
            allowed_hosts,
        )
    else:
        # Loopback-only with no extra hosts — let the SDK use its own defaults.
        logger.info("MCP transport security: using SDK defaults (localhost only)")


def _is_running_in_docker() -> bool:
    """Check if the process is running inside a Docker container."""
    return os.path.exists("/.dockerenv") or os.path.isdir("/app")


def _looks_like_file_path(data: str) -> bool:
    """Check if the data string looks like a local file path."""
    data = data.strip()
    # Unix absolute path, Windows drive letter path, or file:// URI
    if data.startswith("/") or re.match(r"^[A-Za-z]:\\", data) or data.startswith("file://"):
        return True
    return False


def _validate_file_path(data: str) -> Optional[str]:
    """
    If data looks like a file path, validate it exists.
    Returns an error message string if invalid, or None if OK.
    """
    if not _looks_like_file_path(data):
        return None

    path = data.strip()
    if path.startswith("file://"):
        path = path[7:]

    if not os.path.exists(path):
        msg = f"File not found: {path}"
        if _is_running_in_docker():
            msg += (
                "\n\nIt looks like you're running inside Docker. Host file paths are not "
                "accessible inside the container. To ingest local files, mount a volume in "
                "docker-compose.yml:\n"
                "  volumes:\n"
                "    - /path/to/your/data:/data\n"
                "Then reference the file as /data/<filename> instead."
            )
        return msg
    return None


def _get_cors_origins() -> list[str]:
    """Parse CORS allowed origins from MCP_CORS_ALLOW_ORIGINS env var."""
    raw = os.getenv("MCP_CORS_ALLOW_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


async def run_sse_with_cors():
    """Custom SSE transport with CORS middleware."""
    sse_app = mcp.sse_app()
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
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
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
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
@log_usage(function_name="MCP cognify", log_type="mcp_tool")
async def cognify(
    data: str,
    dataset_name: str = "main_dataset",
    graph_model_file: str = None,
    graph_model_name: str = None,
    custom_prompt: str = None,
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

    # Validate file paths before launching background task
    file_error = _validate_file_path(data)
    if file_error:
        return [
            types.TextContent(
                type="text",
                text=f"Error: {file_error}",
            )
        ]

    async def cognify_task(
        data: str,
        dataset_name: str = "main_dataset",
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

            await cognee_client.add(data, dataset_name=dataset_name)

            try:
                await cognee_client.cognify(
                    datasets=[dataset_name], custom_prompt=custom_prompt, graph_model=graph_model
                )
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    async def cognify_task_wrapper(**kwargs):
        """Wrapper that captures errors from the background task."""
        try:
            await cognify_task(**kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset_name", "main_dataset")
            timestamp = datetime.now(timezone.utc).isoformat()
            _task_errors.setdefault(dataset, []).append((timestamp, str(e)))
            logger.error(f"Background cognify task failed for dataset '{dataset}': {e}")

    asyncio.create_task(
        cognify_task_wrapper(
            data=data,
            dataset_name=dataset_name,
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
@log_usage(function_name="MCP save_interaction", log_type="mcp_tool")
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

                user = await get_default_user()
                datasets = await get_datasets_by_name("main_dataset", user_id=user.id)
                dataset = datasets[0]
                added_data = await get_last_added_data(dataset.id)

                logger.info("Save interaction process finished.")

                # Rule associations only work in direct mode
                if not cognee_client.use_api:
                    logger.info("Generating associated rules from interaction data.")
                    await add_rule_associations(
                        data=data,
                        rules_nodeset_name="coding_agent_rules",
                        context={
                            "user": user,
                            "dataset": dataset,
                            "data": added_data,
                        },
                    )
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
@log_usage(function_name="MCP search", log_type="mcp_tool")
async def search(
    search_query: str, search_type: str, top_k: int = 10, datasets: str = None
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
        Maximum number of results to return (default: 10).
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

    async def search_task(
        search_query: str, search_type: str, top_k: int, datasets_list: list = None
    ) -> str:
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
                query_text=search_query,
                query_type=search_type,
                top_k=top_k,
                datasets=datasets_list,
            )

            # Strip embedding vectors from results to save LLM context
            # text_vector contains raw floats (~92KB per result), useless for clients
            search_results = strip_vectors(search_results)

            def _combine_completion_results(results):
                """Combine results from all datasets instead of returning only the first.

                Each result may be a dict (from _backwards_compatible_search_results)
                or a SearchResult object. Results are labeled with their dataset name
                so users can distinguish which dataset each answer came from.
                """
                if not isinstance(results, list) or len(results) == 0:
                    return str(results)
                combined = []
                for sr in results:
                    if isinstance(sr, dict):
                        ds_name = sr.get("dataset_name", "unknown")
                        sr_content = sr.get("search_result", str(sr))
                    elif hasattr(sr, "dataset_name") and hasattr(sr, "search_result"):
                        ds_name = sr.dataset_name or "unknown"
                        sr_content = sr.search_result
                    else:
                        combined.append(str(sr))
                        continue
                    if isinstance(sr_content, list):
                        for item in sr_content:
                            combined.append(f"[{ds_name}] {item}")
                    else:
                        combined.append(f"[{ds_name}] {sr_content}")
                return "\n\n".join(combined)

            # Handle different result formats based on API vs direct mode
            if cognee_client.use_api:
                # API mode returns JSON-serialized results
                if isinstance(search_results, str):
                    return search_results
                elif isinstance(search_results, list):
                    if search_type.upper() in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                        return _combine_completion_results(search_results)
                    return str(search_results)
                else:
                    return json.dumps(search_results, cls=JSONEncoder)
            else:
                # Direct mode processing
                if search_type.upper() == "CODE":
                    return json.dumps(search_results, cls=JSONEncoder)
                elif search_type.upper() in ("GRAPH_COMPLETION", "RAG_COMPLETION"):
                    return _combine_completion_results(search_results)
                elif search_type.upper() == "CHUNKS":
                    return str(search_results)
                elif search_type.upper() == "INSIGHTS":
                    results = retrieved_edges_to_string(search_results)
                    return results
                else:
                    return str(search_results)

    # Parse comma-separated datasets into list
    datasets_list = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else None
    datasets_list = datasets_list or None  # collapse empty list to None
    search_results = await search_task(search_query, search_type, top_k, datasets_list)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
@log_usage(function_name="MCP list_data", log_type="mcp_tool")
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
@log_usage(function_name="MCP delete_dataset", log_type="mcp_tool")
async def delete_dataset(dataset_name: str) -> list:
    """
    Delete an entire dataset and all its data from the knowledge graph.

    This removes the dataset completely: graph data, vector indices,
    and metadata in the relational database. This operation cannot be undone.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to delete (e.g. 'main_dataset').

    Returns
    -------
    list
        A list containing a TextContent with deletion status.
    """
    with redirect_stdout(sys.stderr):
        try:
            if cognee_client.use_api:
                return [
                    types.TextContent(
                        type="text",
                        text="❌ delete_dataset is not available in API mode. Use the API directly.",
                    )
                ]

            from cognee.modules.users.methods import get_default_user
            from cognee.modules.data.methods import delete_dataset as _delete_dataset
            from cognee.modules.data.methods import get_datasets

            user = await get_default_user()
            datasets = await get_datasets(user.id)
            matching = [ds for ds in datasets if ds.name == dataset_name]

            if not matching:
                return [types.TextContent(type="text", text=f"Dataset '{dataset_name}' not found.")]

            if len(matching) > 1:
                ids = ", ".join(str(ds.id) for ds in matching)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Multiple datasets named '{dataset_name}' found (IDs: {ids}). Please delete by ID instead.",
                    )
                ]

            await _delete_dataset(matching[0])
            return [
                types.TextContent(
                    type="text",
                    text=f"Dataset '{dataset_name}' deleted successfully. Graph, vectors, and metadata removed.",
                )
            ]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error deleting dataset: {str(e)}")]


@mcp.tool()
@log_usage(function_name="MCP delete", log_type="mcp_tool")
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
@log_usage(function_name="MCP prune", log_type="mcp_tool")
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


# ---------------------------------------------------------------------------
# Session-aware memory operations (remember, recall, forget, improve)
# ---------------------------------------------------------------------------


@mcp.tool()
@log_usage(function_name="MCP remember", log_type="mcp_tool")
async def remember(
    data: str,
    dataset_name: str = "main_dataset",
    session_id: str = None,
    custom_prompt: str = None,
) -> list:
    """Store data in memory.

    Two modes depending on whether session_id is provided:

    Without session_id (permanent memory): Runs the full add + cognify
    pipeline to ingest data and build the knowledge graph.

    With session_id (session memory): Stores the data in the session
    cache only. Fast, no entity extraction. Use improve() later to
    sync session content into the permanent graph.

    Parameters
    ----------
    data : str
        The data to store (text content).
    dataset_name : str
        Target dataset name (default: main_dataset).
    session_id : str, optional
        Session ID. When set, stores in session cache only.
    custom_prompt : str, optional
        Custom prompt for entity extraction (permanent mode only).
    """
    with redirect_stdout(sys.stderr):
        try:
            result = await cognee_client.remember(
                data=data,
                dataset_name=dataset_name,
                session_id=session_id,
                custom_prompt=custom_prompt,
            )
            status = result.get("status", "completed")
            if session_id:
                text = f"Stored in session cache (session_id={session_id}, status={status})."
            else:
                text = f"Stored permanently in knowledge graph (dataset={dataset_name}, status={status})."
            return [types.TextContent(type="text", text=text)]
        except Exception as e:
            error_msg = f"Remember failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


@mcp.tool()
@log_usage(function_name="MCP recall", log_type="mcp_tool")
async def recall(
    query: str,
    search_type: str = None,
    datasets: str = None,
    session_id: str = None,
    top_k: int = 10,
) -> list:
    """Search memory with auto-routing and session awareness.

    When session_id is provided without datasets or search_type,
    searches session cache first by keyword matching. Falls through
    to the permanent knowledge graph if no session results match.

    Auto-routing picks the best search strategy when search_type
    is not specified.

    Parameters
    ----------
    query : str
        Natural language query to search for.
    search_type : str, optional
        Override auto-routing. Options: GRAPH_COMPLETION,
        GRAPH_COMPLETION_COT, RAG_COMPLETION, CHUNKS, SUMMARIES,
        TEMPORAL, FEELING_LUCKY, etc.
    datasets : str, optional
        Comma-separated dataset names to search within.
    session_id : str, optional
        Session ID for session-first search.
    top_k : int
        Maximum results to return (default: 10).
    """
    with redirect_stdout(sys.stderr):
        try:
            dataset_list = [d.strip() for d in datasets.split(",")] if datasets else None
            results = await cognee_client.recall(
                query_text=query,
                search_type=search_type,
                datasets=dataset_list,
                session_id=session_id,
                top_k=top_k,
            )
            if not results:
                return [types.TextContent(type="text", text="No relevant results found.")]
            # Format results
            lines = []
            for r in results:
                if isinstance(r, dict):
                    source = r.get("_source", "")
                    text = r.get("answer", r.get("text", r.get("content", str(r))))
                    prefix = f"[{source}] " if source else ""
                    lines.append(f"{prefix}{text}")
                else:
                    lines.append(str(r))
            return [types.TextContent(type="text", text="\n\n".join(lines))]
        except Exception as e:
            error_msg = f"Recall failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


@mcp.tool()
@log_usage(function_name="MCP forget", log_type="mcp_tool")
async def forget_memory(
    dataset: str = None,
    everything: bool = False,
) -> list:
    """Delete data from memory.

    Can target a specific dataset or delete everything the user owns.
    Removes data from the relational DB, graph DB, and vector DB.

    Parameters
    ----------
    dataset : str, optional
        Dataset name to delete entirely.
    everything : bool
        If true, delete ALL data across all datasets.
    """
    with redirect_stdout(sys.stderr):
        try:
            if not dataset and not everything:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: Specify 'dataset' name or set 'everything' to true.",
                    )
                ]
            result = await cognee_client.forget(dataset=dataset, everything=everything)
            status = result.get("status", "unknown") if isinstance(result, dict) else "completed"
            if everything:
                text = f"All data deleted (status={status})."
            else:
                text = f"Dataset '{dataset}' deleted (status={status})."
            return [types.TextContent(type="text", text=text)]
        except Exception as e:
            error_msg = f"Forget failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


@mcp.tool()
@log_usage(function_name="MCP improve", log_type="mcp_tool")
async def improve(
    dataset_name: str = "main_dataset",
    session_ids: str = None,
) -> list:
    """Enrich the knowledge graph and bridge session data to the permanent graph.

    When session_ids is provided, runs a 4-stage pipeline:
    1. Apply feedback weights from session scores to graph nodes/edges
    2. Persist session Q&A text into the permanent knowledge graph
    3. Enrich graph with triplet embeddings (memify)
    4. Sync enriched graph knowledge back into session caches

    Without session_ids, only stage 3 runs (triplet enrichment).

    Parameters
    ----------
    dataset_name : str
        Dataset to process (default: main_dataset).
    session_ids : str, optional
        Comma-separated session IDs to bridge into the permanent graph.
    """
    with redirect_stdout(sys.stderr):
        try:
            session_list = [s.strip() for s in session_ids.split(",")] if session_ids else None
            result = await cognee_client.improve(
                dataset_name=dataset_name,
                session_ids=session_list,
            )
            status = result.get("status", "completed") if isinstance(result, dict) else "completed"
            if session_list:
                text = (
                    f"Improve completed (status={status}). "
                    f"Bridged {len(session_list)} session(s) into permanent graph."
                )
            else:
                text = f"Graph enrichment completed (status={status})."
            return [types.TextContent(type="text", text=text)]
        except Exception as e:
            error_msg = f"Improve failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# ---------------------------------------------------------------------------
# V1 pipeline status tool
# ---------------------------------------------------------------------------


@mcp.tool()
@log_usage(function_name="MCP cognify_status", log_type="mcp_tool")
async def cognify_status(
    dataset_name: str = "main_dataset",
    pipelines: List[str] = None,
) -> list:
    """
    Get the current status of selected pipelines.

    This function retrieves information about current and recently completed
    pipeline operations in the selected dataset.

    Returns
    -------
    list
        A list containing a single TextContent object with the status information as a string.
        The status includes information about active and completed jobs for the
        requested pipelines.

    Notes
    -----
    - By default this checks "cognify_pipeline" (backward compatible)
    - Use `pipelines` to restrict to specific pipeline names
    - Status information includes job progress, execution time, and completion status
    - The status is returned in string format for easy reading
    - This operation is not available in API mode
    """
    with redirect_stdout(sys.stderr):
        try:
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()
            dataset_id = await get_unique_dataset_id(dataset_name, user)
            requested_pipelines = list(dict.fromkeys(pipelines or ["cognify_pipeline"]))

            if len(requested_pipelines) == 1:
                status = await cognee_client.get_pipeline_status(
                    [dataset_id], requested_pipelines[0]
                )
            else:
                status: dict[str, dict] = {str(dataset_id): {}}
                for pipeline_name in requested_pipelines:
                    pipeline_status = await cognee_client.get_pipeline_status(
                        [dataset_id], pipeline_name
                    )
                    if str(dataset_id) in pipeline_status:
                        status[str(dataset_id)][pipeline_name] = pipeline_status[str(dataset_id)]

            # Append any background task errors
            status_text = str(status)
            dataset_errors = _task_errors.get(dataset_name, [])
            if dataset_errors:
                error_lines = ["\n\nBackground task errors:"]
                for ts, err in sorted(dataset_errors, reverse=True):
                    error_lines.append(f"  [{ts}] {err}")
                status_text += "\n".join(error_lines)

            return [types.TextContent(type="text", text=status_text)]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get cognify status: {str(e)}"
            # Still report background errors even if pipeline status fails
            dataset_errors = _task_errors.get(dataset_name, [])
            if dataset_errors:
                error_lines = ["\n\nBackground task errors:"]
                for ts, err in sorted(dataset_errors, reverse=True):
                    error_lines.append(f"  [{ts}] {err}")
                error_msg += "\n".join(error_lines)
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

    # Cognee Cloud connection options
    parser.add_argument(
        "--serve-url",
        default=None,
        help="Cognee Cloud or remote instance URL (e.g., https://your-instance.cognee.ai). "
        "Calls cognee.serve() at startup so all SDK operations route to the cloud. "
        "Can also be set via COGNEE_SERVICE_URL env var.",
    )

    parser.add_argument(
        "--serve-api-key",
        default=None,
        help="API key for the Cognee Cloud instance. Can also be set via COGNEE_API_KEY env var.",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    mcp.settings.host = args.host
    mcp.settings.port = int(args.port)
    _configure_transport_security(args.host)

    # Resolve cloud connection: CLI args take precedence over env vars
    serve_url = args.serve_url or os.environ.get("COGNEE_SERVICE_URL", "")
    serve_api_key = args.serve_api_key or os.environ.get("COGNEE_API_KEY", "")

    # Connect to Cognee Cloud if configured (before migrations — cloud handles its own DB)
    if serve_url and not args.api_url:
        import cognee

        serve_kwargs = {"url": serve_url}
        if serve_api_key:
            serve_kwargs["api_key"] = serve_api_key
        await cognee.serve(**serve_kwargs)
        logger.info(f"Connected to Cognee Cloud: {serve_url}")

    # Skip migrations when in API or Cloud mode (remote handles its own database)
    is_remote = bool(args.api_url) or bool(serve_url)
    if not args.no_migration and not is_remote:
        from cognee.modules.engine.operations.setup import setup
        from cognee.run_migrations import run_migrations

        logger.info("Running database migrations...")

        await setup()
        await run_migrations()

        logger.info("Database migrations done.")
    elif not is_remote:
        logger.info("Skipping DB migrations")

    match args.transport.lower():
        case "sse":
            logger.info(f"Running MCP server with SSE transport on {args.host}:{args.port}")
            await run_sse_with_cors()
        case "http":
            logger.info(
                f"Running MCP server with Streamable HTTP transport on {args.host}:{args.port}{args.path}"
            )
            await run_http_with_cors()
        case _:
            logger.info("Running MCP server with stdio")
            await mcp.run_stdio_async()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
