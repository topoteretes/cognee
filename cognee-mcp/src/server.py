import json
import os
import sys
import argparse
import cognee
import asyncio

from cognee.shared.logging_utils import get_logger, setup_logging, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.modules.users.methods import get_default_user
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.storage.utils import JSONEncoder
from .codingagents.coding_rule_associations import (
    add_rule_associations,
    get_existing_rules,
)


mcp = FastMCP("Cognee")

logger = get_logger()


@mcp.tool()
async def cognee_add_developer_rules(
    base_path: str = ".", graph_model_file: str = None, graph_model_name: str = None
) -> list:
    """
    Ingest core developer rule files into Cognee's memory layer.

    This function loads a predefined set of developer-related configuration,
    rule, and documentation files from the base repository and assigns them
    to the special 'developer_rules' node set in Cognee. It ensures these
    foundational files are always part of the structured memory graph.

    Parameters
    ----------
    base_path : str
        Root path to resolve relative file paths. Defaults to current directory.

    graph_model_file : str, optional
        Optional path to a custom schema file for knowledge graph generation.

    graph_model_name : str, optional
        Optional class name to use from the graph_model_file schema.

    Returns
    -------
    list
        A message indicating how many rule files were scheduled for ingestion,
        and how to check their processing status.

    Notes
    -----
    - Each file is processed asynchronously in the background.
    - Files are attached to the 'developer_rules' node set.
    - Missing files are skipped with a logged warning.
    """

    developer_rule_paths = [
        ".cursorrules",
        ".cursor/rules",
        ".same/todos.md",
        ".windsurfrules",
        ".clinerules",
        "CLAUDE.md",
        ".sourcegraph/memory.md",
        "AGENT.md",
        "AGENTS.md",
    ]

    async def cognify_task(file_path: str) -> None:
        with redirect_stdout(sys.stderr):
            logger.info(f"Starting cognify for: {file_path}")
            try:
                await cognee.add(file_path, node_set=["developer_rules"])
                model = KnowledgeGraph
                if graph_model_file and graph_model_name:
                    model = load_class(graph_model_file, graph_model_name)
                await cognee.cognify(graph_model=model)
                logger.info(f"Cognify finished for: {file_path}")
            except Exception as e:
                logger.error(f"Cognify failed for {file_path}: {str(e)}")
                raise ValueError(f"Failed to cognify: {str(e)}")

    tasks = []
    for rel_path in developer_rule_paths:
        abs_path = os.path.join(base_path, rel_path)
        if os.path.isfile(abs_path):
            tasks.append(asyncio.create_task(cognify_task(abs_path)))
        else:
            logger.warning(f"Skipped missing developer rule file: {abs_path}")
    log_file = get_log_file_location()
    return [
        types.TextContent(
            type="text",
            text=(
                f"Started cognify for {len(tasks)} developer rule files in background.\n"
                f"All are added to the `developer_rules` node set.\n"
                f"Use `cognify_status` or check logs at {log_file} to monitor progress."
            ),
        )
    ]


@mcp.tool()
async def cognify(data: str, graph_model_file: str = None, graph_model_name: str = None) -> list:
    """
    Transform data into a structured knowledge graph in Cognee's memory layer.

    This function launches a background task that processes the provided text/file location and
    generates a knowledge graph representation. The function returns immediately while
    the processing continues in the background due to MCP timeout constraints.

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

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.

    Notes
    -----
    - The function launches a background task and returns immediately
    - The actual cognify process may take significant time depending on text length
    - Use the cognify_status tool to check the progress of the operation
    """

    async def cognify_task(
        data: str, graph_model_file: str = None, graph_model_name: str = None
    ) -> str:
        """Build knowledge graph from the input text"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Cognify process starting.")
            if graph_model_file and graph_model_name:
                graph_model = load_class(graph_model_file, graph_model_name)
            else:
                graph_model = KnowledgeGraph

            await cognee.add(data)

            try:
                await cognee.cognify(graph_model=graph_model)
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    asyncio.create_task(
        cognify_task(
            data=data,
            graph_model_file=graph_model_file,
            graph_model_name=graph_model_name,
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

            await cognee.add(data, node_set=["user_agent_interaction"])

            try:
                await cognee.cognify()
                logger.info("Save interaction process finished.")
                logger.info("Generating associated rules from interaction data.")

                await add_rule_associations(data=data, rules_nodeset_name="coding_agent_rules")

                logger.info("Associated rules generated from interaction data.")

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
async def codify(repo_path: str) -> list:
    """
    Analyze and generate a code-specific knowledge graph from a software repository.

    This function launches a background task that processes the provided repository
    and builds a code knowledge graph. The function returns immediately while
    the processing continues in the background due to MCP timeout constraints.

    Parameters
    ----------
    repo_path : str
        Path to the code repository to analyze. This can be a local file path or a
        relative path to a repository. The path should point to the root of the
        repository or a specific directory within it.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.

    Notes
    -----
    - The function launches a background task and returns immediately
    - The code graph generation may take significant time for larger repositories
    - Use the codify_status tool to check the progress of the operation
    - Process results are logged to the standard Cognee log file
    - All stdout is redirected to stderr to maintain MCP communication integrity
    """

    async def codify_task(repo_path: str):
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Codify process starting.")
            results = []
            async for result in run_code_graph_pipeline(repo_path, False):
                results.append(result)
                logger.info(result)
            if all(results):
                logger.info("Codify process finished succesfully.")
            else:
                logger.info("Codify process failed.")

    asyncio.create_task(codify_task(repo_path))

    log_file = get_log_file_location()
    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current codify status use the codify_status tool\n"
        f"or you can check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def search(search_query: str, search_type: str) -> list:
    """
    Search the Cognee knowledge graph for information relevant to the query.

    This function executes a search against the Cognee knowledge graph using the
    specified query and search type. It returns formatted results based on the
    search type selected.

    Parameters
    ----------
    search_query : str
        The search query in natural language. This can be a question, instruction, or
        any text that expresses what information is needed from the knowledge graph.

    search_type : str
        The type of search to perform. Valid options include:
        - "GRAPH_COMPLETION": Returns an LLM response based on the search query and Cognee's memory
        - "RAG_COMPLETION": Returns an LLM response based on the search query and standard RAG data
        - "CODE": Returns code-related knowledge in JSON format
        - "CHUNKS": Returns raw text chunks from the knowledge graph
        - "INSIGHTS": Returns relationships between nodes in readable format

        The search_type is case-insensitive and will be converted to uppercase.

    Returns
    -------
    list
        A list containing a single TextContent object with the search results.
        The format of the result depends on the search_type:
        - For CODE: JSON-formatted search results
        - For GRAPH_COMPLETION/RAG_COMPLETION: A single text completion
        - For CHUNKS: String representation of the raw chunks
        - For INSIGHTS: Formatted string showing node relationships
        - For other types: String representation of the search results

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
            search_results = await cognee.search(
                query_type=SearchType[search_type.upper()], query_text=search_query
            )

            if search_type.upper() == "CODE":
                return json.dumps(search_results, cls=JSONEncoder)
            elif (
                search_type.upper() == "GRAPH_COMPLETION" or search_type.upper() == "RAG_COMPLETION"
            ):
                return search_results[0]
            elif search_type.upper() == "CHUNKS":
                return str(search_results)
            elif search_type.upper() == "INSIGHTS":
                results = retrieved_edges_to_string(search_results)
                return results
            else:
                return str(search_results)

    search_results = await search_task(search_query, search_type)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
async def get_developer_rules() -> list:
    """
    Retrieve all developer rules that were generated based on previous interactions.

    This tool queries the Cognee knowledge graph and returns a list of developer
    rules.

    Parameters
    ----------
    None

    Returns
    -------
    list
        A list containing a single TextContent object with the retrieved developer rules.
        The format is plain text containing the developer rules in bulletpoints.

    Notes
    -----
    - The specific logic for fetching rules is handled internally.
    - This tool does not accept any parameters and is intended for simple rule inspection use cases.
    """

    async def fetch_rules_from_cognee() -> str:
        """Collect all developer rules from Cognee"""
        with redirect_stdout(sys.stderr):
            developer_rules = await get_existing_rules(rules_nodeset_name="coding_agent_rules")
            return developer_rules

    rules_text = await fetch_rules_from_cognee()

    return [types.TextContent(type="text", text=rules_text)]


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
    """
    with redirect_stdout(sys.stderr):
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        return [types.TextContent(type="text", text="Pruned")]


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
    """
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status(
            [await get_unique_dataset_id("main_dataset", user)], "cognify_pipeline"
        )
        return [types.TextContent(type="text", text=str(status))]


@mcp.tool()
async def codify_status():
    """
    Get the current status of the codify pipeline.

    This function retrieves information about current and recently completed codify operations
    in the codebase dataset. It provides details on progress, success/failure status, and statistics
    about the processed code repositories.

    Returns
    -------
    list
        A list containing a single TextContent object with the status information as a string.
        The status includes information about active and completed jobs for the cognify_code_pipeline.

    Notes
    -----
    - The function retrieves pipeline status specifically for the "cognify_code_pipeline" on the "codebase" dataset
    - Status information includes job progress, execution time, and completion status
    - The status is returned in string format for easy reading
    """
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status(
            [await get_unique_dataset_id("codebase", user)], "cognify_code_pipeline"
        )
        return [types.TextContent(type="text", text=str(status))]


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(
            f"Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}"
        )
        await mcp.run_sse_async()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
