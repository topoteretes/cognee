"""Tool for ingesting core developer rule files into Cognee's memory layer."""

import os
import sys
import asyncio
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger, get_log_file_location

from src.shared import context
from .utils import load_class

logger = get_logger()


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
                await context.cognee_client.add(file_path, node_set=["developer_rules"])

                model = None
                if graph_model_file and graph_model_name:
                    if context.cognee_client.use_api:
                        logger.warning(
                            "Custom graph models are not supported in API mode, ignoring."
                        )
                    else:
                        from cognee.shared.data_models import KnowledgeGraph

                        model = load_class(graph_model_file, graph_model_name)

                await context.cognee_client.cognify(graph_model=model)
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
