from typing import AsyncGenerator
from uuid import NAMESPACE_OID, uuid5

# from tqdm import tqdm
from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile, CodePart
from cognee.tasks.repo_processor.extract_code_parts import extract_code_parts
import logging

logger = logging.getLogger(__name__)


def _add_code_parts_nodes_and_edges(code_file: CodeFile, part_type, code_parts) -> None:
    """Add code part nodes and edges for a specific part type."""
    if not code_parts:
        logger.debug(f"No code parts to add for node {code_file.id} and part_type {part_type}.")
        return

    part_nodes = []

    for idx, code_part in enumerate(code_parts):
        if not code_part.strip():
            logger.warning(f"Empty code part in node {code_file.id} and part_type {part_type}.")
            continue

        part_node_id = uuid5(NAMESPACE_OID, f"{code_file.id}_{part_type}_{idx}")

        part_nodes.append(
            CodePart(
                id=part_node_id,
                type=part_type,
                # part_of = code_file,
                source_code=code_part,
            )
        )

        # graph.add_node(part_node_id, source_code=code_part, node_type=part_type)
        # graph.add_edge(parent_node_id, part_node_id, relation="contains")

    code_file.contains = code_file.contains or []
    code_file.contains.extend(part_nodes)


def _process_single_node(code_file: CodeFile) -> None:
    """Process a single Python file node."""
    node_id = code_file.id
    source_code = code_file.source_code

    if not source_code.strip():
        logger.warning(f"Node {node_id} has no or empty 'source_code'. Skipping.")
        return

    try:
        code_parts_dict = extract_code_parts(source_code)
    except Exception as e:
        logger.error(f"Error processing node {node_id}: {e}")
        return

    for part_type, code_parts in code_parts_dict.items():
        _add_code_parts_nodes_and_edges(code_file, part_type, code_parts)


async def expand_dependency_graph(
    data_points: list[DataPoint],
) -> AsyncGenerator[list[DataPoint], None]:
    """Process Python file nodes, adding code part nodes and edges."""
    # for data_point in tqdm(data_points, desc = "Expand dependency graph", unit = "data_point"):
    for data_point in data_points:
        if isinstance(data_point, CodeFile):
            _process_single_node(data_point)
        yield data_point

    # return data_points
