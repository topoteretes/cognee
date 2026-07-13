"""Entry point for AST code-graph ingestion.

Usage:
    from cognee.tasks.codegraph import ingest_code_graph
    await ingest_code_graph("/path/to/repo")
"""

from pathlib import Path
from typing import Union

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage import add_data_points, index_graph_edges

from .parse import parse_file

_SUPPORTED = {".py"}


async def ingest_code_graph(
    path: Union[str, Path],
    extensions: set[str] = _SUPPORTED,
) -> None:
    """Parse every source file under *path* and store the resulting code graph.

    Args:
        path: A file or directory to ingest.
        extensions: Source extensions to process (default: {".py"}).
    """
    root = Path(path)
    files = [root] if root.is_file() else [
        f for f in root.rglob("*") if f.suffix in extensions and f.is_file()
    ]

    all_nodes: list = []
    all_edges: list = []
    for f in files:
        nodes, edges = parse_file(str(f))
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    if not all_nodes:
        return

    await add_data_points(all_nodes)

    if all_edges:
        graph_engine = await get_graph_engine()
        await graph_engine.add_edges(all_edges)
        await index_graph_edges(all_edges)
