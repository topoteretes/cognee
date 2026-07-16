"""Local AST code-graph ingestion (issue #3668).

Walks a Python source tree, parses each file into cognee's code-graph DataPoints
with tree-sitter, and persists the structural graph via ``add_data_points`` — no
LLM calls. The nodes carry relationship fields, so ``add_data_points`` mints the
File->Class / File->Function / File->Import edges natively.

Once stored, the graph is searchable through the normal search types (e.g.
``GRAPH_COMPLETION``), which already index every DataPoint collection.

    from cognee.tasks.codegraph import ingest_code_graph
    await ingest_code_graph("/path/to/repo")
"""

from pathlib import Path
from typing import Iterable, Optional, Union

from cognee.modules.engine.operations.setup import setup
from cognee.shared.CodeGraphEntities import CodeFile, Repository
from cognee.tasks.storage import add_data_points

from .parse import parse_file

DEFAULT_EXTENSIONS = frozenset({".py"})


async def ingest_code_graph(
    path: Union[str, Path],
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> list[CodeFile]:
    """Parse every source file under *path* and store the resulting code graph.

    Args:
        path: A file or directory to ingest.
        extensions: Source-file extensions to process (default: ``{".py"}``).

    Returns:
        The stored :class:`CodeFile` nodes (empty if nothing was parsed).
    """
    root = Path(path)
    suffixes = set(extensions)

    if root.is_file():
        files = [root]
        repository: Optional[Repository] = None
    else:
        files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix in suffixes)
        repository = Repository(id=Repository.id_for(str(root)), path=str(root))

    code_files = [
        code_file
        for f in files
        if (code_file := parse_file(str(f), repository=repository)) is not None
    ]
    if code_files:
        await setup()  # ensure the databases exist, like cognee.add()
        await add_data_points(code_files)
    return code_files
