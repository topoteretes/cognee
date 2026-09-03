"""The cognify CODE-route adapter for code files.

Recognition happens at add time through the loader system: the code loader
(infrastructure/loaders/core/code_loader.py) claims supported code files, and
ingest_data tags them with ``system_metadata = {"source": "code"}``.
Classification maps that tag to CodeFileDocument, and cognify routing sends
such items down the CODE route (modules/cognify/routing.py). The route's task
list is built here: it stages each stored code file into a temporary per-file
directory (enola only accepts directories) and runs the deterministic enola
code graph pipeline on it — no LLM or embedding calls.
"""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from cognee.modules.pipelines.tasks.task import Task, task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_utils import metadata_source

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_graph")

_TSCONFIG_JSON = '{"compilerOptions": {}}\n'

# enola detects PROJECTS, not lone files. Since enola 0.4.5 language detection
# is membership over the walked file list, so a bare source file of nearly
# every supported language produces facts on its own — verified per extension
# of the code loader's SUPPORTED_CODE_EXTENSIONS against enola v0.4.12. The
# exception is plain JavaScript: .js/.jsx are only claimed once a
# tsconfig.json sits next to them. Staging fabricates exactly that marker and
# nothing else. Earlier releases needed a manifest (requirements.txt, go.mod,
# Cargo.toml, package.json, ...) for more languages; those markers are gone on
# purpose, because since enola 0.4.8 the manifests extractor turns a
# manifest's declared dependencies into facts — a fabricated package.json
# declaring vue would add a pkg:npm/vue dependency the user's file never had.
# Also observed at 0.4.12: a lone .kts script or C .h header yields zero facts
# (the route then stores an empty repository, which is harmless).
_DETECTION_MARKERS: dict = {
    "js": {"tsconfig.json": _TSCONFIG_JSON},
    "jsx": {"tsconfig.json": _TSCONFIG_JSON},
}


def is_code_sourced(metadata) -> bool:
    """Check whether system_metadata indicates a code file (source == "code")."""
    return metadata_source(metadata) == "code"


def _original_file_name(data_item) -> str:
    """The original file's basename, e.g. "payments.py".

    enola detects languages by file extension, and only the original path
    keeps the real one (Data.original_extension is content-sniffed to "txt"
    for code files), so the staged copy must reuse this basename.
    """
    return os.path.basename(str(data_item.original_data_location).rstrip("/"))


def _staged_repo_name(data_item) -> str:
    """Stable per-file repo identity for the staged enola run.

    Code graph node ids, the stale-node sweep, and the snapshot-skip marker are
    all keyed on the repo name (fact_node_id), so it must be stable across
    re-ingestions of the same file AND unique per Data record — two files named
    utils.py must not merge into (and sweep) one repo. The Data id provides
    both: it is stable for a re-added file and unique per record. The FULL id
    is used: a truncated prefix would give same-named files a birthday-paradox
    chance of sharing a repo identity, silently overwriting and sweeping each
    other's nodes.
    """
    return f"{_original_file_name(data_item)}_{data_item.id}"


def _stage_code_file(data_item, staging_root, content: str) -> Path:
    """Stage one code file (plus its detection markers) for an enola run."""
    repo_dir = Path(staging_root) / _staged_repo_name(data_item)
    repo_dir.mkdir()
    file_name = _original_file_name(data_item)
    (repo_dir / file_name).write_text(content, encoding="utf-8")
    extension = os.path.splitext(file_name)[1].lstrip(".").lower()
    for marker_name, marker_content in _DETECTION_MARKERS.get(extension, {}).items():
        # Never overwrite the staged file itself, should a marker ever share
        # its name (a user's own tsconfig.json, say).
        if marker_name != file_name:
            (repo_dir / marker_name).write_text(marker_content, encoding="utf-8")
    return repo_dir


@task_summary("Extracted code graph from {n} code file(s)")
async def extract_code_files_graph(
    data_documents: list,
    ctx: Optional["PipelineContext"] = None,
) -> list:
    """Cognify CODE-route adapter: run the enola pipeline on stored code files.

    Each Data record's raw content is staged into a temporary directory under
    its original file name (enola detects languages by extension and only
    accepts directories), then the standard code graph tasks run on it:
    extract_code_graph -> add_code_graph_data_points -> add_code_graph_edges.
    graph_only always: SearchType.CODE uses graph indexes, and the CODE route
    must stay free of embedding calls like the rest of the enola pipeline.
    """
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        add_code_graph_edges,
        extract_code_graph,
    )

    for data_item in data_documents:
        if not is_code_sourced(data_item):
            raise ValueError(
                f"Data item {getattr(data_item, 'id', '?')} is not code-sourced "
                "(system_metadata.source != 'code'); it must not reach the CODE route."
            )

        async with open_data_file(data_item.raw_data_location, mode="r", encoding="utf-8") as file:
            content = file.read()

        with tempfile.TemporaryDirectory(prefix="cognee_code_") as staging_root:
            repo_dir = _stage_code_file(data_item, staging_root, content)

            data_points = await extract_code_graph(repo_path=repo_dir)
            state = await add_code_graph_data_points(data_points, ctx=ctx, graph_only=True)
            await add_code_graph_edges(state, repo_path=repo_dir, ctx=ctx)

        logger.info(
            "Code graph extracted for %s (%s).", _original_file_name(data_item), data_item.id
        )

    return data_documents


def get_code_file_tasks() -> List[Task]:
    """The cognify CODE-route task list: one adapter task, no LLM stages."""
    return [Task(extract_code_files_graph)]
