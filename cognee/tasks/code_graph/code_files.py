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

_NODE_PACKAGE_JSON = '{"name": "staged", "version": "0.0.0"}\n'
_TSCONFIG_JSON = '{"compilerOptions": {}}\n'

# enola detects PROJECTS, not lone files: for these languages (of the code
# loader's SUPPORTED_CODE_EXTENSIONS) a bare source file produces zero facts
# until a project manifest sits next to it. Staging fabricates the minimal
# marker that turns detection on — each entry verified against enola v0.3.13
# (java/c/c++/c#/vb/f#/ruby/php/scala/dart/terraform/proto need none). The
# markers carry no content of their own, so they add no facts beyond the
# staged file's.
_DETECTION_MARKERS: dict = {
    "py": {"requirements.txt": ""},
    "go": {"go.mod": "module staged\n\ngo 1.21\n"},
    "rs": {"Cargo.toml": '[package]\nname = "staged"\nversion = "0.0.0"\n'},
    "js": {"package.json": _NODE_PACKAGE_JSON, "tsconfig.json": _TSCONFIG_JSON},
    "jsx": {"package.json": _NODE_PACKAGE_JSON, "tsconfig.json": _TSCONFIG_JSON},
    "ts": {"package.json": _NODE_PACKAGE_JSON, "tsconfig.json": _TSCONFIG_JSON},
    "tsx": {"package.json": _NODE_PACKAGE_JSON, "tsconfig.json": _TSCONFIG_JSON},
    "kt": {"build.gradle.kts": 'plugins { kotlin("jvm") }\n'},
    "kts": {"build.gradle.kts": 'plugins { kotlin("jvm") }\n'},
    "swift": {
        "Package.swift": (
            "// swift-tools-version:5.5\n"
            "import PackageDescription\n"
            'let package = Package(name: "staged")\n'
        )
    },
    "vue": {
        "package.json": '{"name": "staged", "version": "0.0.0", "dependencies": {"vue": "*"}}\n'
    },
    "svelte": {
        "package.json": (
            '{"name": "staged", "version": "0.0.0", "dependencies": {"svelte": "*"}}\n'
        )
    },
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
    both: it is stable for a re-added file and unique per record.
    """
    return f"{_original_file_name(data_item)}_{str(data_item.id)[:8]}"


def _stage_code_file(data_item, staging_root, content: str) -> Path:
    """Stage one code file (plus its detection markers) for an enola run."""
    repo_dir = Path(staging_root) / _staged_repo_name(data_item)
    repo_dir.mkdir()
    file_name = _original_file_name(data_item)
    (repo_dir / file_name).write_text(content, encoding="utf-8")
    extension = os.path.splitext(file_name)[1].lstrip(".").lower()
    for marker_name, marker_content in _DETECTION_MARKERS.get(extension, {}).items():
        # A staged file may BE its own marker (a user-added Package.swift or
        # build.gradle.kts); never overwrite it.
        if marker_name != file_name:
            (repo_dir / marker_name).write_text(marker_content, encoding="utf-8")
    return repo_dir


@task_summary("Extracted code graph from {n} code file(s)")
async def extract_code_files_graph(
    data_documents: list,
    ctx: Optional["PipelineContext"] = None,
    index_vectors: bool = False,
) -> list:
    """Cognify CODE-route adapter: run the enola pipeline on stored code files.

    Each Data record's raw content is staged into a temporary directory under
    its original file name (enola detects languages by extension and only
    accepts directories), then the standard code graph tasks run on it:
    extract_code_graph -> add_code_graph_data_points -> add_code_graph_edges.
    No LLM calls ever; index_vectors additionally embeds code fact names so
    completion/triplet search can seed from them (SearchType.CODE needs only
    the graph indexes).
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
            state = await add_code_graph_data_points(
                data_points, ctx=ctx, graph_only=not index_vectors
            )
            await add_code_graph_edges(state, repo_path=repo_dir, ctx=ctx)

        logger.info(
            "Code graph extracted for %s (%s).", _original_file_name(data_item), data_item.id
        )

    return data_documents


def get_code_file_tasks(index_vectors: bool = False) -> List[Task]:
    """The cognify CODE-route task list: one adapter task, no LLM stages.

    index_vectors mirrors get_code_graph_tasks: opt-in embedding of code fact
    names for completion/triplet search. cognify passes the deployment default
    (CognifyConfig.code_route_index_vectors, env CODE_ROUTE_INDEX_VECTORS).
    """
    return [Task(extract_code_files_graph, index_vectors=index_vectors)]
