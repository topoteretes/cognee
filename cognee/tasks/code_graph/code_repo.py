"""Code-project detection and repo-level ingestion for directory adds.

When ``add()`` receives a directory that IS a code project (it carries a
project marker such as pyproject.toml or go.mod), flattening it file-by-file
is the wrong granularity: every source file would run its own enola pass in
its own one-file "repo" (no cross-file edges, one whole-graph read per file),
the project manifests would be mis-ingested as prose, and binary build junk
would abort the add. Instead, ``resolve_data_directories`` partitions the
tree:

- ONE repo-level manifest DataItem covers the source files and project
  manifests. Cognify routes it down the CODE_REPO route (same mechanism as
  DLT source manifests), which runs a single enola pass over the original
  directory — cross-file ``calls``/``imports`` edges, one repository node,
  one graph read.
- Document-like files (README.md, docs, PDFs, ...) are emitted individually
  and flow through their normal loaders and cognify routes, exactly as if
  added on their own — a repo's prose stays searchable.
- Everything else (VCS internals, caches, lockfiles, binaries, dotfiles —
  which may hold secrets) is skipped explicitly and counted.

The repo item's raw data is a JSON manifest (repo path + covered files +
combined content hash), so the stored record is small and re-adding a changed
repo resets pipeline status through the normal content-change detection. The
repository itself is read from its original location at cognify time — like
``remember(content_type="code")``, the enola run happens in place.
"""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_utils import metadata_source

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_graph")

# A directory is a code project when any of these exist at its root. The list
# mirrors enola's own project detection (per-language manifests) plus .git.
PROJECT_MARKERS = frozenset(
    {
        ".git",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "go.mod",
        "package.json",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "Package.swift",
        "Gemfile",
        "composer.json",
        "pubspec.yaml",
        "CMakeLists.txt",
    }
)

# Project manifests/build files covered by the repo item (enola reads them for
# detection and dependency facts). Code-extension files are covered via the
# code loader's SUPPORTED_CODE_EXTENSIONS.
PROJECT_MANIFEST_FILES = frozenset(
    PROJECT_MARKERS - {".git"} | {"go.sum", "tsconfig.json", "Makefile", "makefile"}
)

# Directories never worth ingesting from a working tree. Any path with a
# component in this set — or any hidden (dot-prefixed) component, which also
# keeps secrets like .env out of memory — is skipped outright.
SKIP_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        "dist",
        "build",
        "target",
        "coverage",
    }
)

# Audio/video inside a code project is demo/marketing material (logo loops,
# promo clips), not knowledge — and transcription is fragile there: a silent
# .mp4 has no audio stream, so whisper's ffmpeg extraction crashes and one
# decorative asset aborts the whole add (observed on cognee's own repo).
# Images stay documents: architecture diagrams and screenshots carry content.
# TODO: Make media skipping a code-ingestion option (on/off), default OFF —
# media in a repo would then be transcribed like any standalone add unless the
# caller opts into skipping. Requires the loader-level fix first (a silent
# video must yield empty text, not crash the add), else the default is unsafe.
REPO_SKIPPED_MEDIA_EXTENSIONS = frozenset(
    {
        "aac",
        "aiff",
        "amr",
        "avi",
        "flac",
        "m4a",
        "m4v",
        "mid",
        "mkv",
        "mov",
        "mp3",
        "mp4",
        "ogg",
        "wav",
        "webm",
    }
)


def is_code_repo_sourced(metadata) -> bool:
    """Check whether system_metadata indicates a repo manifest (source == "code_repo")."""
    return metadata_source(metadata) == "code_repo"


def detect_code_project(directory: Path) -> bool:
    """Whether a directory's ROOT carries a code-project marker."""
    return any((directory / marker).exists() for marker in PROJECT_MARKERS)


def _is_skipped_path(relative_path: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".") for part in relative_path.parts)


def _document_extensions() -> frozenset:
    """Every path extension a registered non-code loader claims.

    Keyed on path extensions ONLY — cognee's content sniffing reports unknown
    binaries as txt/text-plain, so "some loader would take it" is true for any
    blob and would feed binaries to text_loader (the classic .pyc abort). In a
    repo, a file whose extension no loader names is junk, not a document.
    """
    from cognee.infrastructure.loaders import get_loader_engine
    from cognee.infrastructure.loaders.core.code_loader import SUPPORTED_CODE_EXTENSIONS

    loader_engine = get_loader_engine()
    extensions: set = set()
    for loader_name in loader_engine.get_available_loaders():
        extensions.update(
            ext.lower() for ext in loader_engine.get_loader_info(loader_name)["extensions"]
        )
    return frozenset(extensions - SUPPORTED_CODE_EXTENSIONS)


def partition_repo_files(directory: Path) -> tuple[List[Path], List[Path], List[Path]]:
    """Partition a project tree into (covered, documents, skipped) files.

    covered: source files + project manifests — represented by the single
    repo item and consumed by enola. documents: files whose extension a
    non-code loader claims — emitted individually so their content stays in
    the document pipeline. skipped: hidden files/dirs, build junk, and
    unclaimed extensions (binaries) — never ingested, so one .pyc cannot
    abort the add.
    """
    from cognee.infrastructure.loaders.core.code_loader import SUPPORTED_CODE_EXTENSIONS

    document_extensions = _document_extensions()
    covered: List[Path] = []
    documents: List[Path] = []
    skipped: List[Path] = []

    for file_path in sorted(directory.rglob("*")):
        # is_file() follows symlinks, and read_bytes() below would then pull the
        # TARGET's contents into the manifest. A repo (cloned or local) containing
        # 'creds.py -> ~/.aws/credentials' would otherwise be indexed verbatim.
        # Clones are written with core.symlinks=false, so this covers local paths
        # and any pre-existing clone made before that landed.
        if file_path.is_symlink():
            skipped.append(file_path)
            continue
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(directory)
        if _is_skipped_path(relative):
            skipped.append(file_path)
            continue
        extension = file_path.suffix.lstrip(".").lower()
        if extension in SUPPORTED_CODE_EXTENSIONS or file_path.name in PROJECT_MANIFEST_FILES:
            covered.append(file_path)
        elif extension in REPO_SKIPPED_MEDIA_EXTENSIONS:
            skipped.append(file_path)
        elif extension in document_extensions:
            documents.append(file_path)
        else:
            skipped.append(file_path)

    return covered, documents, skipped


def build_repo_manifest(directory: Path, covered: List[Path]) -> str:
    """The repo item's raw data: covered files + a combined content hash.

    The hash covers every covered file's content, so the manifest text — and
    therefore the record's content hash — changes exactly when the code does,
    driving re-cognify through the normal content-change detection.
    """
    combined = hashlib.sha256()
    files = []
    for file_path in covered:
        relative = str(file_path.relative_to(directory))
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        combined.update(f"{len(relative)}:{relative}:{digest}".encode())
        files.append(relative)

    return json.dumps(
        {
            "kind": "code_repo_manifest",
            "repo_path": str(directory),
            "file_count": len(files),
            "files": files,
            "content_hash": combined.hexdigest(),
        },
        indent=2,
    )


async def resolve_code_repository(directory: Path, user=None, dataset_id=None):
    """Build the repo-level DataItem (and the document file list) for a project.

    Returns (manifest_item, document_paths, skip_count). The data id is pinned
    to the stable identity (user, dataset, repo path) — mirroring DLT source
    manifests — so re-adding the same repo updates one record instead of
    accreting new ones; without a user context the id is left content-derived.

    Without an LLM API key, the project's document files are excluded (logged)
    instead of emitted: their routes need an LLM (images transcribe at add
    time, text is LLM-chunked at cognify), so they would only fail later. The
    code graph itself never needs one — a key-less repo add still works fully.
    """
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.tasks.ingestion.data_item import DataItem

    covered, documents, skipped = partition_repo_files(directory)

    if documents and not get_llm_config().llm_api_key:
        logger.warning(
            "No LLM API key configured (LLM_API_KEY): excluding %d document file(s) of "
            "the code project from processing — their pipelines need an LLM (image "
            "transcription at add time, text extraction at cognify). The code graph is "
            "unaffected (enola is LLM-free). Set LLM_API_KEY and re-add the repository "
            "to ingest its documents.",
            len(documents),
        )
        skipped = skipped + documents
        documents = []

    manifest_text = build_repo_manifest(directory, covered)

    data_id = None
    if user is not None:
        from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id

        data_id = await get_unique_data_id(f"code_repo:{directory}", user, dataset_id)

    manifest_item = DataItem(
        data=manifest_text,
        label=f"code_repo:{directory.name}",
        system_metadata={
            "source": "code_repo",
            "repo_path": str(directory),
            "file_count": len(covered),
        },
        data_id=data_id,
    )

    logger.info(
        "Code project detected: 1 repo item covering %d file(s), %d document(s), %d skipped.",
        len(covered),
        len(documents),
        len(skipped),
    )
    return manifest_item, documents, len(skipped)


async def extract_code_repo_graph(
    data_documents: list,
    ctx: Optional["PipelineContext"] = None,
) -> list:
    """Cognify CODE_REPO-route adapter: one enola run over the whole project.

    Reads the stored manifest for the repo path and runs the standard code
    graph tasks on the ORIGINAL directory (enola writes its .enola snapshot
    there, exactly like remember(content_type="code")). One repository node,
    cross-file edges, one graph read per repo. No LLM, no embeddings.
    """
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        add_code_graph_edges,
        extract_code_graph,
    )

    for data_item in data_documents:
        if not is_code_repo_sourced(data_item):
            raise ValueError(
                f"Data item {getattr(data_item, 'id', '?')} is not a code repo manifest "
                "(system_metadata.source != 'code_repo'); it must not reach the CODE_REPO route."
            )

        async with open_data_file(data_item.raw_data_location, mode="r", encoding="utf-8") as file:
            manifest = json.loads(file.read())

        repo_path = Path(manifest["repo_path"])
        if not repo_path.is_dir():
            raise ValueError(
                f"Code repository '{repo_path}' no longer exists; the repo item reads "
                "the original directory at cognify time. Re-add the repository from "
                "its current location."
            )

        data_points = await extract_code_graph(repo_path=repo_path)
        state = await add_code_graph_data_points(data_points, ctx=ctx, graph_only=True)
        await add_code_graph_edges(state, repo_path=repo_path, ctx=ctx)

        logger.info("Code repo graph extracted for %s (%s).", repo_path, data_item.id)

    return data_documents


def get_code_repo_tasks() -> List:
    """The cognify CODE_REPO-route task list: one adapter task, no LLM stages."""
    from cognee.modules.pipelines.tasks.task import Task

    return [Task(extract_code_repo_graph)]
