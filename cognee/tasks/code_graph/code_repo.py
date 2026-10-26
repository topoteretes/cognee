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
combined content hash), so the stored record is small. Its id is pinned, so
``resolve_code_repository`` clears the stored row's add stamp before the
pipeline's incremental check sees it; re-adding a changed repo then resets
pipeline status through the normal content-change detection. The clear is
unconditional, so an unchanged repo's small manifest is stored again on every
add; its cognify stamps are kept and nothing is rebuilt. The
repository itself is read from its original location at cognify time. Remote
reads hold the clone lock and must match the commit recorded by add().

A GitHub/GitLab repository URL passed to ``add()`` takes the same path after a
shallow clone (``resolve_code_repository_url``): the clone directory is the
"original location" the manifest points at.
"""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import status

from cognee.modules.chunking.external_metadata import decode_external_metadata
from cognee.modules.engine.models.node_set import validate_node_set_names
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.resolve_repo import (
    CodeRepositoryError,
    _clone_lock,
    clone_dir,
    code_repo_clone_url,
    redact_repo_spec,
    repo_identity,
    repo_key,
    repository_commit,
    resolve_repo_source,
)
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


def partition_repo_files(directory: Path) -> tuple[list[Path], list[Path], list[Path]]:
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
    covered: list[Path] = []
    documents: list[Path] = []
    skipped: list[Path] = []

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


def build_repo_manifest(directory: Path, covered: list[Path]) -> str:
    """The repo item's raw data: covered files + a combined content hash.

    The hash covers every covered file's content, so the manifest text — and
    therefore the record's content hash — changes exactly when the code does,
    driving re-cognify through the normal content-change detection.
    """
    combined = hashlib.sha256()
    files = []
    for file_path in covered:
        relative = str(file_path.relative_to(directory))
        file_hash = hashlib.sha256()
        with file_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                file_hash.update(block)
        digest = file_hash.hexdigest()
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


class LegacyRepositoryError(CodeRepositoryError):
    """An existing remote graph must be preserved and rebuilt in a new dataset."""

    def __init__(self, where: str):
        super().__init__(
            message="This dataset contains legacy repository identities. Create a new dataset, "
            "call add(repo_url, dataset_name='new_code'), then cognify(datasets=['new_code']). "
            f"The existing data and graph are unchanged. (found: {where})",
            name="LegacyRepositoryError",
            status_code=status.HTTP_409_CONFLICT,
        )


class RepositoryChangedError(CodeRepositoryError):
    def __init__(self):
        super().__init__(
            message=(
                "The repository checkout changed or its manifest lacks the current selection. "
                "A shared checkout can be moved by another add() in any dataset. "
                "Re-add it to this dataset before cognify()."
            ),
            name="RepositoryChangedError",
            status_code=status.HTTP_409_CONFLICT,
        )


def manifest_repo_name(data_item) -> str | None:
    metadata = getattr(data_item, "system_metadata", None)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    url = metadata.get("repo_url") if isinstance(metadata, dict) else None
    return repo_key(url) if url else None


@asynccontextmanager
async def repository_read(
    directory: Path,
    *,
    remote: bool,
    expected_commit: str | None = None,
    expected_selection: str | None = None,
):
    """Serialize remote source and .enola reads with clone refreshes."""
    if not remote:
        yield None
        return
    async with _clone_lock(directory):
        commit = await repository_commit(directory)
        if expected_commit is not None and commit != expected_commit:
            raise RepositoryChangedError()
        from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

        snapshot = read_repository_snapshot(directory)
        actual_selection = snapshot.get("selection_fingerprint") if snapshot else None
        if expected_commit is not None and (
            not isinstance(expected_selection, str)
            or not expected_selection
            or actual_selection != expected_selection
        ):
            raise RepositoryChangedError()
        yield commit


async def _reject_legacy_repo_row(source_url, dataset_id):
    """Inspect remote provenance in this dataset, never a globally claimed name."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    async with get_relational_engine().get_async_session() as session:
        metadata_rows = (
            (
                await session.execute(
                    select(Data.system_metadata).where(
                        Data.dataset_id == dataset_id,
                        Data.system_metadata["source"].as_string() == "code_repo",
                    )
                )
            )
            .scalars()
            .all()
        )
    for metadata in metadata_rows:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        url = metadata.get("repo_url")
        if url and repo_key(url) == repo_key(source_url) and metadata.get("identity_version") != 2:
            raise LegacyRepositoryError("a remote repository row")


async def resolve_code_repository(
    directory: Path,
    user=None,
    dataset_id=None,
    source_url: str | None = None,
    include_documents: bool = True,
):
    async with repository_read(directory, remote=source_url is not None) as commit:
        item, documents, skipped = await _build_code_repository(
            directory, user, dataset_id, source_url, include_documents
        )
        if source_url is not None:
            manifest = json.loads(item.data)
            from cognee.tasks.code_graph.repo_selection import read_repository_snapshot

            snapshot = read_repository_snapshot(directory)
            manifest.update(commit=commit, repo_key=repo_key(source_url), identity_version=2)
            if snapshot is not None:
                manifest["repository_snapshot"] = snapshot

            item.data = json.dumps(manifest, indent=2)
            item.system_metadata["identity_version"] = 2
        return item, documents, skipped


async def _build_code_repository(
    directory: Path,
    user=None,
    dataset_id=None,
    source_url: str | None = None,
    include_documents: bool = True,
):
    """Build the repo-level DataItem (and the document file list) for a project.

    Returns (manifest_item, document_paths, skip_count). The data id is pinned
    to the stable identity (user, dataset, repo path) — mirroring DLT source
    manifests — so re-adding the same repo updates one record instead of
    accreting new ones; without a user context the id is left content-derived.
    With a pinned id and a dataset, the stored row's add stamp is cleared so
    the re-add reaches ingestion's content comparison (see the reset below).
    ``source_url`` (credential-free) records where a cloned repository came
    from as ``system_metadata["repo_url"]``.

    Without an LLM API key, the project's document files are excluded (logged)
    instead of emitted: their routes need an LLM (images transcribe at add
    time, text is LLM-chunked at cognify), so they would only fail later. The
    code graph itself never needs one — a key-less repo add still works fully.
    ``include_documents=False`` returns no document paths for code-only ingestion.
    """
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.tasks.ingestion.data_item import DataItem

    if source_url is not None and dataset_id is not None:
        await _reject_legacy_repo_row(source_url, dataset_id)

    covered, documents, skipped = partition_repo_files(directory)

    if not include_documents:
        documents = []
    elif documents and not get_llm_config().llm_api_key:
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

        data_id = await get_unique_data_id(
            f"code_repo:remote:{repo_identity(source_url)}"
            if source_url
            else f"code_repo:{directory}",
            user,
            dataset_id,
        )

    if data_id is not None and dataset_id is not None:
        from cognee.modules.data.methods import reset_data_pipeline_status

        # The id is pinned, so add()'s incremental check would find the stored
        # row already added and skip it before ingestion compares content: a
        # changed repo would keep its old manifest and graph. Clearing only the
        # add stamp lets the manifest through; ingestion then clears the other
        # stamps when the content changed, and keeps them when it did not, so
        # an unchanged repo whose later run fails is still marked as built.
        # The clear is unconditional: an unchanged repo's manifest is stored
        # again on every add, which costs one small write, not a rebuild.
        await reset_data_pipeline_status(data_id, dataset_id, pipeline_names=("add_pipeline",))

    system_metadata = {
        "source": "code_repo",
        "repo_path": str(directory),
        "file_count": len(covered),
    }
    if source_url is not None:
        system_metadata["repo_url"] = source_url

    manifest_item = DataItem(
        data=manifest_text,
        label=f"code_repo:{repo_key(source_url) if source_url else directory.name}",
        system_metadata=system_metadata,
        data_id=data_id,
    )

    logger.info(
        "Code project detected: 1 repo item covering %d file(s), %d document(s), %d skipped.",
        len(covered),
        len(documents),
        len(skipped),
    )
    return manifest_item, documents, len(skipped)


async def resolve_code_repository_url(
    spec: str,
    user=None,
    dataset_id=None,
    credentials: str | None = None,
    include_documents: bool = True,
    branch: str | None = None,
):
    """Clone a hosted repository URL and build its repo-level DataItem.

    A GitHub/GitLab repository URL (see ``resolve_repo.code_repo_clone_url``) is
    shallow-cloned into the shared clones directory and then partitioned
    exactly like a local code project -- one ``code_repo`` manifest for the
    CODE_REPO cognify route plus the repository's documents as individual
    items. Returns ``(manifest_item, document_paths, skip_count)``; the
    manifest carries the credential-free URL as ``repo_url``. ``credentials`` authenticates
    the clone of a private repository (see ``resolve_repo_source``); ``include_documents=False``
    returns the manifest alone. ``branch`` clones
    that branch instead of the remote's default; a dataset still holds one data
    row per URL, so adding it on another branch replaces that row's code graph.
    """
    clone_url = code_repo_clone_url(spec)
    if clone_url is None:
        raise ValueError(
            f"'{redact_repo_spec(spec)}' is not a repository URL add() can clone; "
            "expected https://github.com/<owner>/<repo>, a gitlab.com project, or a .git URL."
        )
    from cognee.modules.ingestion import save_data_to_file_detailed

    # Refresh, manifest and document copies observe one checkout. Documents must
    # live in ingestion storage: later cognify must not read a mutable clone.
    from cognee.tasks.code_graph.repo_resources import (
        DocumentCopyBudget,
        repository_disk_budget,
        stat_disk_bytes,
    )
    from cognee.tasks.code_graph.resolve_repo import default_clones_dir
    from cognee.tasks.ingestion.data_item import DataItem

    async with (
        repository_disk_budget(default_clones_dir()),
        _clone_lock(clone_dir(clone_url, branch)),
    ):
        repo_path = await resolve_repo_source(
            clone_url, credentials=credentials, branch=branch, include_documents=include_documents
        )
        source_url = redact_repo_spec(clone_url)
        manifest_item, documents, skipped = await resolve_code_repository(
            repo_path,
            user=user,
            dataset_id=dataset_id,
            source_url=source_url,
            include_documents=include_documents,
        )
        commit = json.loads(manifest_item.data)["commit"]
        copy_budget = await asyncio.to_thread(DocumentCopyBudget, default_clones_dir())
        stored_documents = []
        for path in documents:
            copy_budget.admit(stat_disk_bytes(path.stat()))
            with path.open("rb") as source:
                stored = await save_data_to_file_detailed(
                    source, filename=path.name, file_extension=path.suffix
                )
            stored_documents.append(
                DataItem(
                    data=stored,
                    external_metadata={
                        "source_uri": source_url,
                        "repo_path": path.relative_to(repo_path).as_posix(),
                        "repo_commit": commit,
                    },
                )
            )
        return manifest_item, stored_documents, skipped


async def add_code_repository(
    repo_path: Path,
    user,
    dataset,
    source_url: str | None = None,
    skip_connection_test: bool = False,
):
    """Store a resolved repository as its one code_repo Data row, without cognifying it.

    ``remember(content_type="code")`` builds the graph itself (the
    code_graph_pipeline over this row) and needs the row only for its id and
    its dataset listing. It is the same manifest item ``add(<repo>)`` ingests,
    pinned to the same identity, so both routes keep one record per repository.
    The repository's documents are not ingested. Returns the stored Data row.
    """
    from cognee.api.v1.add import add
    from cognee.modules.data.methods import get_data
    from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info

    manifest_item, _documents, _skipped = await resolve_code_repository(
        repo_path,
        user=user,
        dataset_id=dataset.id,
        source_url=source_url,
        include_documents=False,
    )
    add_result = await add(
        manifest_item,
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        user=user,
        skip_connection_test=skip_connection_test,
    )
    # With RAISE_INCREMENTAL_LOADING_ERRORS=false a failed ingest returns an
    # errored run instead of raising; surface its cause, not a missing row.
    errored = get_errored_run_info(add_result)
    if errored is not None:
        raise CodeRepositoryError(
            message=f"Could not store code repository '{repo_path}': "
            f"{errored.error_message or errored.error_class or 'ingestion failed'}"
        )

    data = await get_data(user.id, manifest_item.data_id, dataset.id)
    if data is None:
        raise ValueError(
            f"Code repository '{repo_path}' was added but its data record "
            f"{manifest_item.data_id} is missing from dataset {dataset.id}."
        )
    return data


async def extract_code_repo_graph(
    data_documents: list,
    ctx: Optional["PipelineContext"] = None,
) -> list:
    """Cognify CODE_REPO-route adapter: one enola run over the whole project.

    Reads the stored manifest for the repo path and runs the standard code
    graph tasks on the original directory under its clone lock. One repository node,
    cross-file edges, one graph read per repo. No LLM, no embeddings.

    A call-level node_set (add()'s ``node_set`` argument) reaches
    every document this manifest's repo carries, but not the code nodes
    themselves -- add_data_points only tags the DataPoints it is given, and
    the code graph tasks build their own DataPoints straight from the enola
    snapshot. This reads the same tag back off the manifest row's
    external_metadata (identical shape to classify_documents) and passes it
    down so every code node -- CodeRepository included -- carries it too.
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

        repo_name = manifest_repo_name(data_item)
        if repo_name and manifest.get("identity_version") != 2:
            raise LegacyRepositoryError("a remote manifest")
        if repo_name and (not manifest.get("commit") or manifest.get("repo_key") != repo_name):
            raise CodeRepositoryError(
                message="Invalid remote manifest; re-add the repository.",
                status_code=status.HTTP_409_CONFLICT,
            )

        repo_path = Path(manifest["repo_path"])
        if not repo_path.is_dir():
            raise ValueError(
                f"Code repository '{repo_path}' no longer exists; the repo item reads "
                "the original directory at cognify time. Re-add the repository from "
                "its current location."
            )

        metadata = decode_external_metadata(data_item.external_metadata)
        node_set = validate_node_set_names(metadata.get("node_set")) if metadata else None

        async with repository_read(
            repo_path,
            remote=repo_name is not None,
            expected_commit=manifest.get("commit"),
            expected_selection=manifest.get("repository_snapshot", {}).get("selection_fingerprint"),
        ):
            data_points = await extract_code_graph(
                repo_path=repo_path, node_set=node_set, repo_name=repo_name, namespace=repo_name
            )
            state = await add_code_graph_data_points(data_points, ctx=ctx, graph_only=True)
            await add_code_graph_edges(
                state,
                repo_path=repo_path,
                ctx=ctx,
                node_set=node_set,
                repo_name=repo_name,
                namespace=repo_name,
            )

        logger.info("Code repo graph extracted for %s (%s).", repo_path, data_item.id)

    return data_documents


def get_code_repo_tasks() -> list:
    """The cognify CODE_REPO-route task list: one adapter task, no LLM stages."""
    from cognee.modules.pipelines.tasks.task import Task

    return [Task(extract_code_repo_graph, needs_llm=False)]
