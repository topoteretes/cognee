import os
from urllib.parse import urlparse
from typing import Any, BinaryIO, List, Optional, Union

from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    get_dataset_data,
)
from cognee.modules.data.methods.delete_data import delete_data
from cognee.modules.graph.methods.delete_data_nodes_and_edges import (
    delete_data_nodes_and_edges,
)
from cognee.modules.graph.methods.has_data_related_nodes import has_data_related_nodes
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

logger = get_logger()


def _normalize_path(path: str) -> str:
    """Normalize a local path (or file:// URL) for cross-platform comparison.

    Non-local inputs (raw text, ``s3://`` URLs) are returned unchanged so they are
    never mistaken for a filesystem path.
    """
    if not isinstance(path, str) or not path:
        return path
    if urlparse(path).scheme == "s3":
        return path
    if path.startswith("file://"):
        parsed = urlparse(path)
        path = parsed.path
        # Windows file URLs parse to "/C:/dir/file" — drop the leading slash.
        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return os.path.normcase(path)


async def incremental_update(
    data: Union[BinaryIO, List[BinaryIO], str, List[str], Any],
    dataset_name: str = "main_dataset",
    user: User = None,
    prune_removed: bool = True,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    incremental_loading: bool = True,
) -> dict:
    """Incrementally sync a set of source paths into a dataset.

    Instead of rebuilding the whole graph, this re-ingests only what actually
    changed and prunes what was removed:

    - **added / modified / unchanged** are handled by ``add()`` +
      ``cognify(incremental_loading=True)``. Unchanged inputs skip LLM extraction
      (their pipeline status stays ``COMPLETED``); modified inputs re-cognify
      because ``ingest_data`` resets the pipeline status when ``content_hash``
      changes. No parallel manifest is kept — the ``Data`` table is the source of
      truth, so it can never drift from the graph.
    - **removed** sources (recorded in the dataset but no longer present under the
      synced paths) are pruned via ``delete_data_nodes_and_edges`` + ``delete_data``.
      Entities that other still-present sources also contribute are preserved by the
      delete path's existing co-ownership check, so shared nodes are never orphaned.

    Pruning is scoped to the synced paths: ``incremental_update("./docs", ...)`` only
    ever prunes rows ingested from under ``./docs``, never data added from elsewhere.
    Removal is only inferred inside directories that currently exist and were scanned,
    so a missing/unmounted/mistyped path is never read as a mass deletion. On the first
    run (dataset absent) nothing is pruned.

    Returns a summary: ``{"dataset_name", "processed", "removed", "cognify"}``.
    """
    # Ensure the databases/tables exist before we resolve the user or query the
    # dataset — `incremental_update` may be the very first cognee call (e.g. a
    # git hook on a fresh checkout), before any add() has run setup(). Idempotent.
    await setup()
    if not user:
        user = await get_default_user()

    # Expand directories to their files so we can diff against recorded sources.
    current_items = await resolve_data_directories(data, include_subdirectories=True)
    current_paths = {_normalize_path(item) for item in current_items if isinstance(item, str)}

    removed = 0
    if prune_removed:
        existing = await get_authorized_existing_datasets(
            datasets=[dataset_name], permission_type="write", user=user
        )
        # Removal is only inferred inside directories that actually exist and were
        # scanned this run. A root that is not an existing directory (typo, unmounted
        # drive, wrong cwd) contributes no scan root, so a transient or mistaken path
        # can never be read as "everything was deleted" and wipe the dataset's graph.
        data_list = data if isinstance(data, list) else [data]
        scan_roots = [
            _normalize_path(item)
            for item in data_list
            if isinstance(item, str) and os.path.isdir(item)
        ]
        if existing and scan_roots:
            dataset = existing[0]
            for record in await get_dataset_data(dataset.id):
                location = getattr(record, "original_data_location", None)
                if not location:
                    continue
                # original_data_location is stored as a file:// URI for local files;
                # _normalize_path strips it back to a comparable absolute path.
                norm = _normalize_path(location)
                if not any(norm.startswith(root + os.sep) for root in scan_roots):
                    continue  # not under a scanned directory
                if norm in current_paths:
                    continue  # still present
                if os.path.exists(norm):
                    continue  # never prune a file that still exists on disk
                try:
                    if await has_data_related_nodes(dataset.id, record.id):
                        await delete_data_nodes_and_edges(dataset.id, record.id, user.id)
                    await delete_data(record, dataset.id)
                    removed += 1
                except Exception:
                    logger.warning(
                        "incremental_update: failed to prune removed source %s",
                        getattr(record, "id", "?"),
                        exc_info=True,
                    )

    cognify_run = None
    if current_items:
        await add(
            data=current_items,
            dataset_name=dataset_name,
            user=user,
            node_set=node_set,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
            incremental_loading=incremental_loading,
        )
        cognify_run = await cognify(
            datasets=[dataset_name],
            user=user,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
            incremental_loading=incremental_loading,
        )

    logger.info(
        "incremental_update: dataset=%s processed=%d removed=%d",
        dataset_name,
        len(current_paths),
        removed,
    )
    return {
        "dataset_name": dataset_name,
        "processed": len(current_paths),
        "removed": removed,
        "cognify": cognify_run,
    }
