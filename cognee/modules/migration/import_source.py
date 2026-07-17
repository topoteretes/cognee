"""Orchestrate importing a MemorySource into Cognee.

Called by ``cognee.remember()`` when it receives a :class:`MemorySource`.
Heavy Cognee imports happen lazily inside the function to avoid import cycles
(remember -> migration -> remember).

Two execution shapes:

- **streaming** (preserve mode + replayable source): records are passed over
  three times — data items chunk-stored via ``add()``, then a single pipeline
  task streams the graph in two passes (nodes, then facts) with bounded
  memory (see :func:`stream_graph_from_source`).
- **buffered** (re-derive/hybrid, or non-replayable sources): records are
  translated once into data items plus bounded graph batches, which run
  through the import pipeline together.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.migration.loader import (
    data_item_from_record,
    store_imported_graph,
    stream_graph_from_source,
    translate_record_stream,
    wrap_graph_batch,
)
from cognee.modules.migration.sources.base import MemorySource
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem
from .sources.langmem import LangMemSource
if TYPE_CHECKING:
    from cognee.api.v1.remember.remember import RememberResult

logger = get_logger("migration.import")

# Data items are stored in chunks of this size in the streaming path so raw
# content never fully materializes in memory.
DATA_ITEMS_PER_ADD = 200

_GRAPH_RECORD_KINDS = ("entity", "fact", "raw_node")


async def _ensure_user(user_payload: Dict[str, Any]):
    """Create-or-match a user by email, transferring credentials on creation.

    An existing target user is returned untouched — their credentials are
    never clobbered by an import. A missing user is created and then given
    the archived credentials (hashed password + account flags) directly, so
    restored accounts authenticate with their original passwords.
    """
    import secrets

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.methods import create_user, get_user_by_email
    from cognee.modules.users.models import User

    existing = await get_user_by_email(user_payload["email"])
    if existing is not None:
        return existing

    created = await create_user(user_payload["email"], secrets.token_urlsafe(32))
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        record = await session.get(User, created.id)
        record.hashed_password = user_payload["hashed_password"]
        record.is_active = user_payload.get("is_active", True)
        record.is_superuser = user_payload.get("is_superuser", False)
        record.is_verified = user_payload.get("is_verified", False)
        await session.commit()
    logger.info("Restored user %s from archive social layer.", user_payload["email"])
    return created


async def _resolve_import_user(source: MemorySource, user):
    """The identity the import runs as.

    Archives carrying a social layer import AS the archived dataset OWNER
    (created/matched by email first): per-dataset databases derive their
    physical location from the owner id, so ownership must be right BEFORE
    the rows land — it cannot be reassigned afterwards. All other imports
    run as the caller's user, exactly as before.

    Processing a social layer requires a SUPERUSER importer: the archive
    supplies emails, password hashes, and account flags verbatim, so an
    unprivileged importer could otherwise mint arbitrary accounts (including
    superusers) with credentials of their choosing — both via the SDK and via
    the /v1/remember archive-upload endpoint.
    """
    social_layer = getattr(source, "social_layer", None)
    owner_payload = (social_layer or {}).get("owner")
    if owner_payload is None:
        return user

    importer = user
    if importer is None:
        from cognee.modules.users.methods import get_default_user

        importer = await get_default_user()
    if not importer.is_superuser:
        from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

        raise PermissionDeniedError(
            message="Importing an archive that carries a social layer (permissions.json) "
            "requires a superuser: it restores user accounts and credentials."
        )
    return await _ensure_user(owner_payload)


async def _apply_social_grants(source: MemorySource, dataset_name: str, owner, importer) -> None:
    """Re-apply the archive's ACL grants onto the freshly imported dataset.

    Users are created/matched by email (credentials transfer on creation);
    ``give_permission_on_dataset`` deduplicates existing ACL rows, so
    re-importing is idempotent. The importing user additionally keeps read
    access — they held the archive bytes, and this prevents silent lockout
    when restoring someone else's dataset.
    """
    social_layer = getattr(source, "social_layer", None)
    if not social_layer:
        return

    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    datasets = await get_authorized_existing_datasets([dataset_name], "read", owner)
    if not datasets:
        logger.warning(
            "No dataset %r found after import; cannot restore its social layer.", dataset_name
        )
        return
    dataset_id = datasets[0].id

    for grant in social_layer.get("grants", []):
        principal = await _ensure_user(grant["user"])
        for permission_name in grant.get("permissions", []):
            await give_permission_on_dataset(principal, dataset_id, permission_name)

    if importer is None:
        from cognee.modules.users.methods import get_default_user

        importer = await get_default_user()
    if importer.id != owner.id:
        await give_permission_on_dataset(importer, dataset_id, "read")


def _revision_to_stamp(
    archive_revision: Optional[str],
    stored_revision: Optional[str],
    ordered_revisions: List[str],
) -> Optional[str]:
    """The revision the imported store should be re-stamped at, or None.

    Stamps only BACKWARD — when the archive's revision is strictly behind the
    store's current stamp — so the next migration run replays exactly
    archive -> head over the imported rows (idempotent over already-current
    data). Never forward: stamping ahead would mark the store's own data as
    migrated when it is not. Unknown revisions (either side written by newer
    code) and an unstamped store (None = base, already minimal) leave the
    stamp untouched.
    """
    if archive_revision is None or stored_revision is None:
        return None
    if archive_revision not in ordered_revisions or stored_revision not in ordered_revisions:
        return None
    if ordered_revisions.index(archive_revision) < ordered_revisions.index(stored_revision):
        return archive_revision
    return None


async def _restamp_to_source_revision(source: MemorySource, dataset_name: str, user) -> None:
    """Align the target's migration stamp with a cognee-origin archive.

    Preserve/hybrid imports write the archive's raw nodes with their
    source-store ids, so the target must not claim a newer data-migration
    revision than the exported data actually has. When the archive carries a
    revision behind the target's stamp, re-stamp backward; the next migration
    gate then replays revision -> head over the imported data. External
    sources carry no revision (their records are written entirely by
    current-code pipelines) and are skipped, as are re-derive imports (no raw
    nodes land).
    """
    archive_revision = getattr(source, "migration_revision", None)
    if archive_revision is None or source.mode == "re-derive":
        return

    from cognee.context_global_variables import backend_access_control_enabled
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.migrations.migration import order_migrations
    from cognee.modules.migrations.registry import MIGRATIONS
    from cognee.modules.migrations.runner import stamp_revisions

    ordered_revisions = [migration.revision for migration in order_migrations(MIGRATIONS)]
    if archive_revision not in ordered_revisions:
        logger.warning(
            "Archive migration revision %r is unknown to this chain — the archive was "
            "exported by newer code; leaving the store's migration stamp unchanged.",
            archive_revision,
        )
        return

    db_engine = get_relational_engine()
    if backend_access_control_enabled():
        from cognee.modules.data.methods import get_authorized_existing_datasets
        from cognee.modules.users.methods import get_default_user
        from cognee.modules.users.models import DatasetDatabase

        if user is None:
            user = await get_default_user()
        datasets = await get_authorized_existing_datasets([dataset_name], "read", user)
        if not datasets:
            logger.warning(
                "No dataset %r found after import; cannot align its migration stamp.",
                dataset_name,
            )
            return
        dataset_id = datasets[0].id
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, dataset_id)
        stored_revision = record.migration_revision if record else None
        target = _revision_to_stamp(archive_revision, stored_revision, ordered_revisions)
        if target is None:
            return
        await stamp_revisions(target=target, dataset_ids=[dataset_id])
    else:
        from cognee.modules.migrations.models import (
            GLOBAL_DATABASE_VERSION_ROW_ID,
            GlobalDatabaseVersion,
        )

        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        stored_revision = record.global_migration_revision if record else None
        target = _revision_to_stamp(archive_revision, stored_revision, ordered_revisions)
        if target is None:
            return
        await stamp_revisions(target=target)

    logger.info(
        "Stamped store back to archive migration revision %r (was %r); the next "
        "migration run replays %r -> head over the imported data.",
        target,
        stored_revision,
        target,
    )


def _pipeline_run_id(pipeline_result: Any) -> Optional[str]:
    """Extract the pipeline run id from a run_custom_pipeline return value.

    Blocking runs return ``{dataset_id: PipelineRunCompleted}``; background
    runs return ``{dataset_id: PipelineRunStarted}``. Both carry
    ``pipeline_run_id``.
    """
    if not pipeline_result:
        return None
    infos = pipeline_result.values() if isinstance(pipeline_result, dict) else [pipeline_result]
    for info in infos:
        run_id = getattr(info, "pipeline_run_id", None)
        if run_id:
            return str(run_id)
    return None


async def import_memory_source(
    source: MemorySource,
    dataset_name: str = "main_dataset",
    user=None,
    run_in_background: bool = False,
    node_set: Optional[list] = None,
    **kwargs,
) -> "RememberResult":
    """Import all records from a memory source into a dataset.

    Returns a RememberResult summarizing what was imported, carrying the
    migration pipeline's ``pipeline_run_id`` for polling when
    ``run_in_background=True`` (``status`` is then ``"started"`` and graph
    counts reflect scheduling, not completion). The import is idempotent:
    record-level deterministic ids (``data_id`` from external_system +
    external_id, node ids from entity names) make re-running an interrupted
    or repeated import safe.
    """
    from cognee.modules.migrations.startup import run_migrations_and_block

    # Imports are writes, so they take the same migration gate as
    # remember()/cognify() (the remember() MemorySource dispatch happens
    # before its own gate). This also records the data-migration revision —
    # stamping a fresh store at head — BEFORE the imported rows arrive;
    # without it the populated store has no recorded revision and the first
    # migration-aware startup replays the entire data chain over it. It must
    # run before user resolution: on a fresh store it also creates the
    # relational schema that user lookup needs.
    await run_migrations_and_block(dataset_name, user)

    # Archives carrying a social layer import AS the archived owner (see
    # _resolve_import_user); everything else runs as the caller's user.
    importer = user
    user = await _resolve_import_user(source, user)

    node_set = node_set or [f"import:{source.source_system}"]

    if source.mode == "preserve" and getattr(source, "replayable", False):
        result = await _import_streaming(source, dataset_name, user, run_in_background, node_set)
    else:
        result = await _import_buffered(
            source, dataset_name, user, run_in_background, node_set, **kwargs
        )

    # After the rows land: cognee-origin archives may need the migration
    # stamp aligned backward to the SOURCE store's revision (see
    # _restamp_to_source_revision) and their social layer restored (grants
    # re-applied for recreated users).
    await _restamp_to_source_revision(source, dataset_name, user)
    if getattr(source, "social_layer", None):
        await _apply_social_grants(source, dataset_name, owner=user, importer=importer)

    return result


async def _import_streaming(
    source: MemorySource,
    dataset_name: str,
    user,
    run_in_background: bool,
    node_set: list,
) -> "RememberResult":
    """Preserve-mode import with bounded memory.

    Pass A streams data items into chunked ``add()`` calls and counts every
    record kind; the graph then imports inside a single pipeline run whose
    task re-streams the source twice (nodes, then facts).
    """
    from cognee.api.v1.add import add
    from cognee.api.v1.remember.remember import RememberResult

    started_at = time.monotonic()

    counts: Dict[str, int] = {}
    pending: List[DataItem] = []
    data_items_stored = 0
    async for record in source.records():
        counts[record.kind] = counts.get(record.kind, 0) + 1
        data_item = data_item_from_record(record)
        if data_item is not None:
            pending.append(data_item)
            if len(pending) >= DATA_ITEMS_PER_ADD:
                await add(pending, dataset_name=dataset_name, user=user, node_set=node_set)
                data_items_stored += len(pending)
                pending = []
    if pending:
        await add(pending, dataset_name=dataset_name, user=user, node_set=node_set)
        data_items_stored += len(pending)

    logger.info("Importing from %s (mode=preserve, streaming): %s", source.source_system, counts)

    stats: Dict[str, int] = {
        "graph_nodes": 0,
        "graph_edges": 0,
        "skipped_facts": 0,
        "deduped_edges": 0,
    }
    pipeline_result = None
    has_graph_records = any(counts.get(kind) for kind in _GRAPH_RECORD_KINDS)
    if has_graph_records:
        from cognee.modules.pipelines.tasks.task import Task
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        async def stream_import_graph(items, ctx=None):
            return await stream_graph_from_source(source, stats, ctx=ctx)

        pipeline_data_item = DataItem(
            data={"source_system": source.source_system, "kind": "graph_stream"},
            label=f"migration-stream-{source.source_system}",
            external_metadata={
                "external_system": source.source_system,
                "kind": "graph_stream",
            },
            data_id=uuid5(NAMESPACE_OID, f"cogx-import:{source.source_system}:{dataset_name}"),
        )
        pipeline_result = await run_custom_pipeline(
            tasks=[Task(stream_import_graph)],
            data=[pipeline_data_item],
            dataset=dataset_name,
            user=user,
            run_in_background=run_in_background,
            pipeline_name="migration_import_pipeline",
        )

    backgrounded = run_in_background and has_graph_records
    if stats["skipped_facts"]:
        logger.warning(
            "Skipped %d facts with unresolvable UUID references during import from %s.",
            stats["skipped_facts"],
            source.source_system,
        )

    run_id = _pipeline_run_id(pipeline_result)
    import_summary = {
        "kind": "migration_import",
        "source_system": source.source_system,
        "mode": source.mode,
        "record_counts": counts,
        "graph_nodes": stats["graph_nodes"],
        "graph_edges": stats["graph_edges"],
        "skipped_facts": stats["skipped_facts"],
        "deduped_edges": stats["deduped_edges"],
        "pipeline_run_id": run_id,
    }
    if backgrounded:
        # Graph counts reflect scheduling, not completion: the pipeline is
        # still running. Poll via pipeline_run_id.
        import_summary["graph_import"] = "running"

    result = RememberResult(
        status="started" if backgrounded else "completed", dataset_name=dataset_name
    )
    result.pipeline_run_id = run_id
    result.raw_result = pipeline_result
    result.items_processed = data_items_stored + stats["graph_nodes"]
    result.items.append(import_summary)
    result.elapsed_seconds = time.monotonic() - started_at
    return result


async def _import_buffered(
    source: MemorySource,
    dataset_name: str,
    user,
    run_in_background: bool,
    node_set: list,
    **kwargs,
) -> "RememberResult":
    """Translate the full record stream, then run data items and graph batches."""
    from cognee.api.v1.remember.remember import RememberResult

    started_at = time.monotonic()

    # Translate the record stream directly: no full raw-record list is kept,
    # so peak memory is bounded by the translation output alone.
    translation = await translate_record_stream(
        source.records(),
        source.mode,
        # Cognee-origin archives keep source node UUIDs verbatim; other
        # systems get class-namespaced ids (see _register_entity).
        preserve_source_ids=source.source_system == "cognee",
    )

    logger.info(
        "Importing %d records from %s (mode=%s): %s",
        sum(translation.counts.values()),
        source.source_system,
        source.mode,
        translation.counts,
    )
    if translation.skipped_facts:
        logger.warning(
            "Skipped %d facts with unresolvable UUID references during import from %s.",
            translation.skipped_facts,
            source.source_system,
        )

    graph_nodes = sum(len(batch["nodes"]) for batch in translation.graph_batches)
    graph_edges = sum(len(batch["edges"]) for batch in translation.graph_batches)
    import_summary = {
        "kind": "migration_import",
        "source_system": source.source_system,
        "mode": source.mode,
        "record_counts": translation.counts,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "skipped_facts": translation.skipped_facts,
    }

    pipeline_result = None
    if translation.graph_batches:
        from cognee.modules.pipelines.tasks.task import Task
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        wrapped_batches = [
            wrap_graph_batch(batch, source.source_system, index)
            for index, batch in enumerate(translation.graph_batches)
        ]
        pipeline_result = await run_custom_pipeline(
            tasks=[Task(store_imported_graph)],
            data=wrapped_batches,
            dataset=dataset_name,
            user=user,
            run_in_background=run_in_background,
            pipeline_name="migration_import_pipeline",
        )

    run_id = _pipeline_run_id(pipeline_result)
    import_summary["pipeline_run_id"] = run_id
    backgrounded = run_in_background and bool(translation.graph_batches)
    if backgrounded:
        import_summary["graph_import"] = "running"

    if translation.data_items and translation.cognify_data_items:
        from cognee.api.v1.remember.remember import remember

        result = await remember(
            translation.data_items,
            dataset_name,
            run_in_background=run_in_background,
            node_set=node_set,
            user=user,
            **kwargs,
        )
        result.items.append(import_summary)
        result.items_processed += graph_nodes
        # The nested remember() owns result.pipeline_run_id (the cognify run);
        # the migration pipeline's run id stays in the import summary.
        if result.pipeline_run_id is None:
            result.pipeline_run_id = run_id
        return result

    if translation.data_items:
        # Preserve mode: store raw content so it is available for a later
        # cognify, but do not run LLM extraction now.
        from cognee.api.v1.add import add

        await add(
            translation.data_items,
            dataset_name=dataset_name,
            user=user,
            node_set=node_set,
        )

    result = RememberResult(
        status="started" if backgrounded else "completed", dataset_name=dataset_name
    )
    result.pipeline_run_id = run_id
    result.raw_result = pipeline_result
    result.items_processed = len(translation.data_items) + graph_nodes
    result.items.append(import_summary)
    result.elapsed_seconds = time.monotonic() - started_at
    return result
