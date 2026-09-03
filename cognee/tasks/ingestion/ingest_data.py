import json
import inspect
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4
from typing import TYPE_CHECKING, Union, BinaryIO, Any, List, Optional

import cognee.modules.ingestion as ingestion
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.ingestion.identify_many import identify_many
from cognee.modules.data.models import Data
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    resolve_data_id,
    load_or_create_datasets,
)

from cognee.shared.logging_utils import get_logger

from .save_data_item_to_storage import save_data_item_to_storage_detailed
from .carried_source import find_carried_source
from .data_item_to_text_file import data_item_to_text_file
from .data_item import DataItem

if TYPE_CHECKING:  # pragma: no cover - import cycle: pipelines imports this package
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger(__name__)


def _display_file_name(file_metadata: Optional[dict], actual_file_path: str) -> str:
    """The filename to show loaders for a payload, as the user would name it.

    ``FileMetadata`` splits a name into an extension-less ``name`` plus a
    separate ``extension``; loaders want them joined.
    """
    name = (file_metadata or {}).get("name")
    if not name:
        return os.path.basename(actual_file_path)

    extension = (file_metadata or {}).get("extension")

    return f"{name}.{extension}" if extension else name


def _pipeline_dataset_for(ctx, dataset_name: Optional[str], dataset_id: Optional[UUID], user: User):
    """The run's dataset from ``ctx`` when it is the one this call targets, else None.

    The pipeline sets ``ctx.dataset`` to the dataset it resolved (with write
    permission) for the run. Reuse it only when it demonstrably matches the
    caller's selector — by id, or by name for a dataset the caller owns — so a
    custom pipeline that points ``ingest_data`` at another dataset still goes
    through the full resolution + permission check.
    """
    pipeline_dataset = getattr(ctx, "dataset", None) if ctx is not None else None
    if pipeline_dataset is None:
        return None
    if dataset_id is not None:
        return pipeline_dataset if str(pipeline_dataset.id) == str(dataset_id) else None
    if (
        dataset_name is not None
        and getattr(pipeline_dataset, "name", None) == dataset_name
        and str(getattr(pipeline_dataset, "owner_id", None)) == str(user.id)
        and getattr(pipeline_dataset, "tenant_id", None) == getattr(user, "tenant_id", None)
    ):
        return pipeline_dataset
    return None


def _source_uri_from_input(data_item: Any) -> Optional[str]:
    """Return an origin locator without ever treating raw text as a URI.

    HTTP content is materialized into Cognee storage before metadata is read, so
    its original URL would otherwise be lost. File and object-storage locators
    are preserved as well; ordinary text input deliberately returns ``None``.
    """
    if isinstance(data_item, DataItem):
        metadata = data_item.external_metadata
        if isinstance(metadata, dict):
            explicit = metadata.get("source_uri")
            if isinstance(explicit, str) and explicit.strip():
                return explicit.strip()
        data_item = data_item.data

    if isinstance(data_item, str):
        parsed = urlparse(data_item)
        if parsed.scheme.lower() in {"http", "https", "s3", "file"}:
            return data_item
        try:
            path = Path(data_item)
            if path.is_absolute() or path.is_file():
                return path.resolve().as_uri()
        except (OSError, ValueError):
            return None
        return None

    filename = getattr(data_item, "filename", None) or getattr(data_item, "name", None)
    if isinstance(filename, str) and filename and not filename.startswith("<"):
        try:
            return Path(filename).resolve().as_uri()
        except (OSError, ValueError):
            return None
    return None


async def ingest_data(
    data: Any,
    dataset_name: str,
    user: User,
    node_set: Optional[List[str]] = None,
    dataset_id: UUID = None,
    preferred_loaders: dict[str, dict[str, Any]] = None,
    importance_weight: float = 0.5,
    ctx: "PipelineContext" = None,
):
    """Store ``data`` in ``dataset`` as ``Data`` rows (files land in storage first).

    ``ctx`` is injected by the pipeline machinery (any task with a ``ctx``
    parameter gets the run's ``PipelineContext``). When it carries the dataset
    this task writes to, that dataset was already resolved and write-checked by
    ``run_pipeline`` — re-resolving it here would cost three more DB sessions
    per call, and the incremental pipeline calls this task once per item.
    """
    if not user:
        user = await get_default_user()

    def get_external_metadata_dict(data_item: Union[BinaryIO, str, Any]) -> dict[str, Any]:
        if hasattr(data_item, "dict") and inspect.ismethod(getattr(data_item, "dict")):
            return {"metadata": data_item.dict(), "origin": str(type(data_item))}
        else:
            return {}

    async def store_data_to_dataset(
        data: Any,
        dataset_name: str,
        user: User,
        node_set: Optional[List[str]] = None,
        dataset_id: UUID = None,
        preferred_loaders: dict[str, dict[str, Any]] = None,
    ):
        import time as _time

        _store_start = _time.monotonic()
        logger.info("cognee-core store_to_dataset starting")

        new_datapoints = []
        existing_data_points = []
        # Originals replaced by a content-changed update: under content-
        # addressed keys the old object is not overwritten, so it is reclaimed
        # after the commit once nothing references it.
        replaced_original_locations = []

        if not isinstance(data, list):
            # Convert data to a list as we work with lists further down.
            data = [data]

        pipeline_dataset = _pipeline_dataset_for(ctx, dataset_name, dataset_id, user)
        if pipeline_dataset is not None:
            # The pipeline resolved (and write-authorized) this dataset once for
            # the whole run; reuse it instead of three lookups per item.
            dataset = pipeline_dataset
        elif dataset_id:
            # Retrieve existing dataset
            dataset = await get_specific_user_permission_datasets(user.id, "write", [dataset_id])
            # Convert from list to Dataset element
            if isinstance(dataset, list):
                dataset = dataset[0]
        else:
            # Find existing dataset or create a new one
            existing_datasets = await get_authorized_existing_datasets(
                user=user, permission_type="write", datasets=[dataset_name]
            )
            dataset = await load_or_create_datasets(
                dataset_names=[dataset_name],
                existing_datasets=existing_datasets,
                user=user,
            )
            if isinstance(dataset, list):
                dataset = dataset[0]

        db_engine = get_relational_engine()

        # Pre-loop: save files and compute content hashes (no DB calls).
        # Dedup is a dataset-scoped LOOKUP batched into a single WHERE-IN query
        # below — instead of one DB connection per file via ingestion.identify().
        precomputed_items = {}
        unique_content_hashes: set[str] = set()

        logger.info(
            "cognee-core store_to_dataset [loop 1/4] starting: save files and compute content hashes (%d items)",
            len(data),
        )
        _loop1_start = _time.monotonic()
        for data_item in data:
            underlying_data = data_item.data if isinstance(data_item, DataItem) else data_item
            item_data_id = data_item.data_id if isinstance(data_item, DataItem) else None
            source_uri = _source_uri_from_input(data_item)

            # The incremental wrapper already saved and hashed this item —
            # match by identity first; a path item whose string the in-chain
            # resolve_data_directories re-created matches by the stored path
            # its (I/O-free) save resolves to.
            carried = find_carried_source(ctx, data_item=data_item)
            if carried is None:
                stored = await save_data_item_to_storage_detailed(underlying_data)
                carried = find_carried_source(ctx, file_path=stored.file_path) or stored

            original_file_path = carried.file_path
            original_file_metadata = carried.metadata

            actual_file_path = get_data_file_path(original_file_path)

            if original_file_metadata is None:
                # Only items cognee did not write reach here (an s3:// URL, a
                # local path): their bytes were never in this process, so the
                # object has to be read to be described.
                async with open_data_file(actual_file_path) as file:
                    classified_data = ingestion.classify(file)
                    original_file_metadata = await classified_data.aget_metadata()

            item_content_hash = original_file_metadata["content_hash"]

            precomputed_items[id(data_item)] = {
                "original_file_path": original_file_path,
                "actual_file_path": actual_file_path,
                "original_file_metadata": original_file_metadata,
                "item_content_hash": item_content_hash,
                "item_data_id": item_data_id,
                "data_id": None,  # resolved below
                "source_uri": source_uri,
            }
            unique_content_hashes.add(item_content_hash)
        logger.info(
            "cognee-core store_to_dataset [loop 1/4] finished in %.3f seconds",
            _time.monotonic() - _loop1_start,
        )

        # All read-only lookups share ONE session: on the cloud pods every
        # session is a fresh TLS+SCRAM connection (NullPool), and this runs
        # once per item. The session is released before the loader/storage
        # work below so no connection is held idle across S3 round trips.
        logger.info(
            "cognee-core store_to_dataset [loop 2/4] starting: resolve data IDs (%d items)",
            len(data),
        )
        _loop2_start = _time.monotonic()
        existing_data_map: dict = {}
        async with db_engine.get_async_session() as session:
            # Single batch query: find existing rows for all content hashes in this
            # dataset+owner+tenant scope — replaces N per-file identify() calls.
            # identify_many() shares the exact same filter as identify() and chunks
            # large inputs to stay within SQLite's bind-parameter limit.
            existing_by_hash: dict[str, UUID] = await identify_many(
                list(unique_content_hashes), user, dataset.id, session=session
            )

            # Resolve pinned IDs (items with explicit data_id) — still needs DB for
            # resolve_data_id, but only for the subset that actually has a pinned id.
            # Unpinned items use the batch result above.
            batch_id_by_hash: dict[str, UUID] = {}
            data_point_ids = []
            for data_item in data:
                cached = precomputed_items[id(data_item)]
                item_data_id = cached["item_data_id"]
                item_content_hash = cached["item_content_hash"]

                if item_data_id is not None:
                    # A pinned id may be one the user held before a fork/update —
                    # resolve it (exact, then legacy) instead of minting a new row
                    # under a legacy value. Unknown pins stay as-is (dlt mints
                    # stable ids through this path deliberately).
                    resolved_pin = await resolve_data_id(dataset.id, item_data_id)
                    data_id = resolved_pin if resolved_pin is not None else item_data_id
                else:
                    # Use batch result; fall back to dedup-within-batch or mint new id
                    data_id = existing_by_hash.get(item_content_hash)
                    if data_id is None:
                        data_id = batch_id_by_hash.get(item_content_hash) or uuid4()
                batch_id_by_hash.setdefault(item_content_hash, data_id)

                cached["data_id"] = data_id
                data_point_ids.append(data_id)

            if data_point_ids:
                result = await session.execute(select(Data).filter(Data.id.in_(data_point_ids)))
                for dp in result.scalars().all():
                    existing_data_map[str(dp.id)] = dp

        logger.info(
            "cognee-core store_to_dataset [loop 2/4] finished in %.3f seconds",
            _time.monotonic() - _loop2_start,
        )

        # Ids already present in THIS dataset. Only rows this batch resolved
        # to can be in here (identify_many is dataset-scoped and pins are
        # re-resolved against the dataset), so the targeted lookup above is
        # enough — loading every Data row of the dataset per call, as before,
        # made a 164-file add read O(N^2) rows for a membership check.
        dataset_data_map = {
            str(dp.id): True
            for dp in existing_data_map.values()
            if str(dp.dataset_id) == str(dataset.id)
        }

        logger.info(
            "cognee-core store_to_dataset [loop 3/4] starting: process files and build data records (%d items)",
            len(data),
        )
        _loop3_start = _time.monotonic()
        for data_item in data:
            # Support for DataItem (custom label + data + optional data_id / external_metadata)
            current_label = None
            underlying_data = data_item
            item_data_id = None
            item_external_metadata = None
            item_system_metadata = None

            if isinstance(data_item, DataItem):
                underlying_data = data_item.data
                current_label = data_item.label
                item_data_id = data_item.data_id
                item_external_metadata = data_item.external_metadata
                item_system_metadata = data_item.system_metadata

            # Retrieve cached intermediate results from pre-loop to avoid re-processing
            cached = precomputed_items.get(id(data_item), {})
            actual_file_path = cached.get("actual_file_path")

            # Store all input data as text files in Cognee data storage.
            # Ingestion context rides to the loader: loaders that own routing
            # (dlt) need the dataset/user to build their staging artifacts.
            cognee_storage_file_path, loader_engine = await data_item_to_text_file(
                actual_file_path,
                preferred_loaders,
                dataset_name=dataset.name,
                dataset_id=dataset.id,
                user=user,
                # The name the user knows this payload by. Taken from the
                # metadata rather than the storage key, which is content
                # addressed for payloads cognee wrote — loaders surface this
                # (dlt names its source from it), so it has to stay a real
                # filename. Falls back to the key's basename for pass-through
                # items, where the key IS the user's path (file:// URIs carry
                # percent-encoding, e.g. spaces as %20, so decode first).
                original_file_name=_display_file_name(
                    cached.get("original_file_metadata"), actual_file_path
                ),
            )

            if loader_engine is None:
                raise IngestionError("Loader cannot be None")

            # Use data_id computed in pre-loop
            data_id = cached.get("data_id")

            # A loader may return a LoaderResult instead of a plain path: it
            # then owns the record's identity and route stamp (dlt: the
            # manifest's stable data_id + the system_metadata routing stamp).
            # The pinned id replaces the pre-loop mint, and existence is
            # re-resolved for it so re-adds hit the update branch.
            storage_file_metadata = None

            if isinstance(cognee_storage_file_path, LoaderResult):
                loader_result = cognee_storage_file_path
                cognee_storage_file_path = loader_result.file_path
                # The loader described the text it wrote, from the content it
                # still had in hand.
                storage_file_metadata = loader_result.file_metadata
                if loader_result.system_metadata is not None and item_system_metadata is None:
                    item_system_metadata = loader_result.system_metadata
                if loader_result.data_id is not None:
                    data_id = loader_result.data_id
                    if str(data_id) not in existing_data_map:
                        async with db_engine.get_async_session() as session:
                            pinned_row = await session.get(Data, data_id)
                            if pinned_row is not None:
                                existing_data_map[str(pinned_row.id)] = pinned_row

            # The original was described in the pre-loop, from the payload's own
            # bytes. Re-opening it here downloaded and re-hashed the whole object
            # a second time per item for a byte-identical result.
            original_file_metadata = cached["original_file_metadata"]

            if storage_file_metadata is None:
                # A loader that returned a bare path did not describe its output,
                # so the stored text has to be read back to be described.
                async with open_data_file(cognee_storage_file_path) as file:
                    classified_data = ingestion.classify(file)
                    storage_file_metadata = await classified_data.aget_metadata()

            data_point = existing_data_map.get(str(data_id))

            # TODO: Maybe allow getting of external metadata through ingestion loader?
            ext_metadata = get_external_metadata_dict(data_item)

            # Merge DataItem.external_metadata if present
            if item_external_metadata:
                ext_metadata.update(item_external_metadata)

            source_uri = cached.get("source_uri")
            if source_uri:
                cognee_metadata = ext_metadata.get("_cognee")
                if not isinstance(cognee_metadata, dict):
                    cognee_metadata = {}
                    ext_metadata["_cognee"] = cognee_metadata
                cognee_metadata.setdefault("source_uri", source_uri)

            if node_set:
                ext_metadata["node_set"] = node_set

            if data_point is not None:
                # Content-change detection: reset pipeline_status when content changed
                new_content_hash = original_file_metadata["content_hash"]
                content_changed = str(data_point.content_hash) != str(new_content_hash)

                new_original_location = original_file_metadata["file_path"]
                if content_changed and data_point.original_data_location != new_original_location:
                    replaced_original_locations.append(data_point.original_data_location)

                # Rows are dataset-scoped (the startup migration backfills
                # legacy rows). A row of another dataset can only reach this
                # branch through a mispinned data_id — never mutate it.
                if str(data_point.dataset_id) != str(dataset.id):
                    # The owning dataset's id goes to the operator log only —
                    # naming it in the exception would tell a caller with no
                    # access to that dataset which dataset holds this id.
                    logger.info(
                        "foreign-pin refused: data %s belongs to dataset %s, pinned from %s",
                        data_point.id,
                        data_point.dataset_id,
                        dataset.id,
                    )
                    raise IngestionError(
                        f"Data id {data_point.id} is not available in dataset {dataset.id}."
                    )

                data_point.name = original_file_metadata["name"]
                data_point.raw_data_location = cognee_storage_file_path
                data_point.original_data_location = original_file_metadata["file_path"]
                data_point.extension = storage_file_metadata["extension"]
                data_point.mime_type = storage_file_metadata["mime_type"]
                data_point.original_extension = original_file_metadata["extension"]
                data_point.original_mime_type = original_file_metadata["mime_type"]
                data_point.loader_engine = loader_engine.loader_name
                data_point.owner_id = user.id
                data_point.content_hash = new_content_hash
                data_point.raw_content_hash = storage_file_metadata["content_hash"]
                data_point.data_size = original_file_metadata["file_size"]
                data_point.external_metadata = ext_metadata
                if item_system_metadata is not None:
                    data_point.system_metadata = item_system_metadata
                data_point.node_set = json.dumps(node_set) if node_set else None
                data_point.tenant_id = user.tenant_id if user.tenant_id else None
                # Absent means "leave unchanged": a re-ingest without a label
                # (current_label None) must not clear a previously stored one.
                if current_label is not None:
                    data_point.label = current_label

                if content_changed:
                    data_point.pipeline_status = {}

                existing_data_points.append(data_point)
                dataset_data_map[str(data_point.id)] = True
            else:
                if str(data_id) in dataset_data_map:
                    continue

                data_point = Data(
                    id=data_id,
                    dataset_id=dataset.id,
                    name=original_file_metadata["name"],
                    raw_data_location=cognee_storage_file_path,
                    original_data_location=original_file_metadata["file_path"],
                    extension=storage_file_metadata["extension"],
                    mime_type=storage_file_metadata["mime_type"],
                    original_extension=original_file_metadata["extension"],
                    original_mime_type=original_file_metadata["mime_type"],
                    loader_engine=loader_engine.loader_name,
                    owner_id=user.id,
                    content_hash=original_file_metadata["content_hash"],
                    raw_content_hash=storage_file_metadata["content_hash"],
                    external_metadata=ext_metadata,
                    system_metadata=item_system_metadata,
                    node_set=json.dumps(node_set) if node_set else None,
                    data_size=original_file_metadata["file_size"],
                    tenant_id=user.tenant_id if user.tenant_id else None,
                    pipeline_status={},
                    token_count=-1,
                    label=current_label,
                    importance_weight=importance_weight,
                )

                new_datapoints.append(data_point)
                dataset_data_map[str(data_point.id)] = True
        logger.info(
            "cognee-core store_to_dataset [loop 3/4] finished in %.3f seconds",
            _time.monotonic() - _loop3_start,
        )

        logger.info(
            "cognee-core store_to_dataset [loop 4/4] starting: DB commit (%d new, %d updated)",
            len(new_datapoints),
            len(existing_data_points),
        )
        _loop4_start = _time.monotonic()
        async with db_engine.get_async_session() as session:
            for data_point in existing_data_points:
                await session.merge(data_point)
            session.add_all(new_datapoints)
            await session.commit()
        logger.info(
            "cognee-core store_to_dataset [loop 4/4] finished in %.3f seconds",
            _time.monotonic() - _loop4_start,
        )

        # After the commit, the updated rows point at their new originals; the
        # replaced objects are removed unless another row still shares them.
        for replaced_location in replaced_original_locations:
            await db_engine.remove_data_file_if_unreferenced(replaced_location)

        _elapsed = _time.monotonic() - _store_start
        logger.info("cognee-core store_to_dataset finished in %.3f seconds", _elapsed)
        return existing_data_points + new_datapoints

    # Concurrent ingests PINNED to the same data_id (dlt derives stable ids;
    # update() re-ingests under the document's own id) can both try to INSERT
    # the same primary key — the loser hits "UNIQUE constraint failed: data.id";
    # retrying re-reads the committed row and takes the existing-data branch.
    # Unpinned rows mint random ids and dedup by lookup instead: a concurrent
    # same-content race there can produce two rows, which under document
    # semantics are simply two documents. File writes are content-addressed,
    # so re-running is idempotent.
    try:
        return await store_data_to_dataset(
            data, dataset_name, user, node_set, dataset_id, preferred_loaders
        )
    except IntegrityError:
        return await store_data_to_dataset(
            data, dataset_name, user, node_set, dataset_id, preferred_loaders
        )
