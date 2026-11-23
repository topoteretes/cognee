from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import DatasetDatabase
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    # Imported for the annotation only: the vector package's imports lead back here.
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

logger = get_logger("reembed_dataset")


@dataclass(frozen=True)
class ReembeddedDataset:
    """What ``reembed_dataset`` left a dataset with."""

    embedding_model: str | None
    embedding_dimensions: int
    # The collections whose vectors were computed again.
    collections: list[str]
    # Rows this call embedded: fewer than the dataset holds when it continued an
    # earlier call, none when that call had only the replacing left.
    rows_embedded: int


async def reembed_dataset(
    dataset_id: UUID, embedding_config: EmbeddingConfig | None = None
) -> ReembeddedDataset:
    """Compute a dataset's vectors again with the configured embedding model.

    The model is the deployment's (the ``.env`` embedding settings, as every other
    call resolves them), unless ``embedding_config`` names another.

    Every collection of the dataset's vector store is rebuilt from the text its rows
    hold, so no document is read, nothing is extracted and no LLM is called; the
    cost is one embedding per stored row. The graph is not touched. Afterwards the
    dataset records the new model, and is used with it like a dataset built with it.

    No collection is dropped before its replacement is complete (see
    ``reembed_collections``), and the dataset's record carries the progress:

    * A failure before the first collection is replaced (the embedding provider, a
      row with no text) leaves the dataset as it was; what was embedded so far is
      discarded.
    * While a call runs, and after one that was interrupted or failed later, every
      other use of the dataset raises ``DatasetReembedIncompleteError``. Calling
      this again with the same model continues where that call stopped.

    The dataset's routing rows are copies of its vectors: the old ones are removed,
    and they are written again when the new model is the deployment's and the
    routing collection has its width (after an ``EMBEDDING_MODEL`` change it has
    the old width until it is dropped, ``prune_dataset_routes``). A failure to
    write them is raised with the dataset already converted and without routing
    rows; ``copy_dataset_routes`` writes them later.

    Runs under the dataset lock, like a pipeline run, and checks no permission. An
    operation already inside the dataset when this starts (in another process, or a
    search in this one) is not stopped: a row it writes is still embedded, but one
    it deletes or changes after its collection was staged is written back as
    staged. Run it where nothing else uses the dataset.

    Args:
        dataset_id: The dataset whose vectors to rebuild.
        embedding_config: The model to embed with, when it is not the deployment's.
            Resolved as for any call that takes an ``embedding_config``.

    Raises:
        CogneeValidationError: Backend access control is off, so the datasets share
            one vector store; the dataset has a truth subspace, which is computed
            from vectors and not rebuilt here; or an unfinished re-embed of the
            dataset used another model.
        CollectionNotReembeddableError: A collection holds a row with no text.
        CogneeSystemError: Something else is using the dataset.
        NotImplementedError: The vector adapter cannot list its collections and rows.
    """
    # Imported here: the context module's imports lead back to this package.
    from cognee.context_global_variables import (
        DatabaseContextManager,
        backend_access_control_enabled,
    )
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
    from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache
    from cognee.infrastructure.locks import dataset_lock

    if not backend_access_control_enabled():
        raise CogneeValidationError(
            message=(
                "A single dataset cannot be re-embedded with backend access control off: "
                "every dataset shares one vector store."
            )
        )

    async with dataset_lock(dataset_id):
        # Read first, and without this call's model in force: a dataset that is refused
        # stays as it is, down to the engine its other users get. This is also where
        # an adapter that cannot list its collections and rows raises.
        async with DatabaseContextManager(dataset_id, check_embedding_model=False):
            await _require_no_truth_subspace(dataset_id, await get_vector_engine_async())

        async with DatabaseContextManager(
            dataset_id, embedding_config=embedding_config, check_embedding_model=False
        ):
            try:
                return await _reembed(dataset_id)
            finally:
                # An engine created in here embeds with this call's model, whatever became
                # of the dataset. Evicted, the next use creates it with the model in force
                # then; an operation still using it keeps it until it is done.
                vector_engine_cache.evict(**get_vectordb_context_config())


async def reembed_dataset_in_context(dataset_id: UUID) -> ReembeddedDataset:
    """Re-embed the dataset whose context the caller is in, with the model in force there.

    For a data migration. The migration runner is already inside the dataset's context
    and holds its queue slot, and the dataset lock ``reembed_dataset`` takes must be
    taken before that slot (SDK-483), so it cannot be taken any more. The runner holds
    the migration lock instead, and every other entry into the dataset is refused while
    the re-embed runs.

    Raises what ``reembed_dataset`` raises, but for access control being off: without
    it there is no dataset context to be in.
    """
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
    from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache

    await _require_no_truth_subspace(dataset_id, await get_vector_engine_async())
    try:
        return await _reembed(dataset_id)
    finally:
        # As in reembed_dataset: the next use creates the engine with the model in force then.
        vector_engine_cache.evict(**get_vectordb_context_config())


async def abandon_reembed_in_context(dataset_id: UUID) -> None:
    """Undo an unfinished re-embed of the dataset whose context the caller is in.

    For a re-embed that stopped before it replaced a collection (the process was
    killed while it embedded): every collection is still as it was, so the staging
    collections are dropped and the dataset keeps the model that built it.

    Raises:
        CogneeValidationError: The re-embed had started replacing collections. The
            dataset holds vectors of two models and can only be finished, with the
            model the re-embed was moving it to.
    """
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        MODEL_KEY,
        REEMBED_KEY,
        STAGED_COLLECTIONS_KEY,
        record_reembed_finished,
    )
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.tasks.storage.reembed_collections import drop_staging_collections

    dataset_database = await _get_dataset_database(dataset_id)
    unfinished = dataset_database.vector_database_connection_info.get(REEMBED_KEY)
    if unfinished is None:
        return
    if STAGED_COLLECTIONS_KEY in unfinished:
        raise CogneeValidationError(
            message=(
                f"Dataset {dataset_id} is partly re-embedded with '{unfinished[MODEL_KEY]}': "
                "some of its collections are already replaced. It can only be finished, "
                "with that model in force."
            )
        )
    await drop_staging_collections(await get_vector_engine_async())
    await record_reembed_finished(dataset_database)
    logger.info("Dataset %s: the unfinished re-embed was undone", dataset_id)


async def _reembed(dataset_id: UUID) -> ReembeddedDataset:
    """``reembed_dataset`` inside the dataset's context, with the new model in force."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        DIMENSIONS_KEY,
        MODEL_KEY,
        REEMBED_KEY,
        STAGED_COLLECTIONS_KEY,
        embedding_model_record,
        record_reembed_finished,
        record_reembed_progress,
    )
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
    from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    from cognee.infrastructure.databases.vector.embeddings.vector_size import engine_vector_size
    from cognee.modules.search.dataset_routing import copy_dataset_routes, remove_dataset_routes
    from cognee.modules.truth_subspace.constants import TRUTH_CENTROID_COLLECTION
    from cognee.tasks.storage.reembed_collections import (
        drop_staging_collections,
        get_reembeddable_collection_names,
        replace_collections,
        stage_collections,
    )

    from .get_data_ids_by_dataset import get_data_ids_by_dataset

    dataset_database = await _get_dataset_database(dataset_id)
    embedding_engine = get_embedding_engine()
    target_model: str | None = embedding_model_record()[MODEL_KEY]
    target_dimensions = await engine_vector_size(embedding_engine)
    target = {MODEL_KEY: target_model, DIMENSIONS_KEY: target_dimensions}

    unfinished = dataset_database.vector_database_connection_info.get(REEMBED_KEY)
    if unfinished is not None and any(unfinished[key] != target[key] for key in target):
        raise CogneeValidationError(
            message=(
                f"Dataset {dataset_id} has an unfinished re-embed with "
                f"'{unfinished[MODEL_KEY]}' ({unfinished[DIMENSIONS_KEY]} dimensions). "
                "Finish it with that model before re-embedding with another one."
            )
        )

    # From here every other entry into the dataset is refused.
    if unfinished is None:
        await record_reembed_progress(dataset_database, target)

    # A cached engine keeps the embedding model it was created with. Evicted, the
    # engine is created again below, in this context, with the new model.
    vector_engine_cache.evict(**get_vectordb_context_config())
    vector_engine = await get_vector_engine_async()

    rows_embedded = 0
    collections = (unfinished or {}).get(STAGED_COLLECTIONS_KEY)
    if collections is None:
        try:
            await _require_embeds_with(vector_engine, embedding_engine)
            collections = [
                name
                for name in await get_reembeddable_collection_names(vector_engine)
                # Turso lists a collection in lowercase.
                if name.casefold() != TRUTH_CENTROID_COLLECTION.casefold()
            ]
            rows_embedded = await stage_collections(vector_engine, collections)
        except Exception:
            # Every collection is still as it was: undo, so the dataset stays usable.
            await drop_staging_collections(vector_engine)
            await record_reembed_finished(dataset_database)
            raise
        await record_reembed_progress(
            dataset_database, {**target, STAGED_COLLECTIONS_KEY: collections}
        )
    else:
        await _require_embeds_with(vector_engine, embedding_engine)

    # Holds no rows (see _require_no_truth_subspace) and has the old width: the next
    # truth-subspace build creates it again.
    await vector_engine.delete_collection(TRUTH_CENTROID_COLLECTION)
    rows_embedded += await replace_collections(vector_engine, collections)

    documents = await get_data_ids_by_dataset([dataset_id])
    await remove_dataset_routes(dataset_id, documents.get(dataset_id, []))
    await record_reembed_finished(dataset_database, target)
    await copy_dataset_routes(dataset_id, vector_engine)

    logger.info(
        "Dataset %s is now embedded with %s (%d dimensions): %d collection(s)",
        dataset_id,
        target_model,
        target_dimensions,
        len(collections),
    )
    return ReembeddedDataset(
        embedding_model=target_model,
        embedding_dimensions=target_dimensions,
        collections=collections,
        rows_embedded=rows_embedded,
    )


async def _require_no_truth_subspace(dataset_id: UUID, vector_engine) -> None:
    """Refuse a dataset that holds truth-subspace centroids.

    They are computed from vectors, not from text, and the chunks' coordinates in
    the graph are relative to them. Dropping them would restart their epoch, which
    is how a search tells current coordinates from stale ones.
    """
    from cognee.modules.truth_subspace.constants import TRUTH_CENTROID_COLLECTION

    for name in await vector_engine.get_collection_names():
        # Turso lists a collection in lowercase.
        if name.casefold() != TRUTH_CENTROID_COLLECTION.casefold():
            continue
        if await vector_engine.get_data_point_ids(name):
            raise CogneeValidationError(
                message=(
                    f"Dataset {dataset_id} has a truth subspace (improve() built it from "
                    "session learnings). It is computed from the dataset's vectors and "
                    "re-embedding does not rebuild it, so the dataset was not changed."
                )
            )


async def _require_embeds_with(vector_engine, embedding_engine) -> None:
    """Refuse a vector engine that does not embed with ``embedding_engine``'s model.

    The engine is created with the model in force when it is first used. Another
    operation that was already inside the dataset can create it first, with its own.
    """
    from cognee.infrastructure.databases.vector.embeddings.vector_size import engine_vector_size

    bound = vector_engine.embedding_engine
    if bound.model != embedding_engine.model or await engine_vector_size(
        bound
    ) != await engine_vector_size(embedding_engine):
        raise CogneeSystemError(
            message=(
                f"The dataset's vector engine embeds with '{bound.model}', not with "
                f"'{embedding_engine.model}': something else is using the dataset. "
                "Run the re-embed again when nothing else does."
            )
        )


async def _get_dataset_database(dataset_id: UUID) -> DatasetDatabase:
    async with get_relational_engine().get_async_session() as session:
        rows = await session.scalars(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
        )
        return rows.one()
