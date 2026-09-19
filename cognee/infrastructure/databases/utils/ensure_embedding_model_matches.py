"""Catch an embedding-model change at the dataset context, not inside the vector store.

A vector collection's width is fixed when it is created, so switching
``EMBEDDING_MODEL`` (or a keyless install gaining an LLM key and leaving the
local embedder) breaks every write and query on a dataset built with the old
model. The vector store reports that as an Arrow cast failure or "expected N
dimensions, not M" -- naming neither the model nor the dataset.

The dataset's registry row records the model that built it, in the
``vector_database_connection_info`` JSON the row already carries for
vector-store details, so the comparison happens once per dataset-context entry
against a row that is already loaded -- no extra call to the vector store on
any operation. A row from before the keys existed is completed once from the
store itself (LanceDB and pgvector), and never touched again.
"""

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_context_config,
    resolve_embedding_defaults,
)
from cognee.infrastructure.databases.vector.exceptions import EmbeddingDimensionMismatchError
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.modules.users.models import DatasetDatabase

MODEL_KEY = "embedding_model"
DIMENSIONS_KEY = "embedding_dimensions"


def embedding_model_record() -> dict:
    """The model and width a dataset is recorded as being embedded with.

    Resolved exactly as ``get_embedding_engine`` resolves them:
    ``resolve_embedding_defaults`` hands its ``(provider, model, dimensions)``
    to ``create_embedding_engine``, the engine keeps ``dimensions``, and every
    engine's ``get_vector_size()`` returns it -- which is the width
    ``create_collection`` gives the store's vector column. So the recorded
    width IS the engine's width, without building an engine here (constructing
    the local one loads its model).

    Reading the raw config instead would be wrong on a keyless install: with no
    embedding settings and no usable LLM key, ``resolve_embedding_defaults``
    reroutes embeddings to the local fastembed model (384), while the config
    still reads as the stock ``openai/text-embedding-3-large`` (3072) that
    nothing ever embeds with.
    """
    _provider, model, dimensions = resolve_embedding_defaults(
        get_embedding_context_config(), get_llm_context_config()
    )
    return {MODEL_KEY: model, DIMENSIONS_KEY: dimensions}


async def ensure_embedding_model_matches(dataset_database: DatasetDatabase) -> None:
    """Raise ``EmbeddingDimensionMismatchError`` if the configured model cannot serve this dataset.

    Runs with the dataset's vector configuration already bound, so a row with
    no recorded width can ask the store once. An empty store adopts the
    configured model; a store this cannot read (a provider without
    ``get_stored_vector_size``) is left to the store's own errors.
    """
    embedding_record = embedding_model_record()
    configured_model = embedding_record[MODEL_KEY]
    configured_dimensions = embedding_record[DIMENSIONS_KEY]
    info = dataset_database.vector_database_connection_info or {}
    stored_dimensions = info.get(DIMENSIONS_KEY)

    if stored_dimensions is None:
        vector_engine = await get_vector_engine_async()
        read_stored_size = getattr(vector_engine, "get_stored_vector_size", None)
        if read_stored_size is None:
            return
        stored_dimensions = await read_stored_size()
        if stored_dimensions is None:
            await _record_embedding_model(dataset_database, embedding_record)
            return
        # The model that built a legacy row is unknown; its width is what matters.
        await _record_embedding_model(
            dataset_database, {MODEL_KEY: None, DIMENSIONS_KEY: stored_dimensions}
        )

    if stored_dimensions == configured_dimensions:
        return

    raise EmbeddingDimensionMismatchError(
        dataset_id=dataset_database.dataset_id,
        stored_model=dataset_database.vector_database_connection_info.get(MODEL_KEY),
        stored_dimensions=stored_dimensions,
        configured_model=configured_model,
        configured_dimensions=configured_dimensions,
    )


async def _record_embedding_model(dataset_database: DatasetDatabase, record: dict) -> None:
    """Merge ``record`` into the row's stored connection info.

    Re-reads the row rather than writing the in-memory dict back: handlers add
    runtime-only values to that dict after loading (pgvector injects the
    database credentials), and those must never reach the table.
    """
    async with get_relational_engine().get_async_session() as session:
        row = await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_database.dataset_id)
        )
        row.vector_database_connection_info = {
            **(row.vector_database_connection_info or {}),
            **record,
        }
        await session.commit()
    dataset_database.vector_database_connection_info = {
        **(dataset_database.vector_database_connection_info or {}),
        **record,
    }


async def clear_embedding_model_records() -> None:
    """Forget every dataset's recorded model: the vectors it described are gone.

    Called after a vector prune that keeps the registry rows, so the next use
    of each dataset records whatever model builds it anew.
    """
    async with get_relational_engine().get_async_session() as session:
        for row in (await session.scalars(select(DatasetDatabase))).all():
            info = row.vector_database_connection_info or {}
            if MODEL_KEY in info or DIMENSIONS_KEY in info:
                row.vector_database_connection_info = {
                    key: value
                    for key, value in info.items()
                    if key not in (MODEL_KEY, DIMENSIONS_KEY)
                }
        await session.commit()
