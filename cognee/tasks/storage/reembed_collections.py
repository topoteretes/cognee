"""Rebuild the vectors of a vector store's collections with another embedding model.

A collection's rows are index rows: each holds the text its vector was computed
from (``IndexSchema.text``), so the vector can be computed again from the store
alone, without the source documents and without an LLM.

A collection's vector width is fixed when it is created, so a collection is
replaced, never changed in place, and it is not dropped before its replacement is
complete. The work is two steps, and the caller records between them which one a
store is in (``reembed_dataset``):

1. ``stage_collections`` embeds every row's text with the engine's embedding
   model into a *staging* collection next to each collection. The collections
   themselves are only read, so stopping here loses nothing. A second call makes
   each staging collection match its collection again: it embeds the rows that
   are not staged and removes the staged rows that are gone.
2. ``replace_collections`` drops each collection and writes it again from its
   staging collection, which holds every payload and the new vector. A second
   call finishes the collections the first did not reach.

The engine must already embed with the new model: ``upsert_raw_vectors`` creates a
collection at the width of the engine's embedding model. Nothing else should write
to the store between the first call and the last. A row written meanwhile is not
lost (it is embedded before its collection is replaced), but a row deleted or
changed after ``stage_collections`` returned is written back as it was staged.
"""

import hashlib

from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import (
    embedding_request_scope,
)
from cognee.infrastructure.databases.vector.exceptions import CollectionNotReembeddableError
from cognee.shared.logging_utils import get_logger

from .index_data_points import indexing_budget

logger = get_logger("reembed_collections")

# No collection cognee writes can have this name: collections are named
# ``<Type>_<field>`` after a DataPoint class.
STAGING_PREFIX = "__reembed_"

# Rows read, embedded and written per step. A step looks its rows up by id, which on
# LanceDB reads the whole collection, so at the size of an index write (128 rows) a
# large collection is slow: 200,000 rows took 8.7 minutes, and 1 minute at this size.
ROWS_PER_STEP = 1000


def staging_collection_name(collection_name: str) -> str:
    """The staging collection of ``collection_name``.

    A hash, not the name with a suffix: Postgres cuts a table name at 63 bytes, and
    two long collection names would then share one staging table.
    """
    return STAGING_PREFIX + hashlib.sha1(collection_name.encode()).hexdigest()[:16]


async def get_reembeddable_collection_names(vector_engine) -> list[str]:
    """The store's collections, without the staging collections of an earlier run."""
    return sorted(
        name
        for name in await vector_engine.get_collection_names()
        if not name.startswith(STAGING_PREFIX)
    )


async def stage_collections(vector_engine, collection_names: list[str]) -> int:
    """Make each collection's staging collection hold its rows, embedded with the new model.

    Returns the number of rows embedded by this call. Only for collections that
    ``replace_collections`` has not started on: a staged row with no row in its
    collection is removed here, and a collection half replaced lacks rows.

    Raises ``CollectionNotReembeddableError``, with every collection as it was, for
    a row whose payload has no ``text`` string: its vector was not computed from text.
    """
    embedded = 0
    for collection_name in collection_names:
        staging_name = staging_collection_name(collection_name)
        row_ids = set(await vector_engine.get_data_point_ids(collection_name))
        deleted_ids = set(await vector_engine.get_data_point_ids(staging_name)) - row_ids
        if deleted_ids:
            await vector_engine.delete_data_points(staging_name, sorted(deleted_ids))
        embedded += await _stage_unstaged_rows(vector_engine, collection_name)
        # An empty collection gets an empty staging collection, so it is replaced too:
        # left alone it would keep the old width and refuse its first write.
        await vector_engine.create_collection(staging_name)
    return embedded


async def replace_collections(vector_engine, collection_names: list[str]) -> int:
    """Replace each collection with its staging collection, then drop the staging one.

    Returns the number of rows embedded by this call: the rows written to a
    collection since ``stage_collections``.

    A collection without a staging collection was replaced by an earlier call. One
    dropped by a call that then stopped is written again from its staging
    collection, which is removed only after the copy.

    A collection is dropped only when its staging collection holds every one of its
    rows. Rows written to it since ``stage_collections`` are embedded first; if one
    is still missing after that, ``CogneeSystemError`` is raised with the collection
    untouched.
    """
    staging_names = set(await vector_engine.get_collection_names())
    embedded = 0
    for collection_name in collection_names:
        staging_name = staging_collection_name(collection_name)
        if staging_name not in staging_names:
            continue
        embedded += await _stage_unstaged_rows(vector_engine, collection_name)
        staged_ids = await vector_engine.get_data_point_ids(staging_name)
        unstaged = set(await vector_engine.get_data_point_ids(collection_name)) - set(staged_ids)
        if unstaged:
            raise CogneeSystemError(
                message=(
                    f"Collection '{collection_name}' holds {len(unstaged)} row(s) its staging "
                    "collection lacks, so something is writing to it. It was not changed; "
                    "run the re-embed again when nothing else uses the store."
                )
            )
        await vector_engine.delete_collection(collection_name)
        await vector_engine.create_collection(collection_name)
        for start in range(0, len(staged_ids), ROWS_PER_STEP):
            rows = await vector_engine.retrieve(
                staging_name, staged_ids[start : start + ROWS_PER_STEP], include_vector=True
            )
            await vector_engine.upsert_raw_vectors(
                collection_name,
                [
                    {
                        "id": str(row.id),
                        "vector": row.payload["vector"],
                        "payload": {
                            key: value for key, value in row.payload.items() if key != "vector"
                        },
                    }
                    for row in rows
                ],
            )
        await vector_engine.delete_collection(staging_name)
    return embedded


async def drop_staging_collections(vector_engine) -> None:
    """Remove every staging collection: undo a ``stage_collections`` that was not replaced."""
    for name in await vector_engine.get_collection_names():
        if name.startswith(STAGING_PREFIX):
            await vector_engine.delete_collection(name)


async def _stage_unstaged_rows(vector_engine, collection_name: str) -> int:
    """Embed the collection's rows that its staging collection lacks; returns how many."""
    staging_name = staging_collection_name(collection_name)
    staged_ids = set(await vector_engine.get_data_point_ids(staging_name))
    unstaged_ids = [
        row_id
        for row_id in await vector_engine.get_data_point_ids(collection_name)
        if row_id not in staged_ids
    ]
    requests = indexing_budget(vector_engine).requests
    if unstaged_ids:
        logger.info("Re-embedding %d row(s) of %s", len(unstaged_ids), collection_name)
    for start in range(0, len(unstaged_ids), ROWS_PER_STEP):
        rows = await vector_engine.retrieve(
            collection_name, unstaged_ids[start : start + ROWS_PER_STEP]
        )
        texts = [_embedded_text(collection_name, row) for row in rows]
        # A step's texts are embedded in provider-sized requests, as an index write's are.
        with embedding_request_scope(requests):
            vectors = await vector_engine.embed_data(texts)
        await vector_engine.upsert_raw_vectors(
            staging_name,
            [
                {"id": str(row.id), "vector": list(vector), "payload": row.payload}
                for row, vector in zip(rows, vectors, strict=True)
            ],
        )
        logger.info(
            "Re-embedded %d of %d row(s) of %s",
            min(start + ROWS_PER_STEP, len(unstaged_ids)),
            len(unstaged_ids),
            collection_name,
        )
    return len(unstaged_ids)


def _embedded_text(collection_name: str, row) -> str:
    """The text an index write embedded for this row (``DataPoint.get_embeddable_data``)."""
    text = (row.payload or {}).get("text")
    if not isinstance(text, str):
        raise CollectionNotReembeddableError(collection_name, row.id)
    return text.strip()
