"""Journey: update() with a chunk task registered (CLO-397).

A plugin screens every chunk with ``register_chunk_task``. Editing one sentence of
a document must still take the chunk-level path, and the function must see the
chunk the edit produced before it is extracted or stored.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.api.v1.update.incremental import RefusalReason
from cognee.modules.cognify import register_chunk_task, unregister_chunk_task
from cognee.modules.data.methods import get_dataset_data
from cognee.tests.journeys import _support

DATASET = "journey_chunk_task_update"

WORKSHOPS = [
    ("ropewalk", "Adaeze Wren"),
    ("sailmaking", "Tomas Ferreira"),
    ("cooperage", "Ingrid Solvay"),
    ("chandlery", "Yusuf Haldane"),
    ("smithy", "Olamide Farquhar"),
    ("boatyard", "Mirela Okonkwo"),
]
DOC_V1 = "\n\n".join(
    f"Section {index}. The {name} workshop in Quillon is run by {person}, who has kept its "
    f"ledgers since the harbour reopened and trains two apprentices every spring season."
    for index, (name, person) in enumerate(WORKSHOPS)
)
DOC_V2 = DOC_V1.replace("Olamide Farquhar", "Bronagh Castellane")
DOC_V3 = DOC_V2.replace("Adaeze Wren", "Leocadia Brandt")


async def _ingest(default_user):
    """One document cut into several chunks; its dataset id and Data row."""
    await cognee.add(DOC_V1, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET], chunk_size=48)
    dataset_id = await _support.dataset_id_by_name(DATASET, default_user)
    stored = await _support.vector_texts(dataset_id, default_user)
    assert len(stored) > 3, f"the document must span several chunks, got {len(stored)}"
    return dataset_id, (await get_dataset_data(dataset_id))[0]


@pytest.mark.journey
@pytest.mark.asyncio
async def test_update_stays_incremental_with_a_chunk_task_registered(clean_env, default_user):
    seen: list[str] = []

    async def screen_chunks(chunks, ctx):
        seen.extend(chunk.text for chunk in chunks)
        return chunks

    register_chunk_task("screen", screen_chunks)
    try:
        dataset_id, row = await _ingest(default_user)
        assert seen, "the function saw no chunk during cognify"

        seen.clear()
        result = await cognee.update(
            data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
        )
    finally:
        unregister_chunk_task("screen")

    assert result["status"] == "incremental", f"update rebuilt the document: {result}"
    assert result["fallback"] is None
    # Only the chunk the edit produced is fresh, and the function saw it.
    assert len(seen) == 1 and "Castellane" in seen[0], seen

    texts = await _support.vector_texts(dataset_id, default_user)
    assert any("castellane" in text for text in texts), "updated chunk was not stored"
    assert not any("farquhar" in text for text in texts), "stale chunk survived the update"


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_chunk_task_that_drops_the_edited_chunk(clean_env, default_user):
    async def drop_castellane(chunks, ctx):
        return [chunk for chunk in chunks if "Castellane" not in chunk.text]

    register_chunk_task("screen", drop_castellane)
    try:
        dataset_id, row = await _ingest(default_user)

        dropped = await cognee.update(
            data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
        )
        assert dropped["status"] == "incremental", f"update rebuilt the document: {dropped}"
        texts = await _support.vector_texts(dataset_id, default_user)
        assert not any("castellane" in text for text in texts), "the dropped chunk was stored"
        assert not any("farquhar" in text for text in texts), "stale chunk survived the update"

        # The stored chunks no longer tile the document, so the next edit rebuilds
        # it, and the rebuild's cognify drops the chunk again.
        rebuilt = await cognee.update(
            data_id=row.id, data=DOC_V3, dataset_id=dataset_id, user=default_user
        )
    finally:
        unregister_chunk_task("screen")

    assert rebuilt["status"] == "full_rebuild", rebuilt
    assert rebuilt["fallback"]["reason"] is RefusalReason.CHUNKS_NOT_TILING
    texts = await _support.vector_texts(dataset_id, default_user)
    assert any("brandt" in text for text in texts), "the rebuild lost the second edit"
    assert not any("castellane" in text for text in texts), "the dropped chunk came back"
