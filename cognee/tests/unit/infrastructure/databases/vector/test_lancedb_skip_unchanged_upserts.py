"""Re-upserting unchanged DataPoints must neither re-embed nor rewrite them.

Every cognify re-indexes the entities and edge types it touches, and most of
them are already stored with the same text. Each upsert used to embed every
point again and merge_insert every row, and LanceDB commits a new table
version (a fragment, plus deletion files for the replaced rows) for every
merge_insert, changed or not. Over a few days of session syncing that left a
production store with 11k versions and 73k deletion files in one collection,
15 GB on disk for ~200k rows, and a vector worker that grew until the kernel
killed it.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed"),
]

COLLECTION = "Note_text"


class _CountingEmbeddingEngine:
    """Text-dependent vectors, so a reused vector is distinguishable from a fresh one."""

    def __init__(self):
        self.embedded: list[str] = []

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        self.embedded.extend(texts)
        return [[float(len(text)), float(len(self.embedded)), 0.5] for text in texts]


class _Note(DataPoint):
    text: str
    note: str = ""
    metadata: dict = {"index_fields": ["text"]}


@pytest_asyncio.fixture(params=["in_process", "subprocess"])
async def adapter(request, tmp_path):
    engine = _CountingEmbeddingEngine()
    url = str(tmp_path / "db")
    if request.param == "in_process":
        lance = LanceDBAdapter(url=url, api_key=None, embedding_engine=engine)
    else:
        lance = LanceDBAdapter.create_subprocess(url=url, api_key=None, embedding_engine=engine)
    lance.test_url = url
    # Created up front, as production callers do: create_data_points on a
    # missing collection re-enters VECTOR_DB_LOCK inside create_collection.
    await lance.create_collection(COLLECTION, _Note)
    try:
        yield lance
    finally:
        await lance.close()


def _version_count(adapter) -> int:
    return len(os.listdir(os.path.join(adapter.test_url, f"{COLLECTION}.lance", "_versions")))


async def _stored(adapter, ids):
    rows = await adapter.retrieve(COLLECTION, [str(point_id) for point_id in ids])
    return {str(row.id): row for row in rows}


async def _vectors(adapter, ids):
    collection = await adapter.get_collection(COLLECTION)
    rows = await collection.query().to_list()
    return {row["id"]: list(row["vector"]) for row in rows if row["id"] in {str(i) for i in ids}}


async def test_identical_reupsert_embeds_nothing_and_commits_no_version(adapter):
    ids = [uuid4(), uuid4()]
    await adapter.create_data_points(
        COLLECTION, [_Note(id=ids[0], text="alpha"), _Note(id=ids[1], text="beta")]
    )
    embedded = list(adapter.embedding_engine.embedded)
    versions = _version_count(adapter)

    # Fresh instances: created_at / updated_at differ, content does not.
    await adapter.create_data_points(
        COLLECTION, [_Note(id=ids[0], text="alpha"), _Note(id=ids[1], text="beta")]
    )

    assert adapter.embedding_engine.embedded == embedded
    assert _version_count(adapter) == versions


async def test_whitespace_only_change_reuses_the_vector_but_stores_the_text(adapter):
    """The embedded text is stripped, the stored payload is not."""
    point_id = uuid4()
    await adapter.create_data_points(COLLECTION, [_Note(id=point_id, text="beta")])
    before = await _vectors(adapter, [point_id])
    adapter.embedding_engine.embedded.clear()

    await adapter.create_data_points(COLLECTION, [_Note(id=point_id, text=" beta ")])

    assert adapter.embedding_engine.embedded == []
    assert await _vectors(adapter, [point_id]) == before
    assert (await _stored(adapter, [point_id]))[str(point_id)].payload["text"] == " beta "


async def test_changed_text_reembeds_only_that_point(adapter):
    kept, changed = uuid4(), uuid4()
    await adapter.create_data_points(
        COLLECTION, [_Note(id=kept, text="alpha"), _Note(id=changed, text="beta")]
    )
    before = await _vectors(adapter, [kept, changed])
    adapter.embedding_engine.embedded.clear()
    versions = _version_count(adapter)

    await adapter.create_data_points(
        COLLECTION, [_Note(id=kept, text="alpha"), _Note(id=changed, text="gamma")]
    )

    assert adapter.embedding_engine.embedded == ["gamma"]
    after = await _vectors(adapter, [kept, changed])
    assert after[str(kept)] == before[str(kept)]
    assert after[str(changed)] != before[str(changed)]
    assert (await _stored(adapter, [changed]))[str(changed)].payload["text"] == "gamma"
    assert _version_count(adapter) == versions + 1


async def test_changed_payload_is_written_with_the_stored_vector(adapter):
    point_id = uuid4()
    await adapter.create_data_points(COLLECTION, [_Note(id=point_id, text="alpha", note="v1")])
    before = await _vectors(adapter, [point_id])
    adapter.embedding_engine.embedded.clear()

    await adapter.create_data_points(COLLECTION, [_Note(id=point_id, text="alpha", note="v2")])

    assert adapter.embedding_engine.embedded == []
    assert (await _stored(adapter, [point_id]))[str(point_id)].payload["note"] == "v2"
    assert await _vectors(adapter, [point_id]) == before


async def test_new_tag_is_merged_without_reembedding(adapter):
    point_id = uuid4()
    await adapter.create_data_points(
        COLLECTION, [_Note(id=point_id, text="alpha", belongs_to_set=["A"])]
    )
    adapter.embedding_engine.embedded.clear()
    versions = _version_count(adapter)

    await adapter.create_data_points(
        COLLECTION, [_Note(id=point_id, text="alpha", belongs_to_set=["B"])]
    )
    assert adapter.embedding_engine.embedded == []
    assert sorted(
        (await _stored(adapter, [point_id]))[str(point_id)].payload["belongs_to_set"]
    ) == ["A", "B"]
    assert _version_count(adapter) == versions + 1

    # Re-sending a tag the row already carries changes nothing.
    await adapter.create_data_points(
        COLLECTION, [_Note(id=point_id, text="alpha", belongs_to_set=["A"])]
    )
    assert _version_count(adapter) == versions + 1


async def test_mixed_batch_embeds_only_new_points(adapter):
    stored = uuid4()
    await adapter.create_data_points(COLLECTION, [_Note(id=stored, text="alpha")])
    adapter.embedding_engine.embedded.clear()

    new = uuid4()
    await adapter.create_data_points(
        COLLECTION, [_Note(id=stored, text="alpha"), _Note(id=new, text="delta")]
    )

    assert adapter.embedding_engine.embedded == ["delta"]
    assert set(await _stored(adapter, [stored, new])) == {str(stored), str(new)}
