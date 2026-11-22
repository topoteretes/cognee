"""Dataset routing on a real LanceDB store: repeated syncs keep the routing table compact.

The routing collection lives in the deployment's default store, which no
cognify run compacts. Every sync that rewrites a row appends a fragment and
marks the old row deleted, so without a pass of its own the table, and the
manifest every read loads, grows with each index build.
"""

from importlib import import_module
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.vector.config import VectorConfig
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.search.dataset_routing import (
    DATASET_ROUTE_COLLECTION,
    dataset_route_ids,
    sync_dataset_routes,
)
from cognee.tasks.summarization.models import GlobalContextSummary

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")

index_module = import_module("cognee.modules.search.dataset_routing.index")
factory = import_module("cognee.infrastructure.databases.vector.create_vector_engine")


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _IndexStore:
    """The dataset's own store, holding the rows the index build just embedded."""

    def __init__(self, texts):
        self.texts = texts

    async def retrieve(self, collection_name, data_point_ids, *, include_vector=False):
        return [
            ScoredResult(
                id=UUID(row_id),
                score=0,
                payload={"id": row_id, "text": self.texts[row_id], "vector": [0.1, 0.2, 0.3]},
            )
            for row_id in data_point_ids
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("subprocess_enabled", [True, False], ids=["worker", "in_process"])
async def test_repeated_syncs_keep_the_routing_table_compact(
    tmp_path, monkeypatch, subprocess_enabled
):
    dataset_id = uuid4()
    documents = [uuid4() for _ in range(20)]
    bucket_ids = dataset_route_ids(dataset_id, documents)[1:]
    db_path = tmp_path / "default.lancedb"
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    store = factory.create_vector_engine(
        **VectorConfig(
            vector_db_provider="lancedb",
            vector_db_url=str(db_path),
            vector_db_name="default",
            vector_db_subprocess_enabled=subprocess_enabled,
        ).to_dict()
    )

    async def route_engine():
        return store

    async def data_ids(dataset_ids=None):
        return {dataset_id: documents}

    monkeypatch.setattr(index_module, "get_dataset_route_engine", route_engine)
    monkeypatch.setattr(index_module, "get_data_ids_by_dataset", data_ids)
    monkeypatch.setattr(index_module, "backend_access_control_enabled", lambda: True)

    texts = {row_id: "first build" for row_id in bucket_ids}
    rounds = 15
    try:
        for round_number in range(rounds + 1):
            if round_number:
                # Each index build changes one document's bucket, as an improve run does.
                texts[bucket_ids[round_number - 1]] = f"rebuilt in round {round_number}"
            desired = [
                GlobalContextSummary(
                    id=UUID(row_id), text=text, dataset_id=str(dataset_id), level=0
                )
                for row_id, text in texts.items()
            ]
            await sync_dataset_routes(str(dataset_id), desired, _IndexStore(texts))

        rows = await store.retrieve(DATASET_ROUTE_COLLECTION, bucket_ids)
    finally:
        await store.close()

    import lance

    table_path = db_path / f"{DATASET_ROUTE_COLLECTION}.lance"
    fragments = len(lance.dataset(str(table_path)).get_fragments())
    assert fragments <= 2, (
        f"{rounds} routing syncs left {fragments} fragments in {DATASET_ROUTE_COLLECTION}; "
        "the default store is never compacted"
    )
    assert {str(row.id): row.payload["text"] for row in rows} == texts
