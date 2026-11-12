"""One contract for moving a document's vectors from one dataset's store to another's.

Promoting a document between datasets copies its graph nodes and edges together
with their embeddings, so the target dataset never re-embeds what the source
already paid for. The vector half of that is two adapter calls:
``retrieve(..., include_vector=True)`` reads the stored vectors of the document's
rows back from the source store, and ``upsert_raw_vectors`` writes them into the
target store as they are. The embedding engine is called for the original ingest
only. The same checks run on every in-tree adapter with two stores each, standing
for two datasets: LanceDB and Turso on temporary files, pgvector against a local
Postgres when one is listening (``cognee`` / ``cognee`` on 5432, as the other live
tests assume).
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

DIM = 4


class CopyPoint(DataPoint):
    text: str
    rank: int
    metadata: dict = {"index_fields": ["text"]}


class CountingEmbedder:
    """Deterministic embeddings keyed on a word, counting every call."""

    def __init__(self):
        self.calls = 0
        self.tokenizer = None

    async def embed_text(self, texts):
        self.calls += 1
        return [
            [1.0 if "apple" in t else 0.0, 1.0 if "pear" in t else 0.0, 0.0, 0.1] for t in texts
        ]

    def get_vector_size(self) -> int:
        return DIM

    def get_dimensions(self) -> int:
        return DIM

    def get_batch_size(self) -> int:
        return 100

    @property
    def model(self) -> str:
        return "counting"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


async def _exercise_promotion(source, target, embedder: CountingEmbedder):
    """Ingest into the source dataset's store, promote the rows into the target's, compare.

    The collection keeps its name across stores, as it does across datasets
    (``DocumentChunk_text`` is ``DocumentChunk_text`` in every dataset).
    """
    collection = f"DocumentChunk_text_{uuid4().hex[:8]}"
    source_collection = target_collection = collection
    apple = CopyPoint(text="apple", rank=1, belongs_to_set=["proj_a"])
    pear = CopyPoint(text="pear", rank=2)
    await source.create_collection(source_collection, payload_schema=CopyPoint)
    await source.create_data_points(source_collection, [apple, pear])
    # The target already holds the apple row under its own tag (ids are
    # content-derived, so a promoted entity often exists in the target already).
    await target.create_collection(target_collection, payload_schema=CopyPoint)
    await target.create_data_points(
        target_collection, [CopyPoint(id=apple.id, text="apple", rank=1, belongs_to_set=["proj_b"])]
    )
    ingest_calls = embedder.calls
    assert ingest_calls >= 1

    # Plain retrieve carries no vector; include_vector carries the stored one.
    plain = await source.retrieve(source_collection, [str(apple.id)])
    assert "vector" not in plain[0].payload
    rows = await source.retrieve(
        source_collection, [str(apple.id), str(pear.id)], include_vector=True
    )
    by_id = {str(row.id): row for row in rows}
    assert [round(v, 3) for v in by_id[str(apple.id)].payload["vector"]] == [1.0, 0.0, 0.0, 0.1]
    assert [round(v, 3) for v in by_id[str(pear.id)].payload["vector"]] == [0.0, 1.0, 0.0, 0.1]

    # Promote: the target store never embeds.
    await target.upsert_raw_vectors(
        target_collection,
        [
            {
                "id": row.id,
                "vector": row.payload["vector"],
                "payload": {k: v for k, v in row.payload.items() if k != "vector"},
            }
            for row in rows
        ],
        payload_schema=CopyPoint,
    )
    assert embedder.calls == ingest_calls

    copied = await target.retrieve(target_collection, [str(apple.id)], include_vector=True)
    assert copied[0].payload["text"] == "apple"
    assert [round(v, 3) for v in copied[0].payload["vector"]] == [1.0, 0.0, 0.0, 0.1]
    # The copy brings the source's tag and keeps the target's own.
    assert set(copied[0].payload["belongs_to_set"]) == {"proj_a", "proj_b"}
    nearest = await target.search(
        target_collection, query_vector=[0.0, 1.0, 0.0, 0.0], limit=1, include_payload=True
    )
    assert str(nearest[0].id) == str(pear.id)
    assert embedder.calls == ingest_calls

    # A second raw upsert of an existing id replaces payload and vector alike.
    await target.upsert_raw_vectors(
        target_collection,
        [
            {
                "id": apple.id,
                "vector": [0.0, 0.0, 1.0, 0.0],
                "payload": {"id": str(apple.id), "text": "apple", "rank": 9},
            }
        ],
        payload_schema=CopyPoint,
    )
    replaced = await target.retrieve(target_collection, [str(apple.id)], include_vector=True)
    assert replaced[0].payload["rank"] == 9
    assert [round(v, 3) for v in replaced[0].payload["vector"]] == [0.0, 0.0, 1.0, 0.0]
    # ... but never the stored tags, even when the point carries none.
    assert set(replaced[0].payload["belongs_to_set"]) == {"proj_a", "proj_b"}
    assert embedder.calls == ingest_calls

    # The same id twice in one call: the last point wins, their tags are unioned.
    await target.upsert_raw_vectors(
        target_collection,
        [
            {
                "id": pear.id,
                "vector": [0.0, 1.0, 0.0, 0.0],
                "payload": {"id": str(pear.id), "text": "pear", "rank": 5, "belongs_to_set": ["x"]},
            },
            {
                "id": pear.id,
                "vector": [0.0, 1.0, 0.0, 0.1],
                "payload": {"id": str(pear.id), "text": "pear", "rank": 6, "belongs_to_set": ["y"]},
            },
        ],
        payload_schema=CopyPoint,
    )
    twice = await target.retrieve(target_collection, [str(pear.id)], include_vector=True)
    assert len(twice) == 1
    assert twice[0].payload["rank"] == 6
    assert set(twice[0].payload["belongs_to_set"]) == {"x", "y"}
    assert [round(v, 3) for v in twice[0].payload["vector"]] == [0.0, 1.0, 0.0, 0.1]

    # A bad point is refused before anything is written: a vector of the wrong
    # width or a payload that is not a dict, and the collection is not created.
    untouched = f"{collection}_untouched"
    with pytest.raises(ValueError):
        await target.upsert_raw_vectors(
            untouched,
            [
                {
                    "id": uuid4(),
                    "vector": [1.0, 0.0],
                    "payload": {"id": "x", "text": "short", "rank": 0},
                }
            ],
            payload_schema=CopyPoint,
        )
    with pytest.raises(ValueError):
        await target.upsert_raw_vectors(
            untouched,
            [{"id": uuid4(), "vector": [1.0, 0.0, 0.0, 0.0]}],
            payload_schema=CopyPoint,
        )
    assert not await target.has_collection(untouched)
    assert embedder.calls == ingest_calls


@pytest.mark.asyncio
async def test_lancedb_promotes_vectors_between_stores():
    pytest.importorskip("lancedb")
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    embedder = CountingEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        source = LanceDBAdapter(url=str(Path(tmp) / "src"), api_key=None, embedding_engine=embedder)
        target = LanceDBAdapter(url=str(Path(tmp) / "dst"), api_key=None, embedding_engine=embedder)
        await _exercise_promotion(source, target, embedder)


@pytest.mark.asyncio
async def test_turso_promotes_vectors_between_stores():
    pytest.importorskip("turso", reason="pyturso driver not installed")
    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
        TursoVectorAdapter,
    )

    embedder = CountingEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        source = TursoVectorAdapter(
            url=str(Path(tmp) / "src.db"), api_key=None, embedding_engine=embedder
        )
        target = TursoVectorAdapter(
            url=str(Path(tmp) / "dst.db"), api_key=None, embedding_engine=embedder
        )
        await _exercise_promotion(source, target, embedder)


@pytest.mark.asyncio
@pytest.mark.skipif(not _port_open("localhost", 5432), reason="no local Postgres on 5432")
async def test_pgvector_promotes_vectors_between_stores():
    asyncpg = pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    async def _admin():
        return await asyncpg.connect(
            host="localhost", port=5432, user="cognee", password="cognee", database="postgres"
        )

    databases = [f"vector_copy_{uuid4().hex[:8]}" for _ in range(2)]
    admin = await _admin()
    for database in databases:
        await admin.execute(f"CREATE DATABASE {database}")
    await admin.close()
    for database in databases:
        setup = await asyncpg.connect(
            host="localhost", port=5432, user="cognee", password="cognee", database=database
        )
        await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await setup.close()

    embedder = CountingEmbedder()
    source, target = (
        PGVectorAdapter(
            connection_string=f"postgresql+asyncpg://cognee:cognee@localhost:5432/{database}",
            api_key=None,
            embedding_engine=embedder,
        )
        for database in databases
    )
    try:
        await _exercise_promotion(source, target, embedder)
    finally:
        await source.engine.dispose()
        await target.engine.dispose()
        admin = await _admin()
        for database in databases:
            await admin.execute(f"DROP DATABASE {database} (FORCE)")
        await admin.close()
