"""Tests for VectorDBInterface.update_payload across the in-tree adapters.

The contract under test: payload fields change, the stored vector does not,
and — the whole point — the embedding engine is never called.
"""

import socket
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint


class PayloadPoint(DataPoint):
    text: str
    chunk_index: int
    metadata: dict = {"index_fields": ["text"]}


class CountingMockEmbedder:
    """Deterministic embedder that counts every embed call."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self.tokenizer = None
        self.calls = 0

    async def embed_text(self, texts):
        self.calls += 1
        return [[float(len(t) % 7)] * self.dimensions for t in texts]

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_dimensions(self) -> int:
        return self.dimensions

    @property
    def model(self) -> str:
        return "counting-mock"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


async def _exercise_update_payload(engine, embedder: Optional[CountingMockEmbedder] = None):
    """Shared contract check for any adapter instance."""
    collection = f"payload_upd_{uuid4().hex[:8]}"
    points = [
        PayloadPoint(text="first chunk text", chunk_index=0),
        PayloadPoint(text="second chunk text", chunk_index=1),
    ]
    await engine.create_collection(collection, payload_schema=PayloadPoint)
    await engine.create_data_points(collection, points)

    calls_before = embedder.calls if embedder else None

    assert engine.supports_payload_update is True
    await engine.update_payload(
        collection,
        {str(points[0].id): {"chunk_index": 41}, str(points[1].id): {"chunk_index": 42}},
    )

    rows = await engine.retrieve(collection, [str(p.id) for p in points])
    by_id = {str(r.id): r.payload for r in rows}
    assert by_id[str(points[0].id)]["chunk_index"] == 41
    assert by_id[str(points[1].id)]["chunk_index"] == 42
    assert by_id[str(points[0].id)]["text"] == "first chunk text"  # untouched fields survive

    if embedder is not None:
        assert embedder.calls == calls_before, "payload update must not embed anything"

    # Caller contract: fields must already exist in the payload schema. An
    # unknown field is refused rather than dropped (LanceDB) or drifted into
    # the JSON payload (PGVector, Turso).
    with pytest.raises(ValueError, match="not_a_payload_field"):
        await engine.update_payload(collection, {str(points[0].id): {"not_a_payload_field": 1}})
    rows = await engine.retrieve(collection, [str(points[0].id)])
    assert rows[0].payload["chunk_index"] == 41, "a refused update must change nothing"

    # Vector integrity: similarity search still finds the rows.
    found = await engine.search(collection, query_text="first chunk text", limit=2)
    assert len(found) >= 1

    # Missing ids are skipped silently.
    await engine.update_payload(collection, {str(uuid4()): {"chunk_index": 99}})


@pytest.mark.asyncio
async def test_lancedb_update_payload():
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    embedder = CountingMockEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        engine = LanceDBAdapter(
            url=str(Path(tmp) / "lance"), api_key=None, embedding_engine=embedder
        )
        await _exercise_update_payload(engine, embedder)


@pytest.mark.asyncio
@pytest.mark.skipif(not _port_open("localhost", 5432), reason="no local Postgres on 5432")
async def test_pgvector_update_payload():
    # An open port is not enough: a machine can be running Postgres while
    # cognee is installed without the postgres extra, and a bare import here
    # fails the test instead of skipping it. The adapter needs both halves of
    # that extra — the driver to connect and pgvector for the column type.
    asyncpg = pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")

    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    database = f"upd_payload_{uuid4().hex[:8]}"
    admin = await asyncpg.connect(
        host="localhost", port=5432, user="cognee", password="cognee", database="postgres"
    )
    await admin.execute(f"CREATE DATABASE {database}")
    await admin.close()
    setup = await asyncpg.connect(
        host="localhost", port=5432, user="cognee", password="cognee", database=database
    )
    await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await setup.close()

    embedder = CountingMockEmbedder()
    engine = PGVectorAdapter(
        connection_string=f"postgresql+asyncpg://cognee:cognee@localhost:5432/{database}",
        api_key=None,
        embedding_engine=embedder,
    )
    try:
        await _exercise_update_payload(engine, embedder)
    finally:
        if hasattr(engine, "engine"):
            await engine.engine.dispose()
        admin = await asyncpg.connect(
            host="localhost", port=5432, user="cognee", password="cognee", database="postgres"
        )
        await admin.execute(f"DROP DATABASE {database} (FORCE)")
        await admin.close()


@pytest.mark.asyncio
async def test_turso_update_payload():
    pytest.importorskip("libsql", reason="libsql driver not installed")
    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
        TursoVectorAdapter,
    )

    embedder = CountingMockEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        engine = TursoVectorAdapter(
            url=str(Path(tmp) / "turso.db"), api_key=None, embedding_engine=embedder
        )
        await _exercise_update_payload(engine, embedder)
