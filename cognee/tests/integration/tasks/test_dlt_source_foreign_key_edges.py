"""Foreign keys of a SQL source become edges between the rows they connect.

dlt stages a database source in tables without key constraints, so the keys come
from dlt's schema hints. Covers the shapes that hints merge or rename: two foreign
keys to one table, a self reference, and a primary key that is neither ``id`` nor
the first column. Rows reach cognify in table order, two per batch, so one edge
has both rows in one batch, and the others have the target row written after or
before its source.

Real add + cognify against local stores; the DLT route is LLM-free, and embeddings
are mocked so the vector store works offline.
"""

import logging
import pathlib
import sqlite3
from unittest.mock import patch

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.modules.engine.operations.setup import setup as engine_setup

logger = logging.getLogger(__name__)

SOURCE_SQL = """
CREATE TABLE users (name TEXT, user_code TEXT PRIMARY KEY);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    buyer_code TEXT REFERENCES users (user_code),
    seller_code TEXT REFERENCES users (user_code),
    parent_id INTEGER REFERENCES orders (id)
);
CREATE TABLE visits (id INTEGER PRIMARY KEY, user_code TEXT REFERENCES users (user_code));
INSERT INTO users VALUES ('Ann', 'u1'), ('Bob', 'u2');
INSERT INTO orders VALUES (1, 'u1', 'u2', NULL), (2, 'u2', 'u1', 1);
INSERT INTO visits VALUES (1, 'u2');
"""


async def _mock_embed_text(self, texts):
    vectors = [[float(len(t) % 7) / 10.0] * self.dimensions for t in texts]
    return EmbeddingResult(vectors=vectors, tokens=0)


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text):
        yield root

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring exception in clean_env", exc_info=True)


def _row_label(text: str) -> str:
    """'users:u1' or 'orders:1' from a row's text, keyed on its primary key."""
    fields = dict(line.strip().split(": ", 1) for line in text.splitlines() if ": " in line)
    table = fields["Table"]
    return f"{table}:{fields['user_code' if table == 'users' else 'id']}"


@pytest.mark.asyncio
async def test_source_foreign_keys_become_edges_between_rows(clean_env):
    source_db = clean_env / "source.sqlite"
    connection = sqlite3.connect(source_db)
    connection.executescript(SOURCE_SQL)
    connection.close()

    await cognee.add(f"sqlite:///{source_db}", dataset_name="shop")
    await cognee.cognify(datasets=["shop"], chunks_per_batch=2)

    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, edges = await (await get_graph_engine()).get_graph_data()
    rows = {
        str(node_id): _row_label(props["text"])
        for node_id, props in nodes
        if props.get("type") == "DltRow"
    }
    row_edges = {
        (rows[str(source)], relation, rows[str(target)])
        for source, target, relation, _ in edges
        if str(source) in rows and str(target) in rows
    }

    assert row_edges == {
        ("orders:1", "buyer_code_references_users", "users:u1"),
        ("orders:1", "seller_code_references_users", "users:u2"),
        ("orders:2", "buyer_code_references_users", "users:u2"),
        ("orders:2", "seller_code_references_users", "users:u1"),
        ("orders:2", "parent_id_references_orders", "orders:1"),
        ("visits:1", "user_code_references_users", "users:u2"),
    }
