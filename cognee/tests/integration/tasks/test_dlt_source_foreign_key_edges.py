"""Foreign keys of a SQL source become edges between the rows they connect.

dlt stages a database source in tables without key constraints, so the keys come
from dlt's schema hints. Covers the shapes that hints merge or rename: two foreign
keys to one table, a self reference, a primary key that is neither ``id`` nor the
first column, a composite key, and keys that only the rows' own keys can resolve. Rows reach cognify in table order, two per batch, so one edge
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
CREATE TABLE order_lines (
    order_id INTEGER REFERENCES orders (id),
    n INTEGER,
    PRIMARY KEY (order_id, n)
);
INSERT INTO users VALUES ('Ann', 'u1'), ('Bob', 'u2');
INSERT INTO orders VALUES (1, 'u1', 'u2', NULL), (2, 'u2', 'u1', 1);
INSERT INTO visits VALUES (1, 'u2');
INSERT INTO order_lines VALUES (1, 1), (1, 2), (2, 1);
"""

# Each table's primary key as its SchemaTable node records it.
EXPECTED_TABLE_KEYS = {
    "users": "user_code",
    "orders": "id",
    "visits": "id",
    "order_lines": "order_id, n",
}


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
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    # Cached per staging database name: a dataset name reused across tests would
    # otherwise stage into the earlier test's files.
    get_dlt_destination.cache_clear()
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
    finally:
        get_dlt_destination.cache_clear()


EXPECTED_ROW_EDGES = {
    ("orders:1", "buyer_code_references_users", "users:u1"),
    ("orders:1", "seller_code_references_users", "users:u2"),
    ("orders:2", "buyer_code_references_users", "users:u2"),
    ("orders:2", "seller_code_references_users", "users:u1"),
    ("orders:2", "parent_id_references_orders", "orders:1"),
    ("visits:1", "user_code_references_users", "users:u2"),
    # A composite key names the row by all its columns.
    ("order_lines:1:1", "order_id_references_orders", "orders:1"),
    ("order_lines:1:2", "order_id_references_orders", "orders:1"),
    ("order_lines:2:1", "order_id_references_orders", "orders:2"),
}


SOURCE_TABLES = frozenset(EXPECTED_TABLE_KEYS)


def _write_source_db(path: pathlib.Path, sql: str = SOURCE_SQL) -> pathlib.Path:
    connection = sqlite3.connect(path)
    connection.executescript(sql)
    connection.close()
    return path


async def _row_edges(tables: frozenset[str] = SOURCE_TABLES) -> set[tuple[str, str, str]]:
    """Edges between the SQL source's rows, by the rows' names (``table:pk``).
    Rows of other sources in the dataset (a CSV's) are left out."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, edges = await (await get_graph_engine()).get_graph_data()
    rows = {
        str(node_id): props["name"]
        for node_id, props in nodes
        if props.get("type") == "DltRow" and props["name"].split(":")[0] in tables
    }
    return {
        (rows[str(source)], relation, rows[str(target)])
        for source, target, relation, _ in edges
        if str(source) in rows and str(target) in rows
    }


async def _table_keys() -> dict[str, str]:
    """Each SQL source table's primary key, as its SchemaTable node records it."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, _ = await (await get_graph_engine()).get_graph_data()
    return {
        props["name"]: props["primary_key"]
        for _, props in nodes
        if props.get("type") == "SchemaTable" and props["name"] in SOURCE_TABLES
    }


@pytest.mark.asyncio
async def test_source_foreign_keys_become_edges_between_rows(clean_env):
    source_db = _write_source_db(clean_env / "source.sqlite")

    await cognee.add(f"sqlite:///{source_db}", dataset_name="shop")
    await cognee.cognify(datasets=["shop"], chunks_per_batch=2)

    assert await _row_edges() == EXPECTED_ROW_EDGES
    assert await _table_keys() == EXPECTED_TABLE_KEYS


@pytest.mark.asyncio
async def test_sql_source_keys_survive_an_earlier_csv_in_the_same_pipeline(clean_env):
    """Unscoped sources share one dlt pipeline, and dlt keeps the first schema it
    saw as the default. A SQL source ingested after a CSV lands in its own schema,
    so its keys must be read from that schema, not from the pipeline default."""
    csv_path = clean_env / "notes.csv"
    csv_path.write_text("id,kind,unit\n1,thermometer,C\n2,hygrometer,%\n3,anemometer,m/s\n")
    source_db = _write_source_db(clean_env / "source.sqlite")

    await cognee.add(str(csv_path), dataset_name="shop")
    await cognee.add(f"sqlite:///{source_db}", dataset_name="shop")
    await cognee.cognify(datasets=["shop"])

    assert await _row_edges() == EXPECTED_ROW_EDGES


KEY_SHAPES_SQL = """
CREATE TABLE order_lines (
    order_id INTEGER, line_no INTEGER, product TEXT, PRIMARY KEY (order_id, line_no)
);
CREATE TABLE shipments (
    id INTEGER PRIMARY KEY,
    ship_line INTEGER,
    ship_order INTEGER,
    FOREIGN KEY (ship_line, ship_order) REFERENCES order_lines (line_no, order_id)
);
CREATE TABLE people (id INTEGER PRIMARY KEY, employee_no INTEGER UNIQUE, name TEXT);
CREATE TABLE badges (
    id INTEGER PRIMARY KEY,
    employee_no INTEGER REFERENCES people (employee_no),
    owner_id INTEGER REFERENCES people (id)
);
CREATE TABLE teams (id INTEGER UNIQUE, name TEXT);
CREATE TABLE members (
    id INTEGER PRIMARY KEY,
    team_id INTEGER REFERENCES teams (id)
);
INSERT INTO order_lines VALUES (1, 1, 'pen'), (1, 2, 'ink'), (2, 1, 'paper');
INSERT INTO shipments VALUES (10, 2, 1);
INSERT INTO people VALUES (1, 2, 'Ann'), (2, 7, 'Bob');
INSERT INTO badges VALUES (100, 2, 1);
INSERT INTO teams VALUES (1, 'red'), (2, 'blue');
INSERT INTO members VALUES (5, 2);
"""
KEY_SHAPES_TABLES = frozenset({"order_lines", "shipments", "people", "badges", "teams", "members"})


async def _table_foreign_keys() -> dict[str, list[dict]]:
    import json

    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, _ = await (await get_graph_engine()).get_graph_data()
    return {
        props["name"]: json.loads(props["foreign_keys"])
        for _, props in nodes
        if props.get("type") == "SchemaTable" and props["name"] in KEY_SHAPES_TABLES
    }


@pytest.mark.asyncio
async def test_each_foreign_key_finds_the_row_it_references(clean_env):
    """A composite foreign key finds its row by all its columns, whatever their
    declared order. A table without a declared primary key is keyed on ``id``, and a
    foreign key to it finds its row. A key to a unique non-primary column finds the
    row holding that value (badge 100 to Ann, whose employee_no is 2), not the row
    whose id equals it (Bob)."""
    source_db = _write_source_db(clean_env / "keys.sqlite", KEY_SHAPES_SQL)

    await cognee.add(f"sqlite:///{source_db}", dataset_name="shop")
    await cognee.cognify(datasets=["shop"])

    assert await _row_edges(KEY_SHAPES_TABLES) == {
        ("shipments:10", "ship_order_ship_line_references_order_lines", "order_lines:1:2"),
        ("badges:100", "owner_id_references_people", "people:1"),
        ("badges:100", "employee_no_references_people", "people:1"),
        ("members:5", "team_id_references_teams", "teams:2"),
    }
    table_foreign_keys = await _table_foreign_keys()
    assert {fk["column"] for fk in table_foreign_keys["shipments"]} == {"ship_line", "ship_order"}
    assert {fk["column"] for fk in table_foreign_keys["badges"]} == {"employee_no", "owner_id"}


@pytest.mark.asyncio
async def test_each_table_names_and_links_the_database_it_came_from(clean_env):
    from cognee.infrastructure.databases.graph import get_graph_engine

    source_db = _write_source_db(clean_env / "shop.sqlite")
    await cognee.add(f"sqlite:///{source_db}", dataset_name="shop")
    await cognee.cognify(datasets=["shop"])

    nodes, edges = await (await get_graph_engine()).get_graph_data()
    types = {str(node_id): props.get("type") for node_id, props in nodes}
    tables = {str(i): p for i, p in nodes if p.get("type") == "SchemaTable"}
    assert {p["name"] for p in tables.values()} == SOURCE_TABLES
    for props in tables.values():
        assert f"'{props['name']}' from shop.sqlite with" in props["description"]
    table_of = {str(s): types[str(t)] for s, t, name, *_ in edges if name == "is_table_of"}
    assert table_of == dict.fromkeys(tables, "DltSourceDocument")


LINKED_SQL = """
CREATE TABLE users (id INTEGER PRIMARY KEY);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users (id));
INSERT INTO users VALUES (1);
INSERT INTO orders VALUES (1, 1);
"""
UNLINKED_SQL = """
CREATE TABLE users (id INTEGER PRIMARY KEY);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER);
INSERT INTO users VALUES (1);
INSERT INTO orders VALUES (1, 1);
"""


@pytest.mark.asyncio
async def test_a_source_does_not_inherit_the_keys_of_an_earlier_source(clean_env):
    """Unscoped sources share one dlt pipeline whose schema never forgets a
    reference. A database without foreign keys, staged after one with them under the
    same table names, must not pick theirs up."""
    from cognee.tasks.ingestion.create_dlt_source import create_dlt_source_from_connection_string
    from cognee.tasks.ingestion.ingest_dlt_source import ingest_dlt_source

    linked = _write_source_db(clean_env / "linked.sqlite", LINKED_SQL)
    unlinked = _write_source_db(clean_env / "unlinked.sqlite", UNLINKED_SQL)

    linked_rows = await ingest_dlt_source(
        create_dlt_source_from_connection_string(f"sqlite:///{linked}"), "shop"
    )
    unlinked_rows = await ingest_dlt_source(
        create_dlt_source_from_connection_string(f"sqlite:///{unlinked}"), "shop"
    )

    def orders_keys(rows):
        return [row.foreign_keys for row in rows if row.table_name == "orders"]

    assert orders_keys(linked_rows) == [
        [{"column": "user_id", "ref_table": "users", "ref_column": "id"}]
    ]
    assert orders_keys(unlinked_rows) == [[]]
