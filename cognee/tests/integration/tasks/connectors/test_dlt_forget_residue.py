"""SDK-1549: forgetting one dlt table must leave none of its text in the staging database.

``ingestion.empty_resource`` empties the table and clears its resource state, but the merge
staging copy (``<dataset>_staging.<table>``) and the older ``_dlt_pipeline_state`` rows stay.
Every test runs on SQLite (one staging file per dataset) and on a real Postgres (schemas
``<dataset>`` and ``<dataset>_staging``; skipped when none answers on localhost:5432 with
cognee/cognee). ``_holders`` reads every table of the staging database, decoding the
base64+zlib state, and reports where a token still lives.
"""

import base64
import sqlite3
import uuid
import zlib
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

import cognee
from cognee.tasks.ingestion.connectors import onenote_source
from cognee.tests.unit.tasks.connectors.fake_onenote import Graph

DATASET = "forget_residue_test"
SECRET = "Zebracorpsecret"
STAGING_DB = f"dlt_database_{DATASET}"
PG = {"host": "localhost", "port": 5432, "user": "cognee", "password": "cognee"}


async def _pg_admin(sql: str) -> None:
    conn = await asyncpg.connect(database="postgres", **PG)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def backend(request, tmp_path, monkeypatch):
    """``(kind, system_root, postgres database name)``, with the relational DB configured."""
    pytest.importorskip("dlt")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    database = None
    if request.param == "postgres":
        database = f"sdk1549_{uuid.uuid4().hex[:8]}"
        try:
            await _pg_admin(f'create database "{database}"')
        except (OSError, asyncpg.PostgresError) as error:
            pytest.skip(f"no Postgres on localhost:5432 ({type(error).__name__})")
        cognee.config.set_relational_db_config(
            {
                "db_provider": "postgres",
                "db_host": PG["host"],
                "db_port": str(PG["port"]),
                "db_username": PG["user"],
                "db_password": PG["password"],
                "db_name": database,
            }
        )
    else:
        cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield request.param, tmp_path / "system", database
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    if database:
        await _pg_admin(f'drop database if exists "{database}" with (force)')
        await _pg_admin(f'drop database if exists "{STAGING_DB}" with (force)')


def _text(cell) -> str:
    """A cell as text, with dlt's base64+zlib pipeline state decoded next to the raw value."""
    text = str(cell)
    try:
        return zlib.decompress(base64.b64decode(text, validate=True)).decode() + text
    except (ValueError, zlib.error):
        return text


async def _holders(backend, token: str, only: str = "") -> list[str]:
    """``<where> <table> (<rows holding token>)`` for every staging-database table."""
    kind, system_root, _ = backend
    found = []
    if kind == "sqlite":
        for path in system_root.rglob("*dlt_database*"):
            if not path.is_file():
                continue
            with sqlite3.connect(path) as db:
                for (table,) in db.execute("select name from sqlite_master where type='table'"):
                    rows = db.execute(f'select * from "{table}"').fetchall()
                    hits = sum(any(token in _text(cell) for cell in row) for row in rows)
                    if hits:
                        found.append(f"{path.name} {table} ({hits})")
    else:
        conn = await asyncpg.connect(database=STAGING_DB, **PG)
        try:
            tables = await conn.fetch(
                "select table_schema, table_name from information_schema.tables "
                "where table_schema like $1 and table_type = 'BASE TABLE'",
                f"{DATASET}%",
            )
            for schema, table in tables:
                rows = await conn.fetch(f'select * from "{schema}"."{table}"')
                hits = sum(any(token in _text(cell) for cell in row.values()) for row in rows)
                if hits:
                    found.append(f"{schema} {table} ({hits})")
        finally:
            await conn.close()
    return [h for h in found if only in h]


async def _state_rows(backend, table: str) -> int:
    """``_dlt_pipeline_state`` rows of the pipeline a table's sync runs under."""
    from types import SimpleNamespace

    from cognee.tasks.ingestion.dlt_utils import PIPELINE_SCOPE_ATTR, pipeline_name_for_source

    pipeline = pipeline_name_for_source(SimpleNamespace(**{PIPELINE_SCOPE_ATTR: table}), DATASET)
    kind, system_root, _ = backend
    if kind == "sqlite":
        total = 0
        for path in system_root.rglob("*dlt_database*"):
            if path.is_file():
                with sqlite3.connect(path) as db:
                    if db.execute(
                        "select 1 from sqlite_master where name = '_dlt_pipeline_state'"
                    ).fetchone():
                        total += db.execute(
                            "select count(*) from _dlt_pipeline_state where pipeline_name = ?",
                            (pipeline,),
                        ).fetchone()[0]
        return total
    conn = await asyncpg.connect(database=STAGING_DB, **PG)
    try:
        return await conn.fetchval(
            f'select count(*) from "{DATASET}"._dlt_pipeline_state where pipeline_name = $1',
            pipeline,
        )
    finally:
        await conn.close()


async def _merge_load(table: str, text: str):
    import dlt

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    @dlt.resource(name=table, primary_key="id", write_disposition="merge")
    def documents():
        yield {"id": "1", "title": "t", "content": text, "url": "https://example.com/1"}

    resource = documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "onenote")
    setattr(resource, PIPELINE_SCOPE_ATTR, table)
    await cognee.add(
        resource,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _forget(name: str):
    from cognee.modules.integrations.ingestion import empty_resource

    await cognee.add(
        empty_resource("onenote", name),
        dataset_name=DATASET,
        write_disposition="replace",
        primary_key="id",
        max_rows_per_table=0,
    )


def _notebook(pages: int = 6) -> Graph:
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{i}") for i in range(pages)]
    graph.html.update({f"p{i}": f"<p>{SECRET} page {i}</p>" for i in range(pages)})
    return graph


async def _onenote_sync(graph: Graph, resource_name: str, **limits):
    with graph.client() as client:
        source = onenote_source(
            "offline-token",
            notebook_id="a",
            account_id="tenant:user",
            resource_name=resource_name,
            http_client=client,
            **limits,
        )
        await cognee.add(
            source,
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    return source


def _bodies_fetched(graph: Graph) -> int:
    return sum(graph.content_requests().values())


@pytest.mark.asyncio
async def test_empty_resource_leaves_no_text_in_the_merge_staging_copy(backend):
    await _merge_load("onenote_x", f"page about {SECRET}")
    assert await _holders(backend, SECRET, "onenote_x"), "the load must stage the text"

    await _forget("onenote_x")

    assert await _holders(backend, SECRET) == []


@pytest.mark.asyncio
async def test_empty_under_another_spelling_of_the_name_leaves_no_text(backend):
    """Cloud's scoped delete empties under ``<dataset>.<resource>``, a pipeline of its own."""
    await _merge_load("onenote_x", f"page about {SECRET}")

    await _forget(f"{DATASET}.onenote_x")

    assert await _holders(backend, SECRET) == []


@pytest.mark.asyncio
async def test_forgetting_one_table_keeps_the_other_tables(backend):
    await _merge_load("onenote_x", f"page about {SECRET}")
    await _merge_load("onenote_other", "unrelated text Keepme")
    state_before = await _state_rows(backend, "onenote_other")
    assert state_before

    await _forget("onenote_x")

    assert await _holders(backend, SECRET) == []
    kept = await _holders(backend, "Keepme")
    assert any("onenote_other" in h and "staging" in h for h in kept), "its merge copy stays"
    assert any("onenote_other" in h and "staging" not in h for h in kept), "its rows stay"
    assert await _state_rows(backend, "onenote_other") == state_before


@pytest.mark.asyncio
async def test_a_reselected_table_fetches_every_record_again(backend):
    graph = _notebook()
    await _onenote_sync(graph, "onenote_x")
    assert _bodies_fetched(graph) == 6
    await _forget("onenote_x")

    before = _bodies_fetched(graph)
    await _onenote_sync(graph, "onenote_x")

    assert _bodies_fetched(graph) - before == 6


@pytest.mark.asyncio
async def test_a_cut_onenote_run_keeps_no_page_text_in_state(backend):
    graph = _notebook()

    source = await _onenote_sync(graph, "onenote_cut", max_requests=6)

    assert source.cognee_sync_stats.get("complete") == 0, "the run must be cut"
    assert await _holders(backend, SECRET, "pipeline_state") == []


@pytest.mark.asyncio
async def test_a_cut_onenote_run_leaves_no_page_text_after_forget(backend):
    graph = _notebook()
    await _onenote_sync(graph, "onenote_cut", max_requests=6)

    await _forget("onenote_cut")

    assert await _holders(backend, SECRET) == []


@pytest.mark.asyncio
async def test_a_cut_run_then_a_complete_run_publishes_every_page_and_the_next_fetches_none(
    backend,
):
    graph = _notebook()
    cut = await _onenote_sync(graph, "onenote_cut", max_requests=6)
    assert cut.cognee_sync_stats.get("complete") == 0

    complete = await _onenote_sync(graph, "onenote_cut")
    assert complete.cognee_sync_stats.get("complete") == 1
    published = await _holders(backend, SECRET, "onenote_cut")
    assert sum(int(h.rsplit("(", 1)[1].rstrip(")")) for h in published if "staging" not in h) == 6

    before = _bodies_fetched(graph)
    await _onenote_sync(graph, "onenote_cut")

    assert _bodies_fetched(graph) == before
