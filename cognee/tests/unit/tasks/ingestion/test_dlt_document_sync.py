"""A document sync resolves only the staged rows the dataset may not hold (SDK-1715).

Every sync used to read the whole staging copy back, build a DataItem per staged row and
sweep every stored document: a sync with nothing new cost 31 s at 10,000 rows. dlt marks
each staged row with the load that wrote it, so a sync now reads the rows of loads cognee
has not committed, plus any staged row the dataset lacks. These run the real dlt staging
and a real SQLite store; ``_commit`` stands in for the add pipeline and the cleanup add
runs once that pipeline completed is awaited after it.
"""

import importlib
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.methods.get_existing_data_ids import DATA_ID_BATCH_SIZE
from cognee.modules.data.models import Data
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")
resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")
unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")
by_ids_module = importlib.import_module("cognee.modules.data.methods.get_dataset_data_by_ids")

USER = SimpleNamespace(id=UUID(int=1), tenant_id=None)
DATASET = UUID(int=2)
OTHER_DATASET = UUID(int=3)
TOMBSTONE_COLUMN = {"_deleted": {"data_type": "bool", "hard_delete": True}}


@pytest_asyncio.fixture
async def sync(tmp_path, monkeypatch):
    """Runs document syncs into the dataset ``brain``; returns what each one resolved."""
    dlt = pytest.importorskip("dlt")
    config = SimpleNamespace(
        db_provider="sqlite",
        db_path=str(tmp_path),
        db_host="",
        db_port=0,
        db_username="",
        db_password="",
    )
    monkeypatch.setattr(ingest, "get_relational_config", lambda: config)
    monkeypatch.setattr(
        ingest,
        "get_dlt_destination",
        lambda dlt_db_name: dlt.destinations.sqlalchemy(
            credentials={"database": str(tmp_path / dlt_db_name), "drivername": "sqlite"},
        ),
    )
    pipeline_factory = dlt.pipeline
    monkeypatch.setattr(
        dlt, "pipeline", lambda **kw: pipeline_factory(pipelines_dir=str(tmp_path / "p"), **kw)
    )

    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'cognee.db'}")
    await engine.create_database()
    for module in (unique_id_module, by_ids_module):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine", lambda: engine
    )

    forgotten: list[set[UUID]] = []
    sweeps: list[dict] = []

    async def record_sweep(
        _dataset_name, _user, fresh_data_ids, *, candidate_ids=frozenset(), **kwargs
    ):
        """The orphan sweep as far as a sync is concerned: forget its candidates, keep what it
        was asked (the sweep itself runs in test_dlt_document_cleanup_scope.py)."""
        sweeps.append({"fresh": set(fresh_data_ids), "candidates": set(candidate_ids), **kwargs})
        forgotten.append(set(candidate_ids))
        await _forget(engine, candidate_ids)

    monkeypatch.setattr(resolve, "_delete_dlt_orphans", record_sweep)

    async def sync_source(source, commit, dataset_id, key="id"):
        return await sync_sources([(source, "notes")], commit, dataset_id, key)

    async def sync_sources(sources, commit, dataset_id, key="id"):
        """One add of several (source, tag) pairs."""
        for source, tag in sources:
            setattr(source, DOCUMENT_SOURCE_ATTR, tag)
        forgotten.clear()
        sweeps.clear()
        items, cleanup = await resolve.resolve_dlt_sources(
            [source for source, _ in sources],
            dataset_name="brain",
            user=USER,
            dataset_id=dataset_id,
            primary_key=key,
        )
        if commit:
            await _commit(engine, items, dataset_id)
            await cleanup()
        return SimpleNamespace(
            items=items,
            cleanup=cleanup,
            ids=sorted(
                item.system_metadata["external_id"]
                for item in items
                if "external_id" in item.system_metadata
            ),
            forgotten=set().union(*forgotten) if forgotten else set(),
            sweeps=list(sweeps),
        )

    def resource(rows, name="docs", key="id"):
        @dlt.resource(
            name=name, write_disposition="merge", primary_key=key, columns=TOMBSTONE_COLUMN
        )
        def docs():
            yield from rows

        return docs()

    async def run(rows, *, commit=True, dataset_id=DATASET, name="docs", key="id"):
        return await sync_source(resource(rows, name, key), commit, dataset_id, key)

    async def earlier_release(rows, key):
        """What a release before committed loads stored: every staged row, read on ``key``."""
        source = resource(rows, key=key)
        setattr(source, DOCUMENT_SOURCE_ATTR, "notes")
        staged = await ingest.ingest_dlt_source(
            source, "brain", primary_key=key, write_disposition=None
        )
        row_ids = await resolve._stable_row_ids(staged, USER, DATASET)
        items = [
            resolve._build_document_data_item(row, data_id, "notes")
            for row, data_id in zip(staged, row_ids)
        ]
        await _commit(engine, items)

    async def replaced(rows):
        @dlt.resource(name="pages", write_disposition="replace")
        def pages():
            yield from rows

        return await sync_source(pages(), True, DATASET)

    run.replaced = replaced
    run.earlier_release = earlier_release
    run.commit = lambda items: _commit(engine, items)
    run.source = lambda source, commit=True: sync_source(source, commit, DATASET)
    run.resource = resource
    run.sources = lambda sources: sync_sources(sources, True, DATASET)
    run.engine = engine
    run.stored = lambda dataset_id=DATASET: _stored(engine, dataset_id)
    run.names = lambda: _names(engine)
    run.forget = lambda key: _forget_key(engine, key)
    return run


def _row(key, version="v1"):
    return {"id": key, "title": f"t-{key}", "content": f"{key} {version}", "_deleted": False}


async def _commit(engine, items, dataset_id=DATASET):
    """What the add pipeline stores for each item: a Data row of the dataset."""
    async with engine.get_async_session() as session:
        known = set((await session.scalars(select(Data.id))).all())
        values = [
            {
                "id": item.data_id,
                "dataset_id": dataset_id,
                "name": item.label,
                "content_hash": str(item.data_id),
                "raw_data_location": "text://",
                "owner_id": USER.id,
                "system_metadata": item.system_metadata,
                "pipeline_status": {},
                "token_count": -1,
            }
            for item in items
            if item.data_id not in known
        ]
        if values:
            await session.execute(insert(Data.__table__), values)
        await session.commit()


async def _forget(engine, ids):
    async with engine.get_async_session() as session:
        await session.execute(delete(Data).where(Data.id.in_(list(ids))))
        await session.commit()


async def _forget_key(engine, key):
    stored = await _stored(engine, DATASET)
    await _forget(engine, [stored[key]])


async def _names(engine):
    async with engine.get_async_session() as session:
        return sorted((await session.scalars(select(Data.name))).all())


async def _stored(engine, dataset_id):
    async with engine.get_async_session() as session:
        statement = select(Data.id, Data.system_metadata).where(Data.dataset_id == dataset_id)
        rows = (await session.execute(statement)).all()
    return {meta.get("external_id"): data_id for data_id, meta in rows}


@pytest.mark.asyncio
async def test_a_sync_resolves_only_what_the_dataset_may_not_hold(sync):
    first = await sync([_row(key) for key in "abcde"])
    assert first.ids == ["a", "b", "c", "d", "e"]
    before = await sync.stored()

    # a re-sent unchanged, b edited, c deleted, f new; d and e not sent.
    changed = await sync([_row("a"), _row("b", "v2"), {"id": "c", "_deleted": True}, _row("f")])

    assert changed.ids == ["a", "b", "f"]
    # The edited row's previous version and the deleted row are forgotten; nothing else.
    assert changed.forgotten == {before["b"], before["c"]}
    after = await sync.stored()
    assert sorted(after) == ["a", "b", "d", "e", "f"]
    assert after["a"] == before["a"] and after["d"] == before["d"]


@pytest.mark.asyncio
async def test_a_sync_with_nothing_new_resolves_nothing(sync):
    await sync([_row(key) for key in "abc"])

    unchanged = await sync([])

    assert unchanged.items == []
    assert unchanged.forgotten == set()


@pytest.mark.asyncio
async def test_a_document_forgotten_here_comes_back_with_the_next_sync(sync):
    await sync([_row(key) for key in "abc"])
    await sync.forget("b")

    restored = await sync([])

    assert restored.ids == ["b"]


@pytest.mark.asyncio
async def test_rows_of_a_sync_whose_add_failed_are_read_again(sync):
    await sync([_row("a")])
    await sync([_row("b")], commit=False)  # staged by dlt, never committed by cognee

    retried = await sync([])

    assert retried.ids == ["b"]
    assert sorted(await sync.stored()) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_load_is_read_whatever_its_order_against_committed_loads(sync, monkeypatch):
    """Another worker's load can land after a sync recorded loads with later ids."""
    await sync([_row("a")])
    monkeypatch.setattr(
        resolve,
        "committed_document_loads",
        AsyncMock(return_value={("docs", "9999999999.000000")}),
    )

    edited = await sync([_row("a", "v2")])

    assert edited.ids == ["a"]


@pytest.mark.asyncio
async def test_a_load_is_not_committed_while_its_document_is_missing(sync):
    """Two syncs resolve before either add commits, and the source reverts an edit between them.

    Ids derive from content, so the reverted row has its stored id back and its add skips
    it, then the first sync's sweep forgets that stored version. Recording the second
    sync's load would leave the edit in the dataset with no sync ever reading the row again.
    """
    await sync([_row("a")])
    original = (await sync.stored())["a"]
    edited = await sync([_row("a", "v2")], commit=False)
    reverted = await sync([_row("a")], commit=False)
    await sync.commit(edited.items)
    await sync.commit(reverted.items)
    await edited.cleanup()
    await reverted.cleanup()

    healed = await sync([])

    assert healed.ids == ["a"]
    assert await sync.stored() == {"a": original}


@pytest.mark.asyncio
async def test_a_sync_that_read_nothing_does_not_vouch_for_a_changed_key(sync):
    """A third sync resolves after the second one committed, so it reads nothing for the key
    and its presence check passes on an empty set; recording its loads would put back the
    pair the first sync replaced, and no later sync would read the key again."""
    await sync([_row("a")])
    original = (await sync.stored())["a"]
    first = await sync([_row("a", "v2")], commit=False)
    second = await sync([_row("a")], commit=False)
    await sync.commit(second.items)
    await second.cleanup()
    third = await sync([], commit=False)
    await sync.commit(first.items)
    await first.cleanup()
    await sync.commit(third.items)
    await third.cleanup()

    await sync([])

    assert await sync.stored() == {"a": original}


@pytest.mark.asyncio
async def test_the_sweep_is_asked_for_exactly_the_stale_documents(sync):
    """What the orphan sweep is given decides what it may forget: its candidates, the ids to
    keep, and the sources and tables it may touch."""
    await sync([_row("a"), _row("b")])
    before = await sync.stored()

    edited = await sync([_row("a", "v2")])

    (sweep,) = edited.sweeps
    assert sweep["candidates"] == {before["a"]}
    assert sweep["fresh"] == {item.data_id for item in edited.items}
    assert sweep["sources"] == ("notes",)
    assert sweep["document_scopes"] == {("notes", "docs")}


@pytest.mark.asyncio
async def test_settling_checks_and_records_under_the_dataset_lock(sync, monkeypatch):
    """The check that a sync's documents are all there and the record that follows must not
    interleave with another sync's sweep."""
    await sync([_row("a")])
    events: list[str] = []
    sweep, record = resolve._delete_dlt_orphans, resolve.record_committed_document_loads
    exist = resolve.get_existing_data_ids

    @asynccontextmanager
    async def lock(_dataset_id):
        events.append("lock")
        yield
        events.append("unlock")

    async def logged(name, function, *args, **kwargs):
        events.append(name)
        return await function(*args, **kwargs)

    monkeypatch.setattr("cognee.infrastructure.locks.dataset_lock", lock)
    monkeypatch.setattr(resolve, "_delete_dlt_orphans", partial(logged, "sweep", sweep))
    monkeypatch.setattr(
        resolve, "record_committed_document_loads", partial(logged, "record", record)
    )
    monkeypatch.setattr(resolve, "get_existing_data_ids", partial(logged, "check", exist))

    await sync([_row("a", "v2")])

    assert events[events.index("lock") :] == ["lock", "sweep", "check", "record", "unlock"]


@pytest.mark.asyncio
async def test_two_sources_in_one_add_are_swept_and_recorded_together(sync):
    """A stale document of the second source is forgotten, the first source's fresh ids are
    kept for the whole sweep, and both sources' loads are recorded."""

    def both(version):
        """The two sources emitting their row at ``version``; None emits nothing."""
        rows = lambda key: [] if version is None else [_row(key, version)]
        return [
            (sync.resource(rows("a"), name="docs"), "notes"),
            (sync.resource(rows("w"), name="wikis"), "wiki"),
        ]

    await sync.sources(both("v1"))
    before = await sync.stored()

    edited = await sync.sources(both("v2"))

    (sweep,) = edited.sweeps
    assert sweep["candidates"] == {before["a"], before["w"]}
    assert sweep["fresh"] == {item.data_id for item in edited.items}
    assert sweep["document_scopes"] == {("notes", "docs"), ("wiki", "wikis")}
    assert set(sweep["sources"]) == {"notes", "wiki"}
    assert sorted(await sync.names()) == ["t-a", "t-w"]
    assert (await sync.sources(both(None))).items == []


@pytest.mark.asyncio
async def test_a_mismatch_clears_the_pairs_another_sync_recorded(sync):
    """The first sync's own view no longer holds (an unrelated document was forgotten after
    it resolved), while the second sync already recorded a load the first one replaced. Only
    clearing the pairs makes the next sync read the key again; recording nothing would leave
    the second sync's pair vouching for a version the first one swept."""
    await sync([_row("a"), _row("b")])
    before = await sync.stored()
    first = await sync([_row("a", "v2")], commit=False)
    second = await sync([_row("a")], commit=False)
    await sync.commit(second.items)
    await second.cleanup()
    await sync.forget("b")
    await sync.commit(first.items)
    await first.cleanup()

    await sync([])

    assert await sync.stored() == before


@pytest.mark.asyncio
async def test_a_stale_document_that_survives_the_sweep_costs_one_key_not_the_table(
    sync, monkeypatch
):
    """Without delete permission the sweep forgets nothing. Its loads are still recorded, so
    the next syncs read the key that has two documents, not every row of the table."""
    await sync([_row(key) for key in "abc"])
    await sync([_row("a", "v2")])

    async def no_permission(*_args, **_kwargs):
        return None

    monkeypatch.setattr(resolve, "_delete_dlt_orphans", no_permission)
    await sync([_row("a", "v3")])

    again = await sync([])

    assert again.ids == ["a"]


@pytest.mark.asyncio
async def test_new_rows_of_a_table_are_read_by_their_load(sync, monkeypatch):
    """dlt indexes neither the id nor the load column, so a read by id costs a scan per batch
    of ids; the rows of a load that cognee has not committed are read by that load."""
    await sync([_row("a"), _row("b")])
    reads: list[tuple] = []
    read = ingest._read_single_table

    async def spy(*args, column_in=None, **kwargs):
        reads.append(column_in)
        return await read(*args, column_in=column_in, **kwargs)

    monkeypatch.setattr(ingest, "_read_single_table", spy)

    added = await sync([_row("c")])

    assert added.ids == ["c"]
    assert [column for column, _ in reads] == ["_dlt_load_id"]


@pytest.mark.asyncio
async def test_documents_keep_the_ids_their_primary_key_gave_them(sync):
    """Ids derive from the caller's primary key, as they did before committed loads, so the
    first sync after an upgrade forgets and re-ingests nothing."""
    await sync.earlier_release([{**_row(key), "key": f"k-{key}"} for key in "ab"], key="key")
    before = await sync.stored()

    upgraded = await sync([], key="key")

    assert upgraded.forgotten == set()
    assert await sync.stored() == before
    assert (await sync([], key="key")).ids == []
    edited = await sync([{**_row("a", "v2"), "key": "k-a"}], key="key")
    assert edited.forgotten == {before["a"]}


@pytest.mark.asyncio
async def test_datasets_sharing_a_name_keep_their_own_committed_loads(sync):
    """Staging is keyed by dataset name; one dataset's commit must not cover another's."""
    await sync([_row("a")], dataset_id=DATASET)
    await sync([_row("a")], dataset_id=OTHER_DATASET)
    await sync([_row("a", "v2")], dataset_id=DATASET)

    behind = await sync([], dataset_id=OTHER_DATASET)

    assert behind.ids == ["a"]


@pytest.mark.asyncio
async def test_a_load_read_for_one_table_is_still_read_for_the_others(sync):
    """One dlt load writes every table of a source; a sync reads only its selected ones."""
    dlt = pytest.importorskip("dlt")

    def two_tables(first_rows, second_rows):
        @dlt.resource(name="r1", table_name="t1", write_disposition="merge", primary_key="id")
        def r1():
            yield from first_rows

        @dlt.resource(name="r2", table_name="t2", write_disposition="merge", primary_key="id")
        def r2():
            yield from second_rows

        @dlt.source(name="two")
        def source():
            return r1, r2

        return source()

    await sync.source(two_tables([_row("k")], [_row("m")]))
    # dlt loads both edits in one load, then the add fails.
    await sync.source(two_tables([_row("k", "v2")], [_row("m", "v2")]), commit=False)
    await sync.source(two_tables([], []).with_resources("r2"))

    both = await sync.source(two_tables([], []))

    assert both.ids == ["k"]


@pytest.mark.asyncio
async def test_sources_sharing_a_pipeline_keep_their_own_committed_loads(sync):
    """Sources without a pipeline scope stage through one dlt pipeline."""
    await sync([_row("a")], name="first")
    await sync([_row("b")], name="second")

    again = await sync([], name="first")

    assert again.ids == []


@pytest.mark.asyncio
async def test_a_table_emptied_by_tombstones_forgets_its_last_document(sync):
    await sync([_row("a")])
    stored = await sync.stored()

    emptied = await sync([{"id": "a", "_deleted": True}])

    assert emptied.forgotten == {stored["a"]}


@pytest.mark.asyncio
async def test_a_source_that_staged_nothing_forgets_nothing(sync):
    await sync([_row("a")])

    nothing = await sync([], name="other_folder")

    assert nothing.items == [] and nothing.forgotten == set()
    assert sorted(await sync.stored()) == ["a"]


@pytest.mark.asyncio
async def test_a_table_with_a_falsy_id_is_compared_whole(sync):
    """An id of 0 or "" gets no external_id, so its table is read whole and compared, as before."""
    await sync([_row(0), _row(5)])
    zero = (await sync.stored())[None]

    deleted = await sync([{"id": 0, "_deleted": True}])

    assert zero in deleted.forgotten
    assert sorted(key for key in await sync.stored() if key) == ["5"]


@pytest.mark.asyncio
async def test_a_stored_document_without_an_id_makes_its_table_compared_whole(sync):
    """Only a whole read can tell whether its row is still staged, as the full sweep did."""
    await sync([_row("a")])
    legacy = SimpleNamespace(
        data_id=uuid4(),
        label="legacy",
        system_metadata={"source": "notes", "title": "legacy", "table_name": "docs"},
    )
    await _commit(sync.engine, [legacy])

    compared = await sync([_row("b")])

    assert compared.ids == ["a", "b"]
    assert compared.forgotten == {legacy.data_id}
    # The whole read committed the table's loads, so the next sync is keyed again.
    assert (await sync([])).ids == []


@pytest.mark.asyncio
async def test_a_table_whose_ids_repeat_is_read_whole(sync):
    """Under another merge key two staged rows can share an id, and both documents are
    stored under it; reading one of them alone would forget the other's document."""
    first = {**_row("a"), "key": "k1", "title": "first"}
    second = {**_row("a"), "key": "k2", "title": "second"}
    await sync([first, second], key="key")

    await sync([{**first, "content": "a v2"}], key="key")

    assert await sync.names() == ["first", "second"]


@pytest.mark.asyncio
async def test_a_table_without_an_id_column_is_compared_whole(sync):
    """Such rows are told apart only by reading them; a row a replace dropped is forgotten."""
    first = await sync.replaced(
        [{"title": "one", "content": "1"}, {"title": "two", "content": "2"}]
    )
    assert len(first.items) == 2

    second = await sync.replaced([{"title": "one", "content": "1"}])

    assert len(second.items) == 1
    assert second.forgotten == {next(i.data_id for i in first.items if i.label == "two")}


@pytest.mark.asyncio
async def test_a_row_with_a_null_id_is_read(sync):
    """A replace or append source can stage a NULL id, which no id lookup can find."""
    synced = await sync.replaced(
        [{"id": None, "title": "null", "content": "n"}, {"id": 1, "title": "one", "content": "1"}]
    )

    assert sorted(item.label for item in synced.items) == ["null", "one"]


@pytest.mark.asyncio
async def test_a_document_with_a_null_id_forgotten_here_comes_back(sync):
    """No id lookup finds a NULL id, so its table is compared whole and the next sync
    restores the document, as it restores any other forgotten one."""
    await sync(
        [{**_row("a"), "id": None, "key": "k1", "title": "null"}, {**_row("b"), "key": "k2"}],
        key="key",
    )
    await sync.forget(None)

    await sync([], key="key")

    assert await sync.names() == ["null", "t-b"]


@pytest.mark.asyncio
async def test_a_nested_table_without_load_ids_is_read_whole(sync):
    """dlt puts _dlt_load_id on root tables only; a nested list of objects has an id but none."""
    rows = [{**_row("p1"), "attachments": [{"id": "a1", "name": "spec.pdf"}]}]

    first = await sync(rows)
    unchanged = await sync([])

    assert "p1" in first.ids
    assert "p1" not in unchanged.ids


def test_postgres_keeps_committed_loads_in_the_schema_dlt_created():
    """dlt names the schema its own way: a__b becomes a_b, which _to_safe_ident keeps as a__b."""
    staged = ingest.StagedSource(
        dlt_db_name="dlt_database_a__b",
        dataset_name="a__b",
        pipeline_name="p",
        schema={"a_b.docs": {}},
        relational_config=SimpleNamespace(db_provider="postgres"),
    )

    assert ingest._document_loads_table(staged) == '"a_b"."_cognee_document_loads"'


@pytest.mark.asyncio
async def test_rows_by_id_are_the_datasets_own_and_read_past_one_batch(sync):
    """The sweep reads only the stale ids it was given, by primary key, never another dataset's."""
    own = await _store_rows(sync.engine, DATASET, DATA_ID_BATCH_SIZE + 1)
    other = await _store_rows(sync.engine, uuid4(), 1)

    rows = await by_ids_module.get_dataset_data_by_ids(DATASET, own + other)

    assert sorted(row.id for row in rows) == sorted(own)


async def _store_rows(engine, dataset_id, count):
    ids = [uuid4() for _ in range(count)]
    async with engine.get_async_session() as session:
        await session.execute(
            insert(Data.__table__),
            [
                {
                    "id": data_id,
                    "dataset_id": dataset_id,
                    "name": "row",
                    "content_hash": str(data_id),
                    "raw_data_location": "text://",
                    "owner_id": USER.id,
                    "system_metadata": {"source": "notes"},
                    "pipeline_status": {},
                    "token_count": -1,
                }
                for data_id in ids
            ],
        )
        await session.commit()
    return ids
