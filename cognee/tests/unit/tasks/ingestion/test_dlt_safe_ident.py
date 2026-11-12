"""The dlt staging identifier derived from a dataset name (SDK-1464).

dlt uses the sanitized dataset name as its staging dataset. On a Postgres
destination that is a schema, and Postgres reserves the ``pg_`` prefix for
system schemas: a dataset named ``pg_sales`` or ``PG Sales`` failed every dlt
input with ``ReservedName: unacceptable schema name``, after the staging
database had already been created. Such a name is now prefixed exactly like a
name that starts with a digit, on every destination. So is ``information_schema``,
a schema every Postgres database already holds (SDK-1466).
"""

import importlib
import re
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR
from cognee.tasks.ingestion.exceptions.exceptions import InvalidDLTArgumentError

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.fixture
def sqlite_staging(tmp_path, monkeypatch):
    """Point dlt staging at a SQLite file under ``tmp_path`` and record what it is told.

    Returns the dict the destination and pipeline factories write their
    ``dlt_db_name`` / ``dataset_name`` arguments into.
    """
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

    seen = {}

    def destination(dlt_db_name):
        seen["dlt_db_name"] = dlt_db_name
        return dlt.destinations.sqlalchemy(
            credentials={"database": str(tmp_path / dlt_db_name), "drivername": "sqlite"},
        )

    monkeypatch.setattr(ingest, "get_dlt_destination", destination)

    pipeline_factory = dlt.pipeline

    def pipeline(**kwargs):
        seen["dataset_name"] = kwargs["dataset_name"]
        return pipeline_factory(pipelines_dir=str(tmp_path / "pipelines"), **kwargs)

    monkeypatch.setattr(dlt, "pipeline", pipeline)
    return seen


@pytest.mark.parametrize(
    ("dataset_name", "ident"),
    [
        ("pg_sales", "d_pg_sales"),
        ("PG Sales", "d_pg_sales"),
        ("pg-sales", "d_pg_sales"),
        ("PG_SALES", "d_pg_sales"),
        # Leading underscores are stripped before the check, so this is a pg_ name too.
        ("_pg_sales", "d_pg_sales"),
        # The bare name would give dlt a merge staging schema named ``pg_staging``.
        ("pg", "d_pg"),
        ("pg_", "d_pg"),
        ("PG", "d_pg"),
        # Present in every Postgres database and owned by the bootstrap superuser.
        ("information_schema", "d_information_schema"),
        ("Information Schema", "d_information_schema"),
    ],
)
def test_a_reserved_name_is_prefixed(dataset_name, ident):
    assert ingest._to_safe_ident(dataset_name) == ident


@pytest.mark.parametrize("dataset_name", ["pg", "pg_", "pg_sales", "PG Sales", "sales"])
def test_the_merge_staging_schema_is_never_reserved(dataset_name):
    """dlt derives the merge staging schema as ``<dataset>_staging``."""
    assert not f"{ingest._to_safe_ident(dataset_name)}_staging".startswith("pg_")


@pytest.mark.parametrize(
    "dataset_name",
    [
        # Only the exact ``pg_`` prefix is reserved by Postgres.
        "pgsales",
        "pgvector_data",
        "my_pg_sales",
        # Only the exact system schema name exists in every database.
        "information_schemas",
        "my_information_schema",
        "sales",
        "main_dataset",
    ],
)
def test_other_names_are_left_alone(dataset_name):
    assert ingest._to_safe_ident(dataset_name) == dataset_name.strip("_")


def test_a_leading_digit_is_still_prefixed():
    assert ingest._to_safe_ident("2024_sales") == "d_2024_sales"


def test_the_prefix_is_applied_before_the_length_cut():
    ident = ingest._to_safe_ident("pg_" + "x" * 80)

    assert ident.startswith("d_pg_")
    assert len(ident) == 63


def test_an_empty_name_is_refused():
    with pytest.raises(InvalidDLTArgumentError):
        ingest._to_safe_ident("___")


@pytest.mark.asyncio
async def test_a_pg_dataset_stages_under_the_prefixed_name(tmp_path, sqlite_staging):
    """The staging database and the dlt dataset agree on the prefixed name.

    Runs the real dlt pipeline on SQLite, where a ``pg_`` schema would have
    worked, to prove the rename applies on every destination.
    """
    import dlt

    @dlt.resource(name="orders", primary_key="id")
    def orders():
        yield {"id": 1, "total": 10}

    rows = await ingest.ingest_dlt_source(
        orders(),
        "PG Sales",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    assert sqlite_staging == {
        "dlt_db_name": "dlt_database_d_pg_sales",
        "dataset_name": "d_pg_sales",
    }
    assert len(rows) == 1
    # The raw name is kept for messages and the pipeline name; nothing on disk uses it.
    assert not any("pg_sales" in p.name and "d_pg_sales" not in p.name for p in tmp_path.iterdir())


def _ident_before_the_pg_rule(name: str) -> str:
    """``_to_safe_ident`` as released before SDK-1466: digits prefixed, ``pg`` not."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return (f"d_{name}" if name[0].isdigit() else name)[:63]


@pytest.mark.asyncio
async def test_an_existing_pg_dataset_keeps_every_document_through_the_rename(
    tmp_path, sqlite_staging, monkeypatch
):
    """A SQLite dataset staged under the old name re-walks fully after the rename.

    Connectors keep their incremental cursor in dlt's local pipeline state,
    whose name derives from the raw dataset name and so survives the rename.
    dlt wipes that state when the destination dataset does not exist, so the
    first sync after the upgrade re-emits every row into the new staging
    database instead of only the delta; otherwise the document orphan cleanup
    would forget every unchanged document.
    """
    import dlt

    source_rows = {
        1: {"id": 1, "updated_at": 5, "content": "unchanged"},
        2: {"id": 2, "updated_at": 5, "content": "second"},
    }

    def make_source():
        @dlt.resource(name="pages", primary_key="id", write_disposition="merge")
        def pages():
            state = dlt.current.resource_state()
            last = state.get("last", 0)
            for row in source_rows.values():
                if row["updated_at"] > last:
                    state["last"] = max(state.get("last", 0), row["updated_at"])
                    yield row

        resource = pages()
        setattr(resource, DOCUMENT_SOURCE_ATTR, "test")
        setattr(resource, PIPELINE_SCOPE_ATTR, "test-scope")
        return resource

    async def sync():
        rows = await ingest.ingest_dlt_source(
            make_source(),
            "pg_sales",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
        return sorted(row.row_data["id"] for row in rows)

    rule = {"ident": _ident_before_the_pg_rule}
    current_rule = ingest._to_safe_ident
    monkeypatch.setattr(ingest, "_to_safe_ident", lambda name: rule["ident"](name))
    assert await sync() == [1, 2]
    assert sqlite_staging["dataset_name"] == "pg_sales"
    # An unchanged sync under the old rule read the retained rows back.
    assert await sync() == [1, 2]

    # The upgrade: the same pipeline state, the current rule, one row changed.
    rule["ident"] = current_rule
    source_rows[2] = {"id": 2, "updated_at": 9, "content": "second, edited"}
    assert await sync() == [1, 2]
    assert sqlite_staging["dataset_name"] == "d_pg_sales"
    # The next delta is incremental again under the new name.
    source_rows[2] = {"id": 2, "updated_at": 12, "content": "second, edited twice"}
    assert await sync() == [1, 2]
