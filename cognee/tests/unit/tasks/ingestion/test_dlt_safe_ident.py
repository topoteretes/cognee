"""The dlt staging identifier derived from a dataset name (SDK-1464).

dlt uses the sanitized dataset name as its staging dataset. On a Postgres
destination that is a schema, and Postgres reserves the ``pg_`` prefix for
system schemas: a dataset named ``pg_sales`` or ``PG Sales`` failed every dlt
input with ``ReservedName: unacceptable schema name``, after the staging
database had already been created. Such a name is now prefixed exactly like a
name that starts with a digit, on every destination.
"""

import importlib
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.exceptions.exceptions import InvalidDLTArgumentError

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.mark.parametrize(
    ("dataset_name", "ident"),
    [
        ("pg_sales", "d_pg_sales"),
        ("PG Sales", "d_pg_sales"),
        ("pg-sales", "d_pg_sales"),
        ("PG_SALES", "d_pg_sales"),
        # Leading underscores are stripped before the check, so this is a pg_ name too.
        ("_pg_sales", "d_pg_sales"),
    ],
)
def test_a_pg_prefixed_name_is_prefixed(dataset_name, ident):
    assert ingest._to_safe_ident(dataset_name) == ident


@pytest.mark.parametrize(
    "dataset_name",
    [
        # Only the exact ``pg_`` prefix is reserved by Postgres.
        "pg",
        "pg_",
        "pgsales",
        "pgvector_data",
        "my_pg_sales",
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
async def test_a_pg_dataset_stages_under_the_prefixed_name(tmp_path, monkeypatch):
    """The staging database and the dlt dataset agree on the prefixed name.

    Runs the real dlt pipeline on SQLite, where a ``pg_`` schema would have
    worked, to prove the rename applies on every destination.
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

    assert seen == {"dlt_db_name": "dlt_database_d_pg_sales", "dataset_name": "d_pg_sales"}
    assert len(rows) == 1
    # The raw name is kept for messages and the pipeline name; nothing on disk uses it.
    assert not any("pg_sales" in p.name and "d_pg_sales" not in p.name for p in tmp_path.iterdir())
