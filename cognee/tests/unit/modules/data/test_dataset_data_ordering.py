"""Regression test: dataset listings are ordered by an indexed column.

get_dataset_data() used to sort by ``data_size DESC``. That column has no
index, and DataDTO does not even expose it -- so every listing seq-scanned
``data`` and sorted it externally to produce an order nobody could see. A LIMIT
does not avoid the sort: the whole partition has to be ordered to find the top
N. It is now ``created_at DESC, id``, which ``ix_data_dataset_created`` serves
end to end.
"""

import re

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex

from cognee.modules.data.models import Data

INDEX_NAME = "ix_data_dataset_created"


def _listing_index():
    for index in Data.__table__.indexes:
        if index.name == INDEX_NAME:
            return index
    return None


def test_model_declares_the_listing_index():
    assert _listing_index() is not None, f"{INDEX_NAME} missing from Data.__table_args__"


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
def test_index_covers_filter_and_sort_in_order(dialect):
    """Column order is what makes it usable: filter first, then the sort keys."""
    ddl = str(CreateIndex(_listing_index()).compile(dialect=dialect))

    assert "(dataset_id, created_at DESC, id)" in " ".join(ddl.split())


@pytest.mark.asyncio
async def test_method_issues_the_indexed_ordering(monkeypatch):
    """Reads the ORDER BY off the statement get_dataset_data actually builds.

    Compiled SQL, not the module source: a comment naming the old column would
    satisfy a text search while the query still sorted the wrong way.
    """
    import uuid
    from contextlib import asynccontextmanager

    import importlib

    # import_module, not `from ... import get_dataset_data`: the package
    # re-exports the function under the module's own name, so the plain import
    # binds the function and monkeypatching it would silently do nothing.
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")

    captured = {}

    class _Session:
        async def execute(self, statement):
            captured["sql"] = " ".join(str(statement.compile(dialect=postgresql.dialect())).split())

            class _Result:
                def scalars(self):
                    class _S:
                        def all(self_inner):
                            return []

                    return _S()

            return _Result()

    class _Engine:
        @asynccontextmanager
        async def get_async_session(self):
            yield _Session()

    monkeypatch.setattr(module, "get_relational_engine", lambda: _Engine())

    await module.get_dataset_data(uuid.uuid4())

    order_by = re.search(r"ORDER BY (.+?)(?: LIMIT|$)", captured["sql"]).group(1).strip()
    assert "data_size" not in order_by, "the unindexed column must be gone from the sort"
    assert "created_at DESC" in order_by
    assert order_by.endswith("data.id"), "id tiebreak keeps paging stable"


def test_migration_and_model_agree_on_the_index():
    """The alembic revision and the model must create the same index."""
    from pathlib import Path

    revision = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "e7f9a1c3d5b8_add_data_dataset_created_index.py"
    )
    body = revision.read_text()

    assert f'INDEX_NAME = "{INDEX_NAME}"' in body
    assert body.count("(dataset_id, created_at DESC, id)") == 2, (
        "both the postgres and non-postgres branches must build the same index"
    )
    assert "CONCURRENTLY" in body, "this table is large on real deployments"
