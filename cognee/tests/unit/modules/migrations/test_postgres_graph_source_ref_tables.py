"""The parts of ``postgres_graph_source_ref_tables`` that need no database.

What the migration does to a graph is covered against a real Postgres in
``cognee/tests/e2e/postgres/test_postgres_graph_source_ref_tables.py``.
"""

import inspect
from types import SimpleNamespace

import pytest

from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions import postgres_graph_source_ref_tables as migration


class _UnusableEngine:
    """Fails on any use: a graph stored elsewhere must not be opened."""

    def __init__(self, dialect: str) -> None:
        self.dialect = SimpleNamespace(name=dialect)

    def connect(self):
        raise AssertionError("the engine was used")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "graph_engine",
    [
        object(),
        SimpleNamespace(engine=None),
        SimpleNamespace(engine=object()),
        SimpleNamespace(engine=_UnusableEngine("sqlite")),
    ],
)
async def test_graph_outside_postgres_is_skipped(graph_engine):
    context = MigrationContext(graph_engine=graph_engine, vector_engine=None)

    await migration.migrate(context)
    await migration.downgrade(context)


@pytest.mark.asyncio
async def test_postgres_engine_is_used_whatever_the_adapter_class():
    context = MigrationContext(
        graph_engine=SimpleNamespace(engine=_UnusableEngine("postgresql")), vector_engine=None
    )

    with pytest.raises(AssertionError, match="the engine was used"):
        await migration.migrate(context)
    with pytest.raises(AssertionError, match="the engine was used"):
        await migration.downgrade(context)


def test_migration_is_frozen_against_the_adapter():
    """A migration must keep meaning what it meant, so it carries its own SQL."""
    source = inspect.getsource(migration)

    assert "postgres_demo" not in source
    assert "infrastructure.databases" not in source


@pytest.mark.parametrize("artifact", migration._ARTIFACTS)
@pytest.mark.parametrize("statement_for", [migration._copy_refs, migration._fill_arrays])
def test_every_batch_statement_is_bounded_by_the_batch_keys(artifact, statement_for):
    width = len(artifact.key)
    upper = "<= (" + ", ".join(f":upper_{index}" for index in range(width)) + ")"
    after = "> (" + ", ".join(f":after_{index}" for index in range(width)) + ")"
    # A copy reads the graph table twice (keys, run refs); a fill reads the ref table once.
    reads = 2 if statement_for is migration._copy_refs else 1

    first = statement_for("s", artifact, None)
    later = statement_for("s", artifact, ("a",) * width)

    assert (first.count(upper), first.count(after)) == (reads, 0)
    assert (later.count(upper), later.count(after)) == (reads, reads)
