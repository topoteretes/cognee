"""CI lockstep guard: every model change and its migration must land together.

The fresh-database bootstrap runs ``create_all()`` from ``Base.metadata`` and
then STAMPS Alembic head — asserting, without checking, that the models and the
chain describe the same schema. When that assertion silently broke (a chain
carrying a ``pipeline_runs`` extension paired with models that predate it),
fresh databases were provisioned with a schema that disagreed with the revision
they were stamped at, and nothing noticed until the missing columns failed at
runtime, eleven days later.

This test makes the assertion checkable at PR time, WITHOUT replaying the
historical chain (the early revisions assume the base schema of their own era,
and no migration ever created the base tables — the chain is a delta log, not
a creation history). Instead:

1. ``schema_baseline.json`` is a FROZEN snapshot of ``Base.metadata`` (tables
   and column names) taken at a certified point — a revision where a real
   chain-migrated database was verified to match the models exactly.
2. The test builds a database from that snapshot, stamps it at the baseline
   revision, and runs ``alembic upgrade head`` through cognee's own runner —
   genuinely executing ONLY the migrations added after the freeze, against a
   base that predates them.
3. The result is compared against the current models, both directions:

   - a migration that adds a table or column the models do not declare fails
     (the mis-stamp incident direction: fresh databases built by ``create_all``
     will LACK it while stamped as having it);
   - a model table or column no migration provides also fails (existing
     databases would never receive it on upgrade).

Re-freezing the baseline erases the guard's memory, so it is a deliberate act,
done only when the head state is re-certified (e.g. a real chain-migrated
database again matches the models exactly):

    uv run python cognee/tests/integration/infrastructure/relational/test_migration_model_lockstep.py

regenerates ``schema_baseline.json`` at the current head. Never regenerate it
to silence a failure — the failure IS the finding.
"""

import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy import inspect

BASELINE_PATH = Path(__file__).with_name("schema_baseline.json")

_GET_ENGINE = "cognee.infrastructure.databases.relational.get_relational_engine"

# Alembic's own bookkeeping lives outside the models on purpose.
_IGNORED_TABLES = {"alembic_version"}


def _register_all_models() -> None:
    """Import every ``cognee.modules.<x>.models`` package so ``Base.metadata``
    is complete and deterministic — importing the API surface registers most
    models as a side effect, but not all of them (e.g. the integrations and
    sync models), and the snapshot must not depend on which corner of cognee a
    test process happened to touch first.

    Only the package ``__init__`` is imported, deliberately: that is the set a
    real process registers, so it is the set ``create_all`` actually builds.
    Orphaned model files a package does not export (e.g. the unimportable
    ``pipelines/models/Task.py``) are not part of any real schema."""
    import cognee.modules

    root = Path(cognee.modules.__file__).parent
    for sub in sorted(path.name for path in root.iterdir() if path.is_dir()):
        if (root / sub / "models" / "__init__.py").exists():
            importlib.import_module(f"cognee.modules.{sub}.models")


def _model_schema() -> dict[str, list[str]]:
    _register_all_models()
    from cognee.infrastructure.databases.relational import Base

    return {
        table.name: sorted(column.name for column in table.columns)
        for table in Base.metadata.sorted_tables
    }


def _current_head() -> str:
    from alembic.script import ScriptDirectory

    from cognee.modules.migrations.startup import _build_alembic_config

    return ScriptDirectory.from_config(_build_alembic_config()).get_current_head()


def _chain_schema_from_baseline(baseline: dict, tmp_dir: str) -> dict[str, list[str]]:
    """Build the baseline snapshot as a real SQLite database stamped at the
    baseline revision, replay every migration above it, and reflect the result.

    Column types in the snapshot database are generic (SQLite is dynamically
    typed and the comparison is by NAME); what matters is that the migrations
    added since the freeze run for real against a base that predates them.
    """
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )
    from cognee.modules.migrations.startup import run_relational_migrations

    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_dir}/lockstep.db")

    async def _run():
        try:
            async with adapter.engine.begin() as connection:

                def _create(sync_connection):
                    metadata = sa.MetaData()
                    for name, columns in baseline["tables"].items():
                        sa.Table(name, metadata, *(sa.Column(col, sa.Text()) for col in columns))
                    metadata.create_all(sync_connection)

                await connection.run_sync(_create)
                await connection.execute(
                    sa.text(
                        "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL, "
                        "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
                    )
                )
                await connection.execute(
                    sa.text("INSERT INTO alembic_version VALUES (:revision)"),
                    {"revision": baseline["baseline_revision"]},
                )

            with patch(_GET_ENGINE, return_value=adapter):
                await run_relational_migrations("head")

            async with adapter.engine.connect() as connection:
                return await connection.run_sync(
                    lambda sync_conn: {
                        table: sorted(
                            column["name"] for column in inspect(sync_conn).get_columns(table)
                        )
                        for table in inspect(sync_conn).get_table_names()
                        if table not in _IGNORED_TABLES
                    }
                )
        finally:
            await adapter.engine.dispose()

    return asyncio.run(_run())


class TestMigrationModelLockstep(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(BASELINE_PATH.read_text())

    def test_baseline_revision_is_still_in_the_chain(self):
        """A vanished baseline revision means history was edited; the guard's
        anchor must then be re-frozen deliberately, not silently skipped."""
        from alembic.script import ScriptDirectory

        from cognee.modules.migrations.startup import _build_alembic_config

        script = ScriptDirectory.from_config(_build_alembic_config())
        self.assertIsNotNone(
            script.get_revision(self.baseline["baseline_revision"]),
            f"Baseline revision {self.baseline['baseline_revision']!r} is no longer in the "
            "Alembic chain. If history was rewritten on purpose, re-certify the schema and "
            "regenerate schema_baseline.json (see module docstring).",
        )

    def test_migrations_since_baseline_keep_chain_and_models_in_lockstep(self):
        expected = _model_schema()
        self.assertTrue(expected, "Base.metadata is empty — models were not imported")

        with tempfile.TemporaryDirectory() as tmp_dir:
            actual = _chain_schema_from_baseline(self.baseline, tmp_dir)

        problems = []
        for table in sorted(set(actual) - set(expected)):
            problems.append(
                f"the chain creates table '{table}' that no model declares "
                "(migration landed without its model change?)"
            )
        for table in sorted(set(expected) - set(actual)):
            problems.append(
                f"models declare table '{table}' that no migration creates "
                "(model change landed without its migration?)"
            )
        for table in sorted(set(actual) & set(expected)):
            extra = sorted(set(actual[table]) - set(expected[table]))
            missing = sorted(set(expected[table]) - set(actual[table]))
            if extra:
                problems.append(
                    f"table '{table}': the chain adds column(s) {extra} the models do not "
                    "declare — fresh databases built by create_all will LACK them while "
                    "stamped as having them (the mis-stamp incident direction)"
                )
            if missing:
                problems.append(
                    f"table '{table}': models declare column(s) {missing} that no migration "
                    "adds — existing databases will never receive them on upgrade"
                )

        self.assertEqual(
            problems,
            [],
            "\n\nThe Alembic chain and the ORM models disagree:\n  - "
            + "\n  - ".join(problems)
            + "\n\nA fresh database is built from the models and STAMPED at the chain's head; "
            "the two must describe the same schema. Ship the model change and its migration "
            "in the same PR.",
        )


def generate_baseline() -> dict:
    """Regenerate the frozen snapshot at the current head. Deliberate act only —
    see the module docstring."""
    baseline = {"baseline_revision": _current_head(), "tables": _model_schema()}
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    return baseline


if __name__ == "__main__":
    generated = generate_baseline()
    print(
        f"schema_baseline.json regenerated at revision {generated['baseline_revision']} "
        f"({len(generated['tables'])} tables). Commit it only alongside a re-certification "
        "of the head schema."
    )
