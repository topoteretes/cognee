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
a creation history). The mechanics live in ``cognee.modules.migrations.lockstep``
so downstream deployments with a vendored chain and extra models (e.g. cognee
cloud) can reuse them with their own baseline, model registration, and
``script_location``:

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
import tempfile
import unittest
from pathlib import Path

from cognee.modules.migrations import lockstep

BASELINE_PATH = Path(__file__).with_name("schema_baseline.json")


class TestMigrationModelLockstep(unittest.TestCase):
    def setUp(self):
        self.baseline = lockstep.load_baseline(BASELINE_PATH)

    def test_baseline_revision_is_still_in_the_chain(self):
        """A vanished baseline revision means history was edited; the guard's
        anchor must then be re-frozen deliberately, not silently skipped."""
        self.assertTrue(
            lockstep.revision_exists(self.baseline["baseline_revision"]),
            f"Baseline revision {self.baseline['baseline_revision']!r} is no longer in the "
            "Alembic chain. If history was rewritten on purpose, re-certify the schema and "
            "regenerate schema_baseline.json (see module docstring).",
        )

    def test_migrations_since_baseline_keep_chain_and_models_in_lockstep(self):
        expected = lockstep.collect_model_schema()
        self.assertTrue(expected, "Base.metadata is empty — models were not imported")

        with tempfile.TemporaryDirectory() as tmp_dir:
            actual = asyncio.run(
                lockstep.replay_from_baseline(
                    self.baseline, f"sqlite+aiosqlite:///{tmp_dir}/lockstep.db"
                )
            )

        problems = lockstep.compare_schemas(expected, actual)
        self.assertEqual(
            problems,
            [],
            "\n\nThe Alembic chain and the ORM models disagree:\n  - "
            + "\n  - ".join(problems)
            + "\n\nA fresh database is built from the models and STAMPED at the chain's head; "
            "the two must describe the same schema. Ship the model change and its migration "
            "in the same PR.",
        )


if __name__ == "__main__":
    generated = lockstep.generate_baseline(BASELINE_PATH)
    print(
        f"schema_baseline.json regenerated at revision {generated['baseline_revision']} "
        f"({len(generated['tables'])} tables). Commit it only alongside a re-certification "
        "of the head schema."
    )
