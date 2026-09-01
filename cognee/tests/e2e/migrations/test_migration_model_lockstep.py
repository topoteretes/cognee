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
a creation history). It runs as a STANDALONE e2e check in CI (the 'Migration/Model Lockstep Guard'
job in e2e_tests.yml) so a lockstep break is visible as its own failed check,
not buried in a shard. The mechanics live in ``cognee.modules.migrations.lockstep``
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

    uv run python cognee/tests/e2e/migrations/test_migration_model_lockstep.py --regenerate --force

regenerates ``schema_baseline.json`` at the current head. Never regenerate it
to silence a failure — the failure IS the finding.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from cognee.modules.migrations import lockstep

BASELINE_PATH = Path(__file__).with_name("schema_baseline.json")

# Pinned explicitly so the ambient COGNEE_ALEMBIC_PATH override (used by
# vendored-chain deployments) can never retarget this upstream test at a
# foreign migration chain.
SCRIPT_LOCATION = lockstep.packaged_script_location()


class TestMigrationModelLockstep(unittest.TestCase):
    def setUp(self):
        self.baseline = lockstep.load_baseline(BASELINE_PATH)

    def test_baseline_revision_is_a_valid_anchor(self):
        """The anchor must resolve to EXACTLY the stored id and be an ancestor
        of head — a history rewrite that breaks either means the guard must be
        re-frozen deliberately, not silently skipped."""
        problem = lockstep.anchor_error(self.baseline["baseline_revision"], SCRIPT_LOCATION)
        self.assertIsNone(
            problem,
            f"Baseline anchor is invalid: {problem}. If migration history was rewritten "
            "on purpose, re-certify the schema and regenerate schema_baseline.json "
            "(see module docstring).",
        )

    def test_migrations_since_baseline_keep_chain_and_models_in_lockstep(self):
        expected = lockstep.collect_model_schema()
        self.assertTrue(expected, "Base.metadata is empty — models were not imported")

        with tempfile.TemporaryDirectory() as tmp_dir:
            actual = asyncio.run(
                lockstep.replay_from_baseline(
                    self.baseline,
                    f"sqlite+aiosqlite:///{tmp_dir}/lockstep.db",
                    script_location=SCRIPT_LOCATION,
                )
            )

        problems = lockstep.compare_schemas(
            expected, actual, baseline_tables=self.baseline["tables"]
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


def _regenerate(force: bool) -> int:
    """Deliberate re-freeze only — never a way to silence a failing guard."""
    if BASELINE_PATH.exists() and not force:
        print(
            "Refusing to overwrite the existing schema_baseline.json: regenerating "
            "erases the guard's memory and would certify whatever skew exists right "
            "now. Re-certify the head schema first, then rerun with:\n"
            f"  python {Path(__file__).name} --regenerate --force"
        )
        return 1
    generated = lockstep.generate_baseline(BASELINE_PATH, script_location=SCRIPT_LOCATION)
    print(
        f"schema_baseline.json regenerated at revision {generated['baseline_revision']} "
        f"({len(generated['tables'])} tables). Commit it only alongside a re-certification "
        "of the head schema."
    )
    return 0


if __name__ == "__main__":
    # Bare execution runs the tests — regeneration is behind an explicit flag,
    # so the habitual "run the file to debug a failure" gesture can never
    # silently rewrite the baseline.
    if "--regenerate" in sys.argv:
        sys.exit(_regenerate(force="--force" in sys.argv))
    unittest.main()
