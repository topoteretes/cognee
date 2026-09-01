"""CI lockstep guard: every model change and its migration must land together.

The fresh-database bootstrap runs ``create_all()`` from ``Base.metadata`` and
then STAMPS Alembic head — asserting, without checking, that the models and the
chain describe the same schema. When that assertion silently broke (a chain
carrying a ``pipeline_runs`` extension paired with models that predate it),
fresh databases were provisioned with a schema that disagreed with the revision
they were stamped at, and nothing noticed until the missing columns failed at
runtime, eleven days later.

This suite makes the assertion checkable at PR time, two ways. The initial
migration now creates the base schema (guarded create_all), turning the chain
into a self-sufficient creation history — safe because every DDL-bearing
revision is inspector-guarded (audited file by file). That enables the
whole-history probe: `alembic upgrade head` on an empty database, compared
against the models. The frozen-baseline probe complements it, catching the
direction the whole-history replay cannot see (a model change with no
migration), for everything after the freeze. It runs as a STANDALONE e2e check in CI (the 'Migration/Model Lockstep Guard'
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

    def test_entire_chain_from_empty_database_matches_the_models(self):
        """Whole-history probe: the initial migration now creates the base
        schema (guarded create_all), so `alembic upgrade head` on an EMPTY
        database builds the chain's complete idea of the schema. Anything it
        contains that the models do not declare — wherever in history it was
        introduced, including a chain paired with an older package downstream
        — fails here. `notebooks` is the one known, accepted legacy table
        (model removed, drop migration never written).

        The reverse direction (model change without a migration) is invisible
        to this probe by construction and is covered by the baseline test
        below."""
        expected = lockstep.collect_model_schema()

        with tempfile.TemporaryDirectory() as tmp_dir:
            actual = asyncio.run(
                lockstep.replay_entire_chain(
                    f"sqlite+aiosqlite:///{tmp_dir}/full_chain.db",
                    script_location=SCRIPT_LOCATION,
                )
            )

        problems = [
            problem
            for problem in lockstep.compare_schemas(expected, actual)
            # Whole-history replay builds its base FROM the models, so "models
            # declare X that no migration creates" is structurally impossible
            # here and would only echo baseline-test findings; keep this probe
            # focused on its one detectable direction.
            if "no migration creates" not in problem and "no migration adds" not in problem
        ]
        self.assertEqual(
            problems,
            [],
            "\n\nThe chain, replayed from an empty database, disagrees with the models:\n  - "
            + "\n  - ".join(problems)
            + "\n\nA migration added something no model declares. Fresh databases built by "
            "create_all will LACK it while stamped as having it (the mis-stamp incident). "
            "Ship the model change and its migration in the same PR.",
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
