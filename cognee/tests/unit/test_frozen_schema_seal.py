"""The frozen base schema is FROZEN: its fingerprint is pinned here.

``cognee.alembic.frozen_schema`` is the one surface both the initial revision
(8057ae7329c2) and the reconcile revision (1c22e6cb5aec) build from. A
single module is easy to "regenerate" from live models — and that is the one
failure the migration/model lockstep guard cannot detect: if the frozen
surface silently absorbs a model change instead of a new migration carrying
it, chain and models agree and every probe stays green while existing
databases never receive the change.

So the fingerprint (tables, columns with type/nullability/server default,
primary keys, foreign keys, unique constraints, indexes — dialect-independent)
is pinned. Changing this constant is a deliberate act: a re-certification that
a real chain-migrated database matches the models, recorded in the same
commit. Comments and formatting in the module do not affect it; any schema
change does.
"""

from cognee.alembic.frozen_schema import fingerprint

# Certified model surface at chain head d1e2f3a4b5c6 (2026-09-01) plus the
# chain-provided ix_pipeline_runs_dataset_pipeline_created_at.
FROZEN_FINGERPRINT = "4662a81112d3f7194a1ebb76e20a0947e053f4c723ae21ef8a2efdd7a59a32d7"


def test_frozen_schema_is_unchanged():
    assert fingerprint() == FROZEN_FINGERPRINT, (
        "cognee.alembic.frozen_schema changed. It is FROZEN: schema changes ship as new "
        "migrations at the head, never by editing the frozen surface. If this is a "
        "deliberate re-certification, update FROZEN_FINGERPRINT in the same commit and "
        "say so in the message."
    )
