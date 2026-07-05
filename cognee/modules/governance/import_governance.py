"""Import a governance bundle into a Cognee deployment.

This module is the round-trip companion to ``export_governance.py``. It
receives a ``GovernanceBundle``, verifies its hash chain for integrity, and
restores ACL rows into the relational database.

What this module does:
  - Verify the bundle's hash chain before touching any database rows
  - Upsert ``Permission`` rows (idempotent: only creates if name does not exist)
  - Upsert ``Principal`` rows (idempotent, keyed by the bundle's ``id``)
  - Upsert ``ACL`` rows (idempotent, keyed by (principal_id, permission_id, dataset_id))
  - Return a summary dict with counts of created/skipped rows

What this module does NOT do:
  - Import graph data — that is ``migration/import_source.py``'s job
  - Create datasets — the target dataset must already exist
  - Replay audit events into a live audit log (there is no such table yet;
    the bundle preserves the history, but the import is idempotent-upsert only)
  - Grant more permissions than the importing user is authorised to grant

Pattern: async, matches the ``export_dataset`` / ``export_governance_bundle``
style with explicit ``get_relational_engine`` usage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.modules.governance.hash_chain import verify_hash_chain
from cognee.modules.governance.models import GovernanceBundle, PrincipalKind

logger = get_logger("governance.import_governance")


async def import_governance_bundle(
    bundle: GovernanceBundle,
    target_dataset_id: str,
    user=None,
    *,
    skip_chain_verification: bool = False,
) -> Dict[str, Any]:
    """Restore ACL state from a ``GovernanceBundle`` into the relational database.

    Args:
        bundle: The governance bundle to import.
        target_dataset_id: UUID string of the dataset in *this* deployment
            to attach ACLs to. May differ from ``bundle.manifest.dataset_id``
            if the dataset was re-created with a new ID after migration.
        user: Authenticated user context. Defaults to the default user.
        skip_chain_verification: Set to ``True`` only in tests that deliberately
            construct un-chained bundles. In production this must be ``False``.

    Returns:
        A dict with keys ``"acls_created"``, ``"acls_skipped"``,
        ``"principals_created"``, ``"principals_skipped"``.

    Raises:
        HashChainError: If the bundle's audit trail fails hash verification and
            ``skip_chain_verification`` is ``False``.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import ACL, Permission, Principal
    from cognee.modules.users.methods import get_default_user
    from sqlalchemy import select

    if user is None:
        user = await get_default_user()

    if not skip_chain_verification:
        verify_hash_chain(bundle.audit_trail)
        logger.info("Governance bundle hash chain verified OK")

    engine = get_relational_engine()

    stats: Dict[str, int] = {
        "acls_created": 0,
        "acls_skipped": 0,
        "principals_created": 0,
        "principals_skipped": 0,
    }

    async with engine.get_async_session() as session:
        # ── 1. Upsert principals ──────────────────────────────────────────────
        for gp in bundle.principals:
            existing = await session.get(Principal, UUID(gp.id))
            if existing is not None:
                stats["principals_skipped"] += 1
                continue
            new_principal = Principal(id=UUID(gp.id))
            # Set polymorphic type attribute safely
            if hasattr(new_principal, "type"):
                new_principal.type = gp.kind.value  # type: ignore[assignment]
            session.add(new_principal)
            stats["principals_created"] += 1

        await session.flush()

        # ── 2. Upsert ACLs ───────────────────────────────────────────────────
        for gacl in bundle.acls:
            # Ensure the permission row exists (idempotent)
            perm_stmt = select(Permission).where(Permission.name == gacl.permission_name)
            perm_result = await session.execute(perm_stmt)
            permission = perm_result.scalar_one_or_none()
            if permission is None:
                permission = Permission(name=gacl.permission_name)
                session.add(permission)
                await session.flush()

            # Check for an existing ACL row with the same triple
            existing_stmt = select(ACL).where(
                ACL.principal_id == UUID(gacl.principal_id),
                ACL.permission_id == permission.id,
                ACL.dataset_id == UUID(target_dataset_id),
            )
            existing_acl = (await session.execute(existing_stmt)).scalar_one_or_none()
            if existing_acl is not None:
                stats["acls_skipped"] += 1
                continue

            new_acl = ACL(
                principal_id=UUID(gacl.principal_id),
                permission_id=permission.id,
                dataset_id=UUID(target_dataset_id),
                created_at=gacl.granted_at or datetime.now(timezone.utc),
            )
            session.add(new_acl)
            stats["acls_created"] += 1

        await session.commit()

    logger.info(
        "Governance bundle import complete: %s",
        stats,
    )
    return stats
