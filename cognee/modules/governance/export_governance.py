"""Export the governance state (ACLs + audit trail) of a Cognee dataset.

This module reads the live ACL/permission rows from the relational database
for a given dataset and serialises them into a ``GovernanceBundle``. It is the
governance-side companion to ``cognee.modules.migration.export.export_dataset``.

What this module does:
  - Query the relational engine for ACL rows scoped to ``dataset_id``
  - Resolve the principals (users/roles/tenants) referenced by those ACLs
  - Build an audit trail of ``grant`` events from the ACL snapshot (one event
    per ACL row) — plus any ``denied`` events provided by the caller
  - Apply a SHA-256 hash chain across the audit trail for tamper evidence
  - Return a ``GovernanceBundle`` ready for serialisation

What this module does NOT do:
  - Export graph data (nodes/edges) — that is ``migration/export.py``'s job
  - Write to disk — callers decide where to save the bundle
  - Implement a live audit log — it synthesises events from the *current* ACL
    state, which is a snapshot, not a full history (a full history would require
    a separate audit-log table that does not exist yet)

Pattern: follows the ``export_dataset`` style — async, accepts a ``user``
argument, uses ``get_default_user`` if none is provided.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.modules.governance.models import (
    AuditEventKind,
    GovernanceACL,
    GovernanceAuditEvent,
    GovernanceBundle,
    GovernanceBundleManifest,
    GovernanceDeniedAction,
    GovernancePrincipal,
    PrincipalKind,
)
from cognee.modules.governance.hash_chain import build_hash_chain

logger = get_logger("governance.export_governance")


def _principal_kind(principal_type: str) -> PrincipalKind:
    """Map a Principal polymorphic identity string to ``PrincipalKind``."""
    mapping = {
        "user": PrincipalKind.user,
        "role": PrincipalKind.role,
        "tenant": PrincipalKind.tenant,
    }
    return mapping.get(principal_type, PrincipalKind.user)


async def export_governance_bundle(
    dataset_id: str,
    dataset_name: str,
    user=None,
    denied_actions: Optional[List[GovernanceDeniedAction]] = None,
    notes: Optional[List[str]] = None,
) -> GovernanceBundle:
    """Build a portable ``GovernanceBundle`` for the given dataset.

    Reads ACL rows from the relational database, resolves principals, and
    synthesises a hash-chained audit trail.  Denied actions (if provided by
    the caller) are appended to the trail as ``denied`` events *before* the
    chain is sealed so they are part of the same tamper-evident log.

    Args:
        dataset_id: UUID string of the dataset to export governance for.
        dataset_name: Human-readable dataset name (stored in the manifest).
        user: Authenticated user context. Defaults to the default user.
        denied_actions: Optional list of ``GovernanceDeniedAction`` records to
            include in the audit trail alongside the snapshot-derived events.
        notes: Optional notes to include in the bundle manifest.

    Returns:
        A ``GovernanceBundle`` with a valid hash chain. Call ``bundle.save(path)``
        to persist it.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import ACL, Permission, Principal
    from cognee.modules.users.methods import get_default_user
    from sqlalchemy import select

    if user is None:
        user = await get_default_user()

    denied_actions = denied_actions or []
    notes = notes or []

    engine = get_relational_engine()

    principals: List[GovernancePrincipal] = []
    acls: List[GovernanceACL] = []
    audit_events: List[GovernanceAuditEvent] = []

    _seen_principal_ids: set[str] = set()

    async with engine.get_async_session() as session:
        # Load ACL rows for this dataset (including their principal + permission)
        stmt = (
            select(ACL, Permission)
            .join(Permission, ACL.permission_id == Permission.id)
            .where(ACL.dataset_id == dataset_id)
        )
        result = await session.execute(stmt)
        acl_rows = result.all()

        for acl_row, perm_row in acl_rows:
            acl_obj = GovernanceACL(
                id=str(acl_row.id),
                principal_id=str(acl_row.principal_id),
                permission_name=perm_row.name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                granted_at=acl_row.created_at,
            )
            acls.append(acl_obj)

            # Resolve principal once per unique id
            if str(acl_row.principal_id) not in _seen_principal_ids:
                _seen_principal_ids.add(str(acl_row.principal_id))
                p_stmt = select(Principal).where(Principal.id == acl_row.principal_id)
                p_result = await session.execute(p_stmt)
                principal_row = p_result.scalar_one_or_none()
                if principal_row is not None:
                    kind = _principal_kind(
                        getattr(principal_row, "type", "user")
                    )
                    principals.append(
                        GovernancePrincipal(
                            id=str(principal_row.id),
                            kind=kind,
                            name=getattr(principal_row, "name", None),
                            email=getattr(principal_row, "email", None),
                            tenant_id=getattr(principal_row, "tenant_id", None),
                        )
                    )

            # Synthesise a ``grant`` audit event from the ACL row
            audit_events.append(
                GovernanceAuditEvent(
                    kind=AuditEventKind.grant,
                    actor_id=str(user.id) if user else None,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    acl_id=str(acl_row.id),
                    occurred_at=acl_row.created_at or datetime.now(timezone.utc),
                    metadata={"permission": perm_row.name},
                )
            )

    # Append denied-action events (these come from the caller, e.g. the search
    # or write path when it rejects an unauthorised request)
    for denied in denied_actions:
        audit_events.append(
            GovernanceAuditEvent(
                kind=AuditEventKind.denied,
                actor_id=denied.principal_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                denied_action=denied,
                occurred_at=denied.occurred_at,
            )
        )

    # Add an export event so the bundle self-documents its own creation
    audit_events.append(
        GovernanceAuditEvent(
            kind=AuditEventKind.export,
            actor_id=str(user.id) if user else None,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            occurred_at=datetime.now(timezone.utc),
            metadata={"bundle_version": "1.0"},
        )
    )

    # Seal the chain
    build_hash_chain(audit_events)

    manifest = GovernanceBundleManifest(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        num_principals=len(principals),
        num_acls=len(acls),
        num_audit_events=len(audit_events),
        chain_head_hash=audit_events[-1].event_hash if audit_events else None,
        notes=notes,
    )

    logger.info(
        "Exported governance bundle: %d principals, %d ACLs, %d audit events",
        len(principals),
        len(acls),
        len(audit_events),
    )

    return GovernanceBundle(
        manifest=manifest,
        principals=principals,
        acls=acls,
        audit_trail=audit_events,
    )
