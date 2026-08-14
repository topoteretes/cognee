"""SDK surface for managing authorized tool connections.

Exposed as ``cognee.tools``:

    await cognee.tools.register_sql_connection(
        name="analytics",
        connection_string="postgresql://ro_user:pw@host:5432/analytics",
    )
    await cognee.tools.list_sql_connections()
    await cognee.tools.remove_sql_connection("analytics")

Connections registered here become usable via
``cognee.recall(query, scope=["tools"])`` once ``TOOL_CALLS_ENABLED=true``.
"""

from typing import Any, Optional


async def _resolve_user_id(user: Optional[object]):
    from cognee.modules.users.methods import get_default_user

    if user is None:
        user = await get_default_user()
    return user.id


async def register_sql_connection(
    name: str,
    connection_string: str,
    *,
    provider: Optional[str] = None,
    allowed_tables: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
    description: Optional[str] = None,
    allow_writes: bool = False,
    user: Optional[object] = None,
) -> dict[str, Any]:
    """Register (or replace) an authorized external SQL database for the user.

    The connection string is encrypted at rest and never returned by any
    listing or recall response. Use a database role that only has SELECT
    grants — cognee additionally enforces read-only execution, but the
    role is the final safety layer.

    Set ``allow_writes=True`` (with a role that has UPDATE grants) to opt
    this connection into approval-gated write-back: correction proposals
    drafted via :func:`propose_sql_write` and applied only by an explicit
    :func:`apply_write_proposal` call.
    """
    from cognee.modules.tools.connections import register_tool_connection

    user_id = await _resolve_user_id(user)
    return await register_tool_connection(
        user_id,
        name,
        connection_string,
        provider=provider,
        allowed_tables=allowed_tables,
        max_rows=max_rows,
        description=description,
        allow_writes=allow_writes,
    )


async def list_sql_connections(user: Optional[object] = None) -> list[dict[str, Any]]:
    """The user's authorized connections (plus deployment-level ones), without secrets."""
    from cognee.modules.tools.connections import list_tool_connections

    user_id = await _resolve_user_id(user)
    return await list_tool_connections(user_id)


async def remove_sql_connection(name: str, user: Optional[object] = None) -> bool:
    """Remove an authorized connection. Returns False when nothing matched."""
    from cognee.modules.tools.connections import delete_tool_connection

    user_id = await _resolve_user_id(user)
    return await delete_tool_connection(user_id, name)


async def propose_sql_write(
    connection: str,
    instruction: str,
    *,
    evidence: Optional[dict[str, Any]] = None,
    user: Optional[object] = None,
) -> dict[str, Any]:
    """Draft a correction UPDATE as a reviewable proposal — nothing executes.

    Requires ``TOOL_WRITE_CALLS_ENABLED=true`` and a connection registered
    with ``allow_writes=True``. The returned proposal carries the drafted
    SQL and a dry-run affected-row estimate; apply it after review with
    :func:`apply_write_proposal`.
    """
    from cognee.modules.tools.text_to_sql import propose_sql_write as _propose

    user_id = await _resolve_user_id(user)
    return await _propose(user_id, connection, instruction, evidence=evidence)


async def propose_corrections(
    connection: str,
    *,
    limit: int = 10,
    user: Optional[object] = None,
) -> list[dict[str, Any]]:
    """Draft correction proposals from contradictions cognee has detected.

    Scans the graph's ``contradicts`` edges (recorded by the opt-in
    contradiction-detection cognify task) and drafts one reviewable UPDATE
    proposal per contradiction that is correctable in the given database.
    """
    from cognee.modules.tools.text_to_sql import propose_corrections_from_contradictions

    user_id = await _resolve_user_id(user)
    return await propose_corrections_from_contradictions(user_id, connection, limit=limit)


async def list_write_proposals(
    status: Optional[str] = None, user: Optional[object] = None
) -> list[dict[str, Any]]:
    """The user's write proposals, optionally filtered by status."""
    from cognee.modules.tools.text_to_sql import list_write_proposals as _list

    user_id = await _resolve_user_id(user)
    return await _list(user_id, status=status)


async def apply_write_proposal(proposal_id, user: Optional[object] = None) -> dict[str, Any]:
    """Execute a reviewed proposal against its source database.

    This is the ONLY path that commits a write. Rolls back (and marks the
    proposal failed) when the affected rowcount exceeds the configured cap
    or no longer matches the dry-run estimate.
    """
    from uuid import UUID

    from cognee.modules.tools.text_to_sql import apply_write_proposal as _apply

    user_id = await _resolve_user_id(user)
    if not isinstance(proposal_id, UUID):
        proposal_id = UUID(str(proposal_id))
    return await _apply(user_id, proposal_id)


async def reject_write_proposal(proposal_id, user: Optional[object] = None) -> dict[str, Any]:
    """Mark a proposal rejected so it can never be applied."""
    from uuid import UUID

    from cognee.modules.tools.text_to_sql import reject_write_proposal as _reject

    user_id = await _resolve_user_id(user)
    if not isinstance(proposal_id, UUID):
        proposal_id = UUID(str(proposal_id))
    return await _reject(user_id, proposal_id)
