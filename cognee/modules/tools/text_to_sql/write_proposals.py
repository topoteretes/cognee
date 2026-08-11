"""Approval-gated write-back: correction proposals for authorized databases.

When cognee determines stored data is inaccurate — a ``contradicts`` edge
from contradiction detection, or an explicit instruction — this module drafts
an UPDATE with the LLM, validates it (single UPDATE, mandatory WHERE, table
allowlist), dry-runs it inside a rolled-back transaction to capture the
affected-row count, and stores everything as a ``ToolWriteProposal``.

**Nothing touches the source database until an explicit apply call**, and the
apply re-verifies every gate, re-checks the rowcount against the dry-run
estimate and the ``text_to_sql_max_affected_rows`` cap, and rolls back on any
mismatch. This mirrors the skill-improvement proposal lifecycle: propose →
review → apply.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select, text

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.tools.config import get_tools_config
from cognee.modules.tools.connections import _ensure_table, get_tool_connection
from cognee.modules.tools.errors import (
    SqlGuardError,
    ToolCallsDisabledError,
    ToolConnectionNotFoundError,
    ToolError,
    ToolWriteNotAllowedError,
)
from cognee.modules.tools.models.ToolWriteProposal import ToolWriteProposal
from cognee.modules.tools.text_to_sql.engine_factory import (
    get_external_sql_engine,
    get_external_sql_write_engine,
    introspect_schema,
)
from cognee.modules.tools.text_to_sql.write_guard import validate_update
from cognee.shared.logging_utils import get_logger

logger = get_logger("text_to_sql_write")

_WRITE_PROMPT_PATH = "text_to_sql_write_system.txt"

# The model answers this verbatim when the correction cannot be expressed as
# a single UPDATE against the given schema.
NOT_APPLICABLE = "NOT_APPLICABLE"

STATUS_PROPOSED = "proposed"
STATUS_APPLIED = "applied"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"


def _check_write_gates(connection: dict[str, Any]) -> None:
    config = get_tools_config()
    if not (config.tool_calls_enabled and config.tool_write_calls_enabled):
        raise ToolCallsDisabledError(
            "Write-back is disabled. Set TOOL_CALLS_ENABLED=true and "
            "TOOL_WRITE_CALLS_ENABLED=true to allow correction proposals."
        )
    if not connection.get("allow_writes"):
        raise ToolWriteNotAllowedError(
            f"Connection '{connection['name']}' was not registered with allow_writes=True."
        )


def _proposal_to_public(row: ToolWriteProposal) -> dict[str, Any]:
    return {
        "proposal_id": str(row.id),
        "connection": row.connection_name,
        "status": row.status,
        "instruction": row.instruction,
        "evidence": row.evidence,
        "sql": row.sql,
        "target_table": row.target_table,
        "estimated_rows": row.estimated_rows,
        "applied_rows": row.applied_rows,
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def _dry_run_update(engine, sql: str) -> int:
    """Execute the UPDATE in a transaction, capture rowcount, ALWAYS roll back."""
    async with engine.connect() as connection:
        result = await connection.execute(text(sql))
        affected = result.rowcount
        await connection.rollback()
    return affected


async def _draft_update(
    connection: dict[str, Any], instruction: str, evidence: Optional[dict[str, Any]]
) -> Optional[tuple[str, str, int]]:
    """Draft, guard, and dry-run one UPDATE.

    Returns ``(sql, target_table, estimated_rows)``, or ``None`` when the
    model concludes the correction is not expressible on this schema.
    Retries with error feedback up to ``text_to_sql_max_attempts``.
    """
    config = get_tools_config()

    read_engine = get_external_sql_engine(
        connection["connection_string"],
        connection["provider"],
        config.text_to_sql_statement_timeout_ms,
    )
    schema = await introspect_schema(
        read_engine,
        allowed_tables=connection.get("allowed_tables"),
        max_tables=config.text_to_sql_max_schema_tables,
    )
    if not schema:
        raise ToolError("The database exposes no readable tables to correct.")

    write_engine = get_external_sql_write_engine(
        connection["connection_string"],
        connection["provider"],
        config.text_to_sql_statement_timeout_ms,
    )
    dialect = write_engine.dialect.name

    previous_attempts = ""
    last_error: Optional[Exception] = None
    for attempt in range(1, config.text_to_sql_max_attempts + 1):
        system_prompt = render_prompt(
            _WRITE_PROMPT_PATH,
            context={
                "schema": schema,
                "dialect": dialect,
                "evidence": evidence or {},
                "max_affected_rows": config.text_to_sql_max_affected_rows,
                "previous_attempts": previous_attempts or "No attempts yet",
            },
        )
        generated = await LLMGateway.acreate_structured_output(
            text_input=instruction,
            system_prompt=system_prompt,
            response_model=str,
        )

        if generated and generated.strip().upper() == NOT_APPLICABLE:
            return None

        try:
            sql, target_table = validate_update(
                generated, allowed_tables=connection.get("allowed_tables")
            )
        except SqlGuardError as error:
            previous_attempts += f"Query: {generated!r} -> Rejected: {error}\n"
            logger.warning("Write guard rejected attempt %d: %s", attempt, error)
            last_error = error
            continue

        try:
            estimated_rows = await _dry_run_update(write_engine, sql)
        except Exception as error:
            previous_attempts += f"Query: {sql} -> Executed with error: {error}\n"
            logger.warning("Write dry-run failed on attempt %d: %s", attempt, error)
            last_error = error
            continue

        return sql, target_table, estimated_rows

    raise ToolError(
        f"Could not draft a valid correction after "
        f"{config.text_to_sql_max_attempts} attempts: {last_error}"
    )


async def propose_sql_write(
    user_id: UUID,
    connection_name: str,
    instruction: str,
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Draft a correction UPDATE and store it as a reviewable proposal.

    The statement is dry-run (executed and rolled back) to capture the
    affected-row estimate, but no change is committed — apply it with
    :func:`apply_write_proposal` after review.
    """
    connection = await get_tool_connection(user_id, connection_name)
    _check_write_gates(connection)

    drafted = await _draft_update(connection, instruction, evidence)
    if drafted is None:
        raise ToolError(
            "The model concluded this correction cannot be expressed as a single "
            "UPDATE on this database's schema."
        )
    sql, target_table, estimated_rows = drafted

    await _ensure_table()
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = ToolWriteProposal(
            user_id=user_id,
            connection_name=connection_name,
            instruction=instruction,
            evidence=evidence,
            sql=sql,
            target_table=target_table,
            estimated_rows=estimated_rows,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

    logger.info(
        "Write proposal %s drafted for '%s' (%d estimated row(s)): %s",
        row.id,
        connection_name,
        estimated_rows,
        sql,
    )
    return _proposal_to_public(row)


async def propose_corrections_from_contradictions(
    user_id: UUID,
    connection_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Draft correction proposals from the graph's ``contradicts`` edges.

    Reads the contradictions recorded by ``detect_contradictions`` (each
    carries both fact texts, the reason, and a confidence), asks the LLM
    whether each is correctable in the given database, and stores a proposal
    for those that are. Contradictions the model marks NOT_APPLICABLE are
    skipped — with approval gating, a wrong mapping costs a rejected
    proposal, never a wrong write.
    """
    connection = await get_tool_connection(user_id, connection_name)
    _check_write_gates(connection)

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    _, edges = await graph_engine.get_graph_data()

    contradictions = []
    for _source_id, _target_id, relationship_name, properties in edges:
        if relationship_name != "contradicts":
            continue
        properties = properties or {}
        contradictions.append(
            {
                "type": "contradiction",
                "first_fact": properties.get("first_fact"),
                "second_fact": properties.get("second_fact"),
                "reason": properties.get("reason"),
                "confidence": properties.get("confidence"),
            }
        )
        if len(contradictions) >= limit:
            break

    proposals: list[dict[str, Any]] = []
    for evidence in contradictions:
        instruction = (
            f"Cognee detected a contradiction: '{evidence['first_fact']}' contradicts "
            f"'{evidence['second_fact']}' ({evidence['reason']}). If this database "
            "holds the inaccurate value, draft the correction."
        )
        try:
            drafted = await propose_sql_write(
                user_id, connection_name, instruction, evidence=evidence
            )
        except ToolError as error:
            logger.info("Skipped contradiction (not correctable here): %s", error)
            continue
        proposals.append(drafted)

    logger.info(
        "Drafted %d correction proposal(s) from %d contradiction(s) for '%s'",
        len(proposals),
        len(contradictions),
        connection_name,
    )
    return proposals


async def _get_owned_proposal(session, user_id: UUID, proposal_id) -> ToolWriteProposal:
    if not isinstance(proposal_id, UUID):
        proposal_id = UUID(str(proposal_id))
    result = await session.execute(
        select(ToolWriteProposal).where(
            ToolWriteProposal.id == proposal_id, ToolWriteProposal.user_id == user_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        # Fail closed: non-owned and non-existent look identical.
        raise ToolConnectionNotFoundError(f"No write proposal '{proposal_id}' for this user.")
    return row


async def list_write_proposals(user_id: UUID, status: Optional[str] = None) -> list[dict[str, Any]]:
    await _ensure_table()
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = select(ToolWriteProposal).where(ToolWriteProposal.user_id == user_id)
        if status:
            query = query.where(ToolWriteProposal.status == status)
        result = await session.execute(query.order_by(ToolWriteProposal.created_at.desc()))
        return [_proposal_to_public(row) for row in result.scalars().all()]


async def apply_write_proposal(user_id: UUID, proposal_id: UUID) -> dict[str, Any]:
    """Execute an approved proposal against the source database.

    Every gate is re-checked at apply time (they may have changed since the
    proposal was drafted). The UPDATE runs in a transaction and is rolled
    back — with the proposal marked ``failed`` — when the actual rowcount
    exceeds ``text_to_sql_max_affected_rows`` or no longer matches the
    dry-run estimate (the underlying data changed since review).
    """
    config = get_tools_config()

    await _ensure_table()
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = await _get_owned_proposal(session, user_id, proposal_id)
        if row.status != STATUS_PROPOSED:
            raise ToolError(f"Proposal is '{row.status}', only 'proposed' ones can be applied.")

        connection = await get_tool_connection(user_id, row.connection_name)
        _check_write_gates(connection)

        write_engine = get_external_sql_write_engine(
            connection["connection_string"],
            connection["provider"],
            config.text_to_sql_statement_timeout_ms,
        )

        error_message: Optional[str] = None
        affected = 0
        try:
            async with write_engine.connect() as db_connection:
                result = await db_connection.execute(text(row.sql))
                affected = result.rowcount

                if affected > config.text_to_sql_max_affected_rows:
                    await db_connection.rollback()
                    error_message = (
                        f"Rolled back: {affected} row(s) affected, cap is "
                        f"{config.text_to_sql_max_affected_rows}."
                    )
                elif row.estimated_rows is not None and affected != row.estimated_rows:
                    await db_connection.rollback()
                    error_message = (
                        f"Rolled back: {affected} row(s) affected but {row.estimated_rows} "
                        "were estimated at proposal time — the data changed since review. "
                        "Re-propose to get a fresh estimate."
                    )
                else:
                    await db_connection.commit()
        except Exception as error:
            error_message = f"Rolled back: execution failed: {error}"

        row.applied_rows = affected if error_message is None else None
        row.error = error_message
        row.status = STATUS_APPLIED if error_message is None else STATUS_FAILED
        row.resolved_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(row)

    if error_message is None:
        logger.info(
            "Applied write proposal %s on '%s': %d row(s) updated",
            proposal_id,
            row.connection_name,
            affected,
        )
    else:
        logger.warning("Write proposal %s failed: %s", proposal_id, error_message)
    return _proposal_to_public(row)


async def reject_write_proposal(user_id: UUID, proposal_id: UUID) -> dict[str, Any]:
    """Mark a proposal rejected; it can never be applied afterwards."""
    await _ensure_table()
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = await _get_owned_proposal(session, user_id, proposal_id)
        if row.status != STATUS_PROPOSED:
            raise ToolError(f"Proposal is '{row.status}', only 'proposed' ones can be rejected.")
        row.status = STATUS_REJECTED
        row.resolved_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(row)
        return _proposal_to_public(row)
