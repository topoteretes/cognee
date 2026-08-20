"""Text-to-SQL: natural-language question → guarded SELECT → rows.

Ports the generate → execute → feed-errors-back retry loop from
``NaturalLanguageRetriever`` (NL→Cypher) to SQL, with two deliberate
differences: an empty result set counts as success (no wasted LLM retries),
and exhausted attempts return an explicit error result instead of a silent
empty list.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.tools.config import get_tools_config
from cognee.modules.tools.connections import get_tool_connection
from cognee.modules.tools.errors import SqlGuardError, ToolCallsDisabledError
from cognee.modules.tools.text_to_sql.engine_factory import (
    get_external_sql_engine,
    introspect_schema,
)
from cognee.modules.tools.text_to_sql.executor import execute_readonly
from cognee.modules.tools.text_to_sql.sql_guard import ensure_limit, validate_select
from cognee.shared.logging_utils import get_logger

logger = get_logger("text_to_sql")

TOOL_NAME = "text_to_sql"

_SYSTEM_PROMPT_PATH = "text_to_sql_system.txt"


@dataclass
class TextToSqlResult:
    connection: str
    dialect: str
    question: str
    success: bool
    sql: Optional[str] = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    attempts: int = 0
    error: Optional[str] = None

    def render_text(self) -> str:
        """Compact human-readable rendering for the recall entry text."""
        if not self.success:
            return f"[{self.connection}] text-to-SQL failed: {self.error}"
        if not self.rows:
            return f"[{self.connection}] {self.sql} → no rows"
        columns = list(self.rows[0].keys())
        lines = [" | ".join(columns)]
        for row in self.rows:
            lines.append(" | ".join(str(row.get(column, "")) for column in columns))
        suffix = " (truncated)" if self.truncated else ""
        return f"[{self.connection}] {self.sql} → {self.row_count} row(s){suffix}\n" + "\n".join(
            lines
        )

    def structured(self) -> dict[str, Any]:
        return {
            "connection": self.connection,
            "dialect": self.dialect,
            "sql": self.sql,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "attempts": self.attempts,
        }


async def _generate_sql(
    question: str, schema: dict, dialect: str, max_rows: int, previous_attempts: str
) -> str:
    system_prompt = render_prompt(
        _SYSTEM_PROMPT_PATH,
        context={
            "schema": schema,
            "dialect": dialect,
            "max_rows": max_rows,
            "previous_attempts": previous_attempts or "No attempts yet",
        },
    )
    return await LLMGateway.acreate_structured_output(
        text_input=question,
        system_prompt=system_prompt,
        response_model=str,
    )


async def run_text_to_sql(
    user_id: UUID,
    connection_name: str,
    question: str,
) -> TextToSqlResult:
    """Answer ``question`` with a read-only SQL query on an authorized database.

    Raises for authorization/configuration failures (feature gate off,
    unknown connection); execution and generation failures are reported in
    the returned result so one dead database never aborts a multi-source
    recall.
    """
    config = get_tools_config()
    if not config.tool_calls_enabled:
        raise ToolCallsDisabledError(
            "Tool calls are disabled. Set TOOL_CALLS_ENABLED=true to allow recall "
            "to query authorized external databases."
        )

    connection = await get_tool_connection(user_id, connection_name)
    max_rows = connection.get("max_rows") or config.text_to_sql_max_rows

    engine = get_external_sql_engine(
        connection["connection_string"],
        connection["provider"],
        config.text_to_sql_statement_timeout_ms,
    )
    dialect = engine.dialect.name

    result = TextToSqlResult(
        connection=connection_name, dialect=dialect, question=question, success=False
    )

    try:
        schema = await introspect_schema(
            engine,
            allowed_tables=connection.get("allowed_tables"),
            max_tables=config.text_to_sql_max_schema_tables,
        )
    except Exception as error:
        logger.error("Schema introspection failed for '%s': %s", connection_name, error)
        result.error = f"Could not read the database schema: {error}"
        return result

    if not schema:
        result.error = "The database exposes no readable tables."
        return result

    previous_attempts = ""
    for attempt in range(1, config.text_to_sql_max_attempts + 1):
        result.attempts = attempt
        generated = ""
        try:
            generated = await _generate_sql(question, schema, dialect, max_rows, previous_attempts)
            sql = ensure_limit(validate_select(generated), max_rows)
        except SqlGuardError as error:
            previous_attempts += f"Query: {generated!r} -> Rejected: {error}\n"
            logger.warning("SQL guard rejected attempt %d: %s", attempt, error)
            result.error = str(error)
            continue
        except Exception as error:
            previous_attempts += f"Query: <not generated> -> Error: {error}\n"
            logger.error("SQL generation failed on attempt %d: %s", attempt, error)
            result.error = f"SQL generation failed: {error}"
            continue

        result.sql = sql
        try:
            rows, truncated = await execute_readonly(engine, sql, max_rows)
        except Exception as error:
            previous_attempts += f"Query: {sql} -> Executed with error: {error}\n"
            logger.warning("SQL execution failed on attempt %d: %s", attempt, error)
            result.error = f"SQL execution failed: {error}"
            continue

        # An empty result set is a valid answer — don't burn retries on it.
        result.success = True
        result.error = None
        result.rows = rows
        result.row_count = len(rows)
        result.truncated = truncated
        return result

    logger.warning(
        "text_to_sql exhausted %d attempts for connection '%s'",
        config.text_to_sql_max_attempts,
        connection_name,
    )
    return result
