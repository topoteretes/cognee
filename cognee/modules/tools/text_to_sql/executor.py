"""Rollback-only execution of guarded SQL against an external database.

Never commits: the transaction is explicitly marked read-only on Postgres
and always rolled back, so even a statement that somehow passed the SQL
guard cannot persist a write.
"""

import base64
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return str(value)


async def execute_readonly(
    engine: AsyncEngine, sql: str, max_rows: int
) -> tuple[list[dict[str, Any]], bool]:
    """Execute one SELECT and return (rows, truncated).

    Fetches at most ``max_rows + 1`` rows — the extra row only signals
    truncation and is dropped. Rows are coerced JSON-safe so they can travel
    through the recall response union unchanged.
    """
    async with engine.connect() as connection:
        if connection.dialect.name == "postgresql":
            # Belt and braces on top of default_transaction_read_only=on in
            # the connection's server settings.
            await connection.execute(text("SET TRANSACTION READ ONLY"))

        result = await connection.execute(text(sql))
        raw_rows = result.mappings().fetchmany(int(max_rows) + 1)

        truncated = len(raw_rows) > max_rows
        rows = [
            {key: _json_safe(value) for key, value in row.items()} for row in raw_rows[:max_rows]
        ]

        await connection.rollback()

    return rows, truncated
