"""UPDATE-only validation for approval-gated write proposals.

Same philosophy as :mod:`sql_guard` (reject fast, feed errors back), but the
statement shape is inverted: exactly one UPDATE, a **mandatory WHERE
clause**, and an extractable target table so per-connection ``allowed_tables``
can be enforced. Everything else — DELETE, DDL, multi-statements — is still
forbidden; corrections in v1 are UPDATEs only.

Pure functions, no I/O.
"""

import re
from typing import Optional

from cognee.modules.tools.errors import SqlGuardError
from cognee.modules.tools.text_to_sql.sql_guard import _STRING_LITERAL_RE, strip_sql

# Keywords that can never appear in a correction UPDATE (scanned outside
# string literals, word-bounded). UPDATE/SET/WHERE are the statement itself.
_FORBIDDEN_WRITE_KEYWORDS = (
    "insert",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "attach",
    "detach",
    "vacuum",
    "pragma",
    "call",
    "merge",
    "execute",
    "exec",
    "replace",
    "reindex",
    "returning",
)

_UPDATE_TARGET_RE = re.compile(r'^\s*update\s+("?[\w.]+"?)\s+set\b', re.IGNORECASE)


def validate_update(sql: str, allowed_tables: Optional[list[str]] = None) -> tuple[str, str]:
    """Validate a single-table UPDATE with a WHERE clause.

    Returns ``(cleaned_sql, target_table)``. Raises :class:`SqlGuardError`
    with a retry-hint message otherwise.
    """
    cleaned = strip_sql(sql)
    if not cleaned:
        raise SqlGuardError("The generated SQL is empty.")

    scannable = _STRING_LITERAL_RE.sub("''", cleaned)

    if ";" in scannable:
        raise SqlGuardError("Only a single SQL statement is allowed (no ';' separators).")

    target_match = _UPDATE_TARGET_RE.match(scannable)
    if target_match is None:
        raise SqlGuardError(
            "Only a statement of the form 'UPDATE <table> SET ... WHERE ...' is allowed."
        )
    target_table = target_match.group(1).strip('"')

    if not re.search(r"\bwhere\b", scannable, re.IGNORECASE):
        raise SqlGuardError(
            "The UPDATE must have a WHERE clause — unbounded updates are never allowed."
        )

    for keyword in _FORBIDDEN_WRITE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", scannable, re.IGNORECASE):
            raise SqlGuardError(
                f"Forbidden keyword '{keyword.upper()}' found — a correction may only "
                "be a plain UPDATE with a WHERE clause."
            )

    if allowed_tables:
        allowed = {table.lower() for table in allowed_tables}
        if target_table.lower() not in allowed:
            raise SqlGuardError(
                f"Table '{target_table}' is not in this connection's allowed tables."
            )

    return cleaned, target_table
