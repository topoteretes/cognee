"""SELECT-only validation for LLM-generated SQL.

The guard is the first of three read-only layers (guard → read-only
connection + rollback-only executor → SELECT-only DB role). It exists to
reject bad statements fast and feed a useful error back into the retry loop —
the connection-level read-only enforcement is what actually stops anything
that slips through.

Pure functions, no I/O.
"""

import re

from cognee.modules.tools.errors import SqlGuardError

# Statement keywords that can never appear in a read-only query, matched on
# word boundaries after comments and string literals are stripped — so a
# column named "updated_at" passes while "; UPDATE ..." (or an UPDATE hidden
# in a CTE body) does not.
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
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
    "set",
    "reindex",
)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# Single-quoted SQL string literals, with '' as the escaped quote.
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


def strip_sql(sql: "str | None") -> str:
    """Remove markdown fences, comments, and outer whitespace/semicolon."""
    if sql is None:
        raise SqlGuardError("No SQL was generated.")
    cleaned = _FENCE_RE.sub("", sql.strip()).strip()
    cleaned = _BLOCK_COMMENT_RE.sub(" ", cleaned)
    cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def validate_select(sql: str) -> str:
    """Validate that ``sql`` is a single SELECT statement; return it cleaned.

    Raises :class:`SqlGuardError` with a message written to be fed back to
    the LLM as a retry hint.
    """
    cleaned = strip_sql(sql)
    if not cleaned:
        raise SqlGuardError("The generated SQL is empty.")

    scannable = _STRING_LITERAL_RE.sub("''", cleaned)

    if ";" in scannable:
        raise SqlGuardError("Only a single SQL statement is allowed (no ';' separators).")

    first_token_match = re.match(r"\s*(\w+)", scannable)
    first_token = first_token_match.group(1).lower() if first_token_match else ""
    if first_token not in ("select", "with"):
        raise SqlGuardError(
            f"Only SELECT statements are allowed; got a statement starting with "
            f"'{first_token.upper() or '?'}'."
        )

    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", scannable, re.IGNORECASE):
            raise SqlGuardError(
                f"Forbidden keyword '{keyword.upper()}' found — only read-only "
                "SELECT statements are allowed."
            )

    return cleaned


def ensure_limit(sql: str, max_rows: int) -> str:
    """Cap the result set: clamp an existing LIMIT or append one.

    Textual, not a full parser — LIMITs inside subqueries are clamped too,
    which errs on the side of returning fewer rows. The executor's
    fetch-at-most-``max_rows`` cap is the authoritative bound either way.
    """
    match = _LIMIT_RE.search(sql)
    if match is None:
        return f"{sql} LIMIT {int(max_rows)}"

    def _clamp(limit_match: re.Match) -> str:
        value = int(limit_match.group(1))
        return f"LIMIT {min(value, int(max_rows))}"

    return _LIMIT_RE.sub(_clamp, sql)
