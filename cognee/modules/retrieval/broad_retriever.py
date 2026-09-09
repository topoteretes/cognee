"""BROAD: wide retrieval over row-shaped data with a bounded prompt (SDK-324).

Row-shaped sources (CSV and anything else ingested through DLT) land one
``DltRow`` node per row in its own ``DltRow_text`` collection — document chunk
search is documents-only and deliberately excludes them. Rows are small, so the
retrieval budget that suits a document chunk is far too narrow here: the
default ``top_k`` of 15 answers "who has the most issues assigned?" from 15 of
140 rows and returns a confident, plausible, wrong name. Nothing errors, so the
caller cannot tell a truncated answer from a correct one.

BROAD changes two things and nothing else:

* it retrieves with its own wide default (``BROAD_DEFAULT_TOP_K``) instead of
  the global 15, because a row costs a fraction of a chunk; and
* it bounds what reaches the prompt by a CHARACTER budget rather than a row
  count, so widening retrieval cannot grow the completion input without limit.

A character budget, not a row cap, is what "bounded context" has to mean here:
rows vary in width by an order of magnitude between sources, so any fixed count
is either wasteful on narrow rows or over budget on wide ones.

Widening alone is NOT enough, and this was measured rather than assumed. With
all 140 rows of the SDK-324 CSV in context the model still answered "Akshats-git
and koopatroopa787, 5 each" — the right name, an invented tie, because
koopatroopa787 has 4. An LLM asked to count 140 rows miscounts whether or not
the set was truncated. So BROAD counts in Python: it parses the row fields,
computes exact per-column value frequencies over the FULL retrieved set, and
puts that table in front of the evidence. The model reads a number instead of
deriving one. That is the "defined notion of what is being counted" the issue
asks for: per-column value frequency over every row retrieved.

Rows repeat their table's schema header verbatim in every row's text (see
``_build_schema_context_text``). Emitting it once and stripping it from the
remaining rows is what makes the budget go far enough to hold a whole small
table — on the SDK-324 CSV it is the difference between ~40 and all 140 rows.
"""

import re
from collections import Counter
from typing import Any

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")

# Retrieval budget. Rows are small, so this buys breadth cheaply; the prompt is
# bounded separately by BROAD_DEFAULT_CONTEXT_CHARS, so a wide value here can
# never on its own blow up the completion input.
BROAD_DEFAULT_TOP_K = 500

# Ceiling on the characters formatted into the completion prompt. ~60k chars is
# roughly 15k tokens — comfortably inside current context windows while holding
# a few hundred typical rows.
BROAD_DEFAULT_CONTEXT_CHARS = 60_000

# Row text is written as "<schema header>\n\nRow Data:\n<fields>". Rows from one
# table repeat the header verbatim, so it is emitted once and elided after that.
_ROW_DATA_MARKER = "Row Data:"

# Row-shaped collections first; document chunks are the fallback so BROAD still
# answers on corpora ingested through the plain (non-DLT) cognify flow.
BROAD_ROW_COLLECTION = "DltRow_text"
BROAD_FALLBACK_COLLECTION = "DocumentChunk_text"


def _split_schema_header(text: str) -> tuple[str, str]:
    """Split row text into (schema header, row body).

    Returns ``("", text)`` when the text carries no recognisable header, which
    is the case for ordinary document chunks on the fallback path.
    """
    marker_at = text.find(_ROW_DATA_MARKER)
    if marker_at == -1:
        return "", text
    return text[:marker_at].rstrip(), text[marker_at:]


# --- deterministic aggregation ---------------------------------------------
# A row line is exactly two spaces, a column name, a colon and the value (see
# _build_schema_context_text). Anything else is a continuation of the previous
# value — free-text columns such as `notes` legitimately contain newlines.
_ROW_FIELD = re.compile(r"^  (\w+): ?(.*)$")

# Values longer than this on average mark a free-text column (notes, comments).
# Counting them produces one bucket per row and tells the reader nothing.
_MAX_CATEGORICAL_VALUE_LEN = 80

# Per column, how many of the most common values to show.
_TOP_VALUES_PER_COLUMN = 10

# A column whose values are this close to all-distinct is an identifier
# (issue_number, uuid, primary key). Its frequency table is noise: it costs
# context budget and can only mislead a ranking question.
_IDENTIFIER_DISTINCT_RATIO = 0.9

# Values that mean "no value" in a DLT row rendering.
_EMPTY_VALUES = {"", "None", "null", "NULL", "nan"}


def parse_row_fields(text: str) -> dict[str, str]:
    """Extract ``{column: value}`` from a row's ``Row Data:`` block."""
    marker_at = text.find(_ROW_DATA_MARKER)
    if marker_at == -1:
        return {}
    fields: dict[str, str] = {}
    last_key: str | None = None
    for line in text[marker_at + len(_ROW_DATA_MARKER) :].splitlines():
        match = _ROW_FIELD.match(line)
        if match:
            last_key = match.group(1)
            fields[last_key] = match.group(2).strip()
        elif last_key is not None and line.strip():
            # Continuation of a multi-line value.
            fields[last_key] = f"{fields[last_key]} {line.strip()}"
    return fields


def build_aggregate_block(retrieved_objects: Any) -> str:
    """Exact per-column value frequencies over every retrieved row.

    Returns "" when the retrieved objects are not rows (the document-chunk
    fallback), so BROAD degrades to plain wide RAG rather than emitting a
    meaningless table.
    """
    parsed = []
    for found in retrieved_objects:
        payload = getattr(found, "payload", None) or {}
        fields = parse_row_fields(payload.get("text") or "")
        if fields:
            parsed.append(fields)
    if not parsed:
        return ""

    columns: dict[str, Counter] = {}
    lengths: dict[str, list[int]] = {}
    for fields in parsed:
        for key, value in fields.items():
            columns.setdefault(key, Counter())[value] += 1
            lengths.setdefault(key, []).append(len(value))

    sections: list[str] = []
    for column, counter in columns.items():
        mean_len = sum(lengths[column]) / len(lengths[column])
        if mean_len > _MAX_CATEGORICAL_VALUE_LEN:
            continue  # free text, not a category
        populated = Counter(
            {value: n for value, n in counter.items() if value not in _EMPTY_VALUES}
        )
        if not populated or max(populated.values()) < 2:
            continue  # every value unique — an identifier, nothing to count
        if len(populated) / sum(populated.values()) > _IDENTIFIER_DISTINCT_RATIO:
            continue  # near-unique: an identifier column, not a category
        empty = sum(n for value, n in counter.items() if value in _EMPTY_VALUES)
        head = "\n".join(
            f"    {value}: {n}" for value, n in populated.most_common(_TOP_VALUES_PER_COLUMN)
        )
        more = len(populated) - _TOP_VALUES_PER_COLUMN
        tail = f"\n    (+{more} more distinct values, all with lower counts)" if more > 0 else ""
        sections.append(
            f'  Column "{column}" — {sum(populated.values())} non-empty values, '
            f"{len(populated)} distinct"
            + (f", {empty} empty" if empty else "")
            + f":\n{head}{tail}"
        )

    if not sections:
        return ""

    body = "\n".join(sections)
    return (
        f"EXACT COUNTS computed over all {len(parsed)} retrieved rows.\n"
        "These were counted programmatically, not estimated. For any question about "
        "which value occurs most, how many, or ranking by frequency, use these numbers "
        "directly and do not recount the rows below.\n"
        f"{body}\n"
    )


class BroadRetriever(CompletionRetriever):
    """Wide row retrieval with a character-bounded completion context."""

    def __init__(
        self,
        *args: Any,
        context_max_chars: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("top_k", BROAD_DEFAULT_TOP_K)
        super().__init__(*args, **kwargs)
        self.context_max_chars = (
            context_max_chars if context_max_chars is not None else BROAD_DEFAULT_CONTEXT_CHARS
        )

    async def get_retrieved_objects(self, query: str) -> Any:
        """Search the row collection, falling back to document chunks.

        The fallback is not a silent degradation: a corpus ingested through the
        plain cognify flow has no rows at all, and answering it from chunks is
        the only sensible reading of a BROAD query there.
        """
        vector_engine = await get_vector_engine_async()

        for collection in (BROAD_ROW_COLLECTION, BROAD_FALLBACK_COLLECTION):
            try:
                found = await vector_engine.search(
                    collection,
                    query,
                    limit=self.top_k,
                    include_payload=True,
                    node_name=self.node_name,
                    node_name_filter_operator=self.node_name_filter_operator,
                )
            except CollectionNotFoundError:
                logger.debug("BROAD: collection %s not present, trying next", collection)
                continue
            if found:
                logger.debug("BROAD: %s returned %d rows", collection, len(found))
                return found

        raise NoDataError("No data found in the system, please add data first.")

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """Format rows into a context bounded by ``context_max_chars``.

        The schema header is emitted once; subsequent rows contribute only their
        own data. When the budget runs out the context says so explicitly, so a
        truncated answer is visible in the context rather than silently wrong.
        """
        if not retrieved_objects:
            return ""

        # Counted first, and over the FULL retrieved set — the aggregate must
        # not depend on how many rows survive the character budget below.
        aggregate = build_aggregate_block(retrieved_objects)

        parts: list[str] = [aggregate] if aggregate else []
        used = len(aggregate)
        header_emitted = ""
        included = 0

        for found in retrieved_objects:
            payload = getattr(found, "payload", None) or {}
            text = payload.get("text") or ""
            if not text:
                continue

            header, body = _split_schema_header(text)
            piece = body
            if header and header != header_emitted:
                # First row of a table (or a new table in a multi-table source):
                # carry its header so the LLM can read the column names.
                piece = f"{header}\n{body}"
                header_emitted = header

            if used + len(piece) > self.context_max_chars:
                break
            parts.append(piece)
            used += len(piece)
            included += 1

        omitted = len(retrieved_objects) - included
        if omitted > 0:
            # Say it in the context: an aggregate computed over a truncated set
            # is wrong, and the model should be able to see that it happened.
            parts.append(
                f"\n[NOTE: context truncated to {included} of {len(retrieved_objects)} "
                f"retrieved records; {omitted} were omitted. Any count or ranking "
                f"below is computed over the shown records only. The EXACT COUNTS "
                f"block above still covers all {len(retrieved_objects)} retrieved rows.]"
            )
            logger.info(
                "BROAD context truncated: %d/%d records (%d chars budget)",
                included,
                len(retrieved_objects),
                self.context_max_chars,
            )

        return "\n".join(parts)
