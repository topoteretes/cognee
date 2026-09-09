"""BROAD search type: wide row retrieval, exact counts, bounded prompt (SDK-324).

The end-to-end behaviour these protect was measured against the issue's CSV:
140 rows, ground truth "Akshats-git with 5". Before the fix the answer was a
confident wrong name; widening retrieval alone still produced an invented tie,
because an LLM miscounts 140 rows. Hence the exact-count block.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.retrieval.broad_retriever import (
    _ROW_INDEX_KEY,
    BROAD_DEFAULT_TOP_K,
    BroadRetriever,
    build_aggregate_block,
    parse_row_fields,
    parse_rows,
)
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    DEFAULT_TOP_K,
    resolve_top_k,
)
from cognee.modules.search.types import SearchType


def _row(**fields) -> SimpleNamespace:
    """A DltRow-shaped ScoredResult, in the exact text format cognee writes."""
    header = "Table: t\nColumns:\n" + "".join(f"  - {k} (TEXT)\n" for k in fields)
    body = "Row Data:\n" + "".join(f"  {k}: {v}\n" for k, v in fields.items())
    return SimpleNamespace(payload={"text": f"{header}\n{body}"})


# --- per-search-type retrieval budget --------------------------------------


def test_broad_gets_its_own_wide_default():
    assert resolve_top_k(SearchType.BROAD, None) == BROAD_DEFAULT_TOP_K
    assert BROAD_DEFAULT_TOP_K > DEFAULT_TOP_K


def test_other_search_types_keep_the_global_default():
    """AC: existing callers see no behaviour change."""
    for search_type in (
        SearchType.GRAPH_COMPLETION,
        SearchType.RAG_COMPLETION,
        SearchType.CHUNKS,
        SearchType.SUMMARIES,
        SearchType.TRIPLET_COMPLETION,
    ):
        assert resolve_top_k(search_type, None) == DEFAULT_TOP_K == 15


def test_hybrid_completion_keeps_its_none():
    """HYBRID_COMPLETION splits one budget across a chunk lane and an entity
    lane and owns per-lane defaults. Resolving None to a number here silently
    overrides them — it set chunks_top_k to 10 instead of 5."""
    assert resolve_top_k(SearchType.HYBRID_COMPLETION, None) is None
    assert resolve_top_k(SearchType.HYBRID_COMPLETION, 20) == 20


def test_explicit_top_k_overrides_the_per_type_default():
    assert resolve_top_k(SearchType.BROAD, 7) == 7
    assert resolve_top_k(SearchType.GRAPH_COMPLETION, 300) == 300


# --- row parsing ------------------------------------------------------------


def test_parse_row_fields_reads_the_row_data_block():
    fields = parse_row_fields(_row(issue="3601", assignee="kr3shna").payload["text"])
    assert fields == {"issue": "3601", "assignee": "kr3shna"}


def test_parse_row_fields_keeps_multi_line_values_whole():
    """Free-text columns contain newlines; a naive line split would drop them
    and, worse, invent columns out of the continuation lines."""
    text = "Row Data:\n  assignee: ada\n  notes: first line\nsecond line\n  state: open\n"
    fields = parse_row_fields(text)
    assert fields["notes"] == "first line second line"
    assert fields["state"] == "open"
    assert set(fields) == {"assignee", "notes", "state"}


def test_parse_row_fields_ignores_text_without_a_row_block():
    assert parse_row_fields("just a document chunk") == {}


# --- exact counting ---------------------------------------------------------


def test_counts_are_exact_over_every_row():
    rows = [_row(assignee="ada") for _ in range(5)]
    rows += [_row(assignee="grace") for _ in range(4)]
    block = build_aggregate_block(rows)
    assert "ada: 5" in block
    assert "grace: 4" in block
    assert "9 retrieved rows" in block


def test_empty_values_are_reported_but_not_counted_as_a_winner():
    rows = [_row(assignee="ada")] * 2 + [_row(assignee="None")] * 6
    block = build_aggregate_block(rows)
    assert "ada: 2" in block
    assert "6 empty" in block
    assert "None: 6" not in block


def test_identifier_columns_are_skipped():
    """A near-unique column is an id; its frequency table is noise that costs
    budget and can only mislead a ranking question."""
    rows = [_row(issue_number=str(i), state="open") for i in range(50)]
    block = build_aggregate_block(rows)
    assert "issue_number" not in block
    assert "state" in block


def test_free_text_columns_are_skipped():
    rows = [_row(notes="x" * 200, state="open") for _ in range(10)]
    block = build_aggregate_block(rows)
    assert "notes" not in block
    assert "state" in block


def test_no_block_for_non_row_objects():
    """Document chunks have no row structure — BROAD must degrade to plain wide
    RAG rather than emit a meaningless table."""
    chunks = [SimpleNamespace(payload={"text": "an ordinary paragraph"})] * 3
    assert build_aggregate_block(chunks) == ""


# --- bounded prompt context -------------------------------------------------


@pytest.mark.asyncio
async def test_context_is_bounded_by_the_character_budget():
    """AC: prompt context stays bounded regardless of how wide retrieval goes."""
    retriever = BroadRetriever(context_max_chars=2000)
    rows = [_row(assignee=f"user{i}", filler="y" * 400) for i in range(200)]

    context = await retriever.get_context_from_objects("q", rows)

    assert len(context) < 2000 * 2  # budget plus the aggregate block and note
    assert "context truncated" in context


@pytest.mark.asyncio
async def test_counts_cover_every_row_even_when_evidence_is_truncated():
    """The aggregate is computed over the full retrieved set, so truncating the
    evidence shown to the LLM must not change the numbers."""
    retriever = BroadRetriever(context_max_chars=1500)
    rows = [_row(assignee="ada", filler="y" * 300) for _ in range(30)]
    rows += [_row(assignee="grace", filler="y" * 300) for _ in range(10)]

    context = await retriever.get_context_from_objects("q", rows)

    assert "ada: 30" in context
    assert "grace: 10" in context
    assert "40 retrieved rows" in context
    assert "context truncated" in context


@pytest.mark.asyncio
async def test_schema_header_is_emitted_once():
    """Rows repeat their table header verbatim; emitting it per row is what
    pushes a small table over the budget."""
    retriever = BroadRetriever()
    rows = [_row(assignee=f"user{i}") for i in range(10)]

    context = await retriever.get_context_from_objects("q", rows)

    assert context.count("Table: t") == 1


@pytest.mark.asyncio
async def test_empty_retrieval_gives_empty_context():
    assert await BroadRetriever().get_context_from_objects("q", []) == ""


def test_broad_retriever_defaults_to_the_wide_budget():
    assert BroadRetriever().top_k == BROAD_DEFAULT_TOP_K
    assert BroadRetriever(top_k=25).top_k == 25


# --- the plain (non-DLT) cognify route --------------------------------------
# csv_loader renders rows differently from DltRow: "Row 7:" then one line of
# comma-joined pairs, and a document chunk holds MANY rows rather than one.


def _plain_chunk(*rows) -> SimpleNamespace:
    """A DocumentChunk in csv_loader's rendering. `rows` are (index, fields)."""
    blocks = []
    for index, fields in rows:
        pairs = ", ".join(f"{k}: {v}" for k, v in fields.items())
        blocks.append(f"Row {index}:\n{pairs}\n")
    return SimpleNamespace(payload={"text": "\n".join(blocks)})


def test_parses_many_rows_out_of_one_chunk():
    chunk = _plain_chunk(
        (1, {"assignee": "ada", "state": "open"}),
        (2, {"assignee": "grace", "state": "open"}),
    )
    rows = parse_rows(chunk.payload["text"])
    assert [r["assignee"] for r in rows] == ["ada", "grace"]


def test_plain_values_may_contain_commas():
    """csv_loader joins pairs with ', ' and free-text values contain commas —
    splitting naively invents columns out of prose."""
    chunk = _plain_chunk(
        (1, {"assignee": "ada", "notes": "broke, then fixed, then shipped", "state": "open"}),
    )
    fields = parse_rows(chunk.payload["text"])[0]
    assert fields["notes"] == "broke, then fixed, then shipped"
    # parse_rows carries the source row index alongside the columns so the
    # caller can detect boundary loss; build_aggregate_block strips it.
    assert set(fields) - {_ROW_INDEX_KEY} == {"assignee", "notes", "state"}


def test_counts_the_plain_route_when_no_row_was_lost():
    chunks = [
        _plain_chunk((1, {"assignee": "ada"}), (2, {"assignee": "ada"})),
        _plain_chunk((3, {"assignee": "grace"})),
    ]
    block = build_aggregate_block(chunks)
    assert "ada: 2" in block
    assert "EXACT COUNTS" in block


def test_no_counts_when_chunking_cut_a_row():
    """Chunk boundaries do not respect rows, so the document route can lose one.
    A tally that is short by an unknown amount is worse than none: the model
    trusts a labelled number over its own reading of the rows. Exact or nothing.
    """
    chunks = [
        _plain_chunk((1, {"assignee": "ada"})),
        _plain_chunk((3, {"assignee": "grace"})),  # row 2 was cut in half
    ]
    assert build_aggregate_block(chunks) == ""


def test_dlt_rows_always_count_even_though_they_carry_no_index():
    """A DltRow record IS one whole row, so nothing can be lost mid-row and the
    missing-row check must not suppress the block for it."""
    rows = [_row(assignee="ada") for _ in range(3)]
    assert "EXACT COUNTS" in build_aggregate_block(rows)


def test_inconsistent_column_sets_are_refused():
    """If the pair split went wrong the recovered columns disagree row to row;
    counting that would be quietly incorrect."""
    chunks = [_plain_chunk((i, {f"col{i}": "v", f"other{i}": "w"})) for i in range(1, 11)]
    assert build_aggregate_block(chunks) == ""


def test_row_index_never_leaks_into_the_counts():
    chunks = [_plain_chunk((1, {"assignee": "ada"})), _plain_chunk((2, {"assignee": "ada"}))]
    block = build_aggregate_block(chunks)
    assert "__row_index__" not in block


# --- a sample must never be presented as a total -----------------------------


def test_capped_retrieval_emits_no_counts():
    """When the search returns exactly top_k rows the store may hold more, so
    the rows are a similarity-ranked sample of the table.

    Measured on a 2000-row table with the default top_k of 500: the block read
    "EXACT COUNTS computed over all 500 retrieved rows" and the model answered
    321 where the truth was 676. Every word was true and the answer was still
    wrong by half.
    """
    rows = [_row(assignee="ada") for _ in range(10)]
    assert build_aggregate_block(rows, retrieval_capped=True) == ""
    assert "EXACT COUNTS" in build_aggregate_block(rows, retrieval_capped=False)


@pytest.mark.asyncio
async def test_context_has_no_counts_when_retrieval_was_capped():
    retriever = BroadRetriever(top_k=3)
    retriever._retrieval_capped = True
    rows = [_row(assignee="ada") for _ in range(3)]

    context = await retriever.get_context_from_objects("q", rows)

    assert "EXACT COUNTS" not in context
    assert "ada" in context  # the rows themselves are still shown
