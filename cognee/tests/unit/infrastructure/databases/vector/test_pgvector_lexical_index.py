"""The pgvector lexical (BM25) index helpers, without a Postgres.

``lexical_index.py`` is pure functions: statement builders, the passage /
doc_id mapping, the per-language tokenizer settings and the reference BM25
arithmetic. The adapter glue (``lexical_search`` short-circuits, result
mapping, the posting-table skip in ``remove_belongs_to_set_tags``) is checked
with a bare adapter instance and mocked sessions. The statements themselves
run against a live database in
``cognee/tests/integration/infrastructure/vector/test_pgvector_lexical_search.py``.
"""

import math
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.vector.lexical import (
    DEFAULT_STOP_WORDS,
    split_passages,
    tokenize_words,
)
from cognee.infrastructure.databases.vector.pgvector.lexical_index import (
    BM25_B,
    BM25_K1,
    MAX_PASSAGE_CHARS,
    PASSAGE_BATCH_SIZE,
    PassageRow,
    bm25_scores,
    bm25_term_score,
    build_passage_rows,
    clean_text,
    create_lexical_tables_statements,
    cut_passage,
    dedupe_owned_rows,
    delete_postings_statement,
    fit_identifier,
    insert_passages_statement,
    is_lexical_table_name,
    lexical_table_names,
    passage_doc_id,
    query_terms_statement,
    quote_identifier,
    search_statement,
    tokenizer_settings,
    upsert_owned_rows_statement,
)

ADAPTER_MODULE = "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"


# --- names ----------------------------------------------------------------------


def test_lexical_table_names_follow_the_collection():
    docs, terms = lexical_table_names("DocumentChunk_text")

    assert docs == "DocumentChunk_text__lexical_docs"
    assert terms == "DocumentChunk_text__lexical_terms"
    assert is_lexical_table_name(docs) and is_lexical_table_name(terms)
    assert not is_lexical_table_name("DocumentChunk_text")
    assert not is_lexical_table_name("Entity_name")


def test_fit_identifier_keeps_short_names_and_disambiguates_long_ones():
    assert fit_identifier("DocumentChunk_text__lexical_docs") == "DocumentChunk_text__lexical_docs"

    long_collection = "A" * 70 + "_text"
    docs, terms = lexical_table_names(long_collection)
    assert docs != terms, "Postgres would cut both to the same 63-byte prefix"
    assert len(docs.encode()) <= 63 and len(terms.encode()) <= 63
    assert lexical_table_names(long_collection) == (docs, terms), "deterministic"


def test_quote_identifier_escapes_embedded_quotes():
    assert quote_identifier("Entity_name") == '"Entity_name"'
    assert quote_identifier('odd"name') == '"odd""name"'


# --- tokenizer settings -------------------------------------------------------------


def test_tokenizer_settings_per_language():
    config, stop_words = tokenizer_settings("simple")
    assert config == "simple"
    assert stop_words == sorted(DEFAULT_STOP_WORDS)
    assert "the" in stop_words

    config, stop_words = tokenizer_settings("english")
    assert config == "english"
    assert stop_words == [], "Postgres's english config drops its own stop words"

    with pytest.raises(ValueError, match="Unsupported lexical search language"):
        tokenizer_settings("german")


# --- passages -----------------------------------------------------------------------


class _Point:
    def __init__(self, text, point_id=None):
        self.id = point_id or uuid4()
        self.text = text


def test_build_passage_rows_maps_each_passage_to_its_doc_id():
    text = "one two three four five six seven\n\neight nine ten"
    point = _Point(text)
    expected_passages = split_passages(text, max_tokens=4)
    assert len(expected_passages) > 1, "fixture must split into several passages"

    rows = build_passage_rows([point], "text", max_tokens=4)

    assert [row.text for row in rows] == expected_passages
    assert [row.passage_index for row in rows] == list(range(len(expected_passages)))
    assert all(row.row_id == str(point.id) for row in rows)
    assert [row.doc_id for row in rows] == [
        passage_doc_id(str(point.id), index) for index in range(len(expected_passages))
    ]
    assert rows[0].doc_id == f"{point.id}:0"


def test_build_passage_rows_skips_missing_and_blank_text():
    blank = _Point("   \n\n  ")
    missing = SimpleNamespace(id=uuid4())  # no ``text`` attribute at all
    kept = _Point("a passage")

    rows = build_passage_rows([blank, missing, kept], "text")

    assert [row.row_id for row in rows] == [str(kept.id)]


def test_nul_characters_are_dropped_from_indexed_text():
    """Postgres text cannot hold NUL; pypdf output and binary-ish files carry it."""
    point = SimpleNamespace(id=uuid4(), text="alpha\x00beta gamma")
    (row,) = build_passage_rows([point], "text")
    assert row.text == "alphabeta gamma"
    assert clean_text("q\x00uery") == "query"


def test_a_whitespace_free_run_is_cut_below_the_tsvector_limit():
    """An inline base64 image is one word to split_passages but megabytes long."""
    blob = "A" * (MAX_PASSAGE_CHARS * 2 + 10)
    point = SimpleNamespace(id=uuid4(), text=f"Intro text.\n\n![img](data:image/png;base64,{blob})")
    rows = build_passage_rows([point], "text")
    assert all(len(row.text) <= MAX_PASSAGE_CHARS for row in rows)
    assert [row.passage_index for row in rows] == list(range(len(rows)))
    assert "".join(row.text for row in rows).replace(" ", "").count("A") == len(blob)


def test_cut_passage_prefers_whitespace_and_keeps_short_passages_whole():
    assert cut_passage("short passage", max_chars=100) == ["short passage"]
    pieces = cut_passage("aaaa bbbb cccc dddd", max_chars=10)
    assert pieces == ["aaaa bbbb", "cccc dddd"]
    assert all(len(piece) <= 10 for piece in cut_passage("x" * 25, max_chars=10))


def test_dedupe_owned_rows_last_wins_and_unites_tags():
    row_id = str(uuid4())
    rows = [
        {"id": row_id, "payload": {"text": "old", "belongs_to_set": ["A"]}},
        {"id": str(uuid4()), "payload": {"text": "other", "belongs_to_set": []}},
        {"id": row_id, "payload": {"text": "new", "belongs_to_set": ["B", "A"]}},
    ]

    deduped = dedupe_owned_rows(rows)

    assert len(deduped) == 2
    merged = next(row for row in deduped if row["id"] == row_id)
    assert merged["payload"]["text"] == "new"
    assert merged["payload"]["belongs_to_set"] == ["A", "B"]
    assert rows[2]["payload"]["belongs_to_set"] == ["B", "A"], "inputs are not mutated"


# --- statements ---------------------------------------------------------------------


def test_create_statements_build_the_two_cascading_tables():
    statements = [str(statement) for statement in create_lexical_tables_statements("Doc_text")]

    assert len(statements) == 4
    docs, docs_index, terms, terms_index = statements
    assert docs.startswith('CREATE TABLE IF NOT EXISTS "Doc_text__lexical_docs" (')
    assert "doc_id TEXT PRIMARY KEY" in docs
    assert 'row_id UUID NOT NULL REFERENCES "Doc_text" (id) ON DELETE CASCADE' in docs
    assert "passage_index INTEGER NOT NULL" in docs and "doc_len INTEGER NOT NULL" in docs
    assert docs_index == (
        'CREATE INDEX IF NOT EXISTS "Doc_text__lexical_docs_row_id_idx" '
        'ON "Doc_text__lexical_docs" (row_id)'
    )
    assert terms.startswith('CREATE TABLE IF NOT EXISTS "Doc_text__lexical_terms" (')
    assert (
        'doc_id TEXT NOT NULL REFERENCES "Doc_text__lexical_docs" (doc_id) ON DELETE CASCADE'
        in (terms)
    )
    assert "PRIMARY KEY (term, doc_id)" in terms
    assert terms_index == (
        'CREATE INDEX IF NOT EXISTS "Doc_text__lexical_terms_doc_id_idx" '
        'ON "Doc_text__lexical_terms" (doc_id)'
    )


def test_statements_bind_user_text_instead_of_formatting_it():
    insert = str(insert_passages_statement("Doc_text"))
    for bind in (":doc_ids", ":row_ids", ":passage_indexes", ":texts", ":config", ":stop_words"):
        assert bind in insert
    assert "to_tsvector(CAST(CAST(:config AS text) AS regconfig), p.body)" in insert
    assert "cardinality(v.positions) AS tf" in insert
    assert "v.lexeme <> ALL(CAST(:stop_words AS text[]))" in insert
    assert 'INSERT INTO "Doc_text__lexical_docs" (doc_id, row_id, passage_index, doc_len)' in insert
    assert 'INSERT INTO "Doc_text__lexical_terms" (term, doc_id, tf)' in insert
    assert "CAST(COALESCE(l.doc_len, 0) AS integer)" in insert, "termless passages get doc_len 0"

    delete = str(delete_postings_statement("Doc_text"))
    assert delete == (
        'DELETE FROM "Doc_text__lexical_docs" '
        "WHERE row_id = ANY(CAST(CAST(:row_ids AS text[]) AS uuid[]))"
    )

    upsert = str(upsert_owned_rows_statement("Doc_text"))
    assert 'INSERT INTO "Doc_text" (id, payload, vector)' in upsert
    assert "CAST(r.payload AS json), NULL" in upsert, "owned rows carry no vector"
    assert "ON CONFLICT (id) DO UPDATE SET payload = jsonb_set(" in upsert
    assert "vector =" not in upsert.split("ON CONFLICT")[1], "a stored vector is never touched"
    assert "'{belongs_to_set}'" in upsert

    query_terms = str(query_terms_statement())
    for bind in (":config", ":query", ":stop_words"):
        assert bind in query_terms
    assert "SELECT DISTINCT v.lexeme" in query_terms, "one contribution per query term"

    search = str(search_statement("Doc_text"))
    for bind in (":terms", ":k1", ":b", ":limit"):
        assert bind in search
    assert "to_tsvector" not in search, "the terms arrive bound, so the term key drives the plan"
    assert search.count("t.term = ANY(CAST(:terms AS text[]))") == 2
    assert "ln(1 + (c.n - f.df + 0.5) / (f.df + 0.5))" in search
    assert "max(score)" in search, "a row keeps its best passage"
    assert 'JOIN "Doc_text" c ON c.id = r.row_id' in search
    assert search.rstrip().endswith(
        "ORDER BY r.score DESC, r.row_id ASC\n        LIMIT CAST(:limit AS integer)"
    )


def test_passage_batch_size_bounds_one_statement():
    assert PASSAGE_BATCH_SIZE == 500


# --- BM25 arithmetic ----------------------------------------------------------------


def test_bm25_term_score_matches_the_formula_written_out():
    tf, df, n, doc_len, avgdl = 2, 1, 4, 5, 4.0
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl))

    assert BM25_K1 == 1.2 and BM25_B == 0.75
    assert bm25_term_score(tf, df, n, doc_len, avgdl) == pytest.approx(expected)
    # No length normalization without an average length.
    assert bm25_term_score(1, 1, 1, 0, 0.0) == pytest.approx(
        math.log(1 + 0.5 / 1.5) * 2.2 / (1 + BM25_K1 * (1 - BM25_B))
    )


def _term_frequencies(passages: list[PassageRow]) -> dict[str, dict[str, int]]:
    corpus = {}
    for passage in passages:
        counts: dict[str, int] = {}
        for token in tokenize_words(passage.text, DEFAULT_STOP_WORDS):
            counts[token] = counts.get(token, 0) + 1
        corpus[passage.doc_id] = counts
    return corpus


def test_bm25_scores_rank_the_rare_term_first_on_the_fixture():
    """The fixture the live test indexes; the values are what the search statement
    returns for it on Postgres, so the SQL and this reference agree to 1e-6."""
    rare = _Point("apple banana zebra")
    common_1 = _Point("apple banana cherry")
    common_2 = _Point("apple cherry date and the of")
    passages = build_passage_rows([rare, common_1, common_2], "text")
    corpus = _term_frequencies(passages)
    assert corpus[f"{common_2.id}:0"] == {"apple": 1, "cherry": 1, "date": 1}, "stop words gone"

    scores = bm25_scores(corpus, tokenize_words("zebra apple", DEFAULT_STOP_WORDS))

    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[0] == f"{rare.id}:0"
    assert scores[f"{rare.id}:0"] == pytest.approx(1.114361, abs=1e-6)
    assert scores[f"{common_1.id}:0"] == pytest.approx(0.133531, abs=1e-6)
    assert scores[f"{common_1.id}:0"] == pytest.approx(scores[f"{common_2.id}:0"])
    # Sum over the distinct query terms of the per-term contribution.
    n, avgdl = 3, 3.0
    assert scores[f"{rare.id}:0"] == pytest.approx(
        bm25_term_score(1, 1, n, 3, avgdl) + bm25_term_score(1, 3, n, 3, avgdl)
    )
    assert bm25_scores(corpus, ["kiwi"]) == {}, "a passage with no query term has no score"


# --- adapter glue -------------------------------------------------------------------


def _bare_adapter():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter._write_locks = {}
    adapter._lexical_language = lambda: "simple"
    return adapter


def test_adapter_advertises_the_lexical_index():
    adapter = _bare_adapter()

    assert type(adapter).supports_lexical_search is True


@pytest.mark.asyncio
async def test_lexical_search_short_circuits_without_touching_the_database():
    adapter = _bare_adapter()
    adapter.has_collection = AsyncMock(return_value=False)

    assert await adapter.lexical_search("Doc_text", "   ") == []
    assert await adapter.lexical_search("Doc_text", "zebra", limit=0) == []
    adapter.has_collection.assert_not_awaited()

    assert await adapter.lexical_search("Doc_text", "zebra") == []
    adapter.has_collection.assert_awaited_once_with("Doc_text__lexical_docs")


@pytest.mark.asyncio
async def test_lexical_search_maps_rows_to_scored_results_best_first():
    adapter = _bare_adapter()
    adapter.has_collection = AsyncMock(return_value=True)
    session = object()

    @asynccontextmanager
    async def fake_session():
        yield session

    adapter.get_async_session = fake_session
    first, second = uuid4(), uuid4()
    rows = [(first, 2.5, {"text": "zebra"}), (second, 0.25, {"text": "apple"})]

    with patch(f"{ADAPTER_MODULE}.search_postings", AsyncMock(return_value=rows)) as search:
        results = await adapter.lexical_search("Doc_text", "zebra apple", limit=7)

    search.assert_awaited_once_with(session, "Doc_text", "zebra apple", 7, language="simple")
    assert [(result.id, result.score, result.payload) for result in results] == rows
    assert all(isinstance(result.id, UUID) for result in results)
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_remove_belongs_to_set_tags_skips_the_posting_tables():
    adapter = _bare_adapter()
    adapter.get_table_names = AsyncMock(
        return_value=[
            "Doc_text",
            "Doc_text__lexical_docs",
            "Doc_text__lexical_terms",
            "public.Entity_name",
            "users",
        ]
    )
    executed: list[str] = []

    class _Session:
        async def execute(self, statement, params=None):
            executed.append(str(statement))
            return SimpleNamespace(all=list)

        async def commit(self):
            return None

    @asynccontextmanager
    async def fake_session():
        yield _Session()

    adapter.get_async_session = fake_session

    await adapter.remove_belongs_to_set_tags(["DatasetA"])

    touched = {line.split("FROM ")[1].split()[0] for line in executed if "FROM " in line}
    assert touched == {'"Doc_text"', '"Entity_name"'}
    assert not any("__lexical_" in statement for statement in executed)
