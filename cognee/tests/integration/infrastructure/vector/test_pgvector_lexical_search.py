"""The pgvector lexical (BM25) index against a live Postgres with the pgvector extension.

No LLM and no embedding API: the adapter gets a fake embedding engine that
returns one fixed vector, so the tests only cost a database round trip.

Configured the way CI's pgvector jobs are (``.github/workflows/vector_db_tests.yml``):
``VECTOR_DB_PROVIDER=pgvector`` with ``VECTOR_DB_HOST`` / ``VECTOR_DB_PORT`` /
``VECTOR_DB_USERNAME`` / ``VECTOR_DB_PASSWORD`` / ``VECTOR_DB_NAME``, each falling
back to the relational ``DB_HOST`` / ``DB_PORT`` / ``DB_USERNAME`` / ``DB_PASSWORD`` /
``DB_NAME`` as ``create_vector_engine`` does (defaults ``127.0.0.1:5432``,
``cognee`` / ``cognee``, ``cognee_db``). ``TEST_PGVECTOR_URL`` (a full
``postgresql+asyncpg://`` URL, as ``test_belongs_to_set_pgvector.py`` takes it)
overrides all of that. Skipped when the provider is not pgvector, the postgres
extra is not installed, or the database cannot be reached.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text as sql_text
from sqlalchemy.engine import make_url

from cognee.infrastructure.databases.vector.lexical import (
    DEFAULT_STOP_WORDS,
    split_passages,
    tokenize_words,
)
from cognee.infrastructure.databases.vector.pgvector.lexical_index import (
    bm25_scores,
    build_passage_rows,
    lexical_table_names,
)
from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    HAS_PGVECTOR = True
except ModuleNotFoundError:
    HAS_PGVECTOR = False


def _pg_url() -> str | None:
    explicit = os.getenv("TEST_PGVECTOR_URL")
    if explicit:
        return explicit
    if os.getenv("VECTOR_DB_PROVIDER", "").lower() != "pgvector":
        return None
    host = os.getenv("VECTOR_DB_HOST") or os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("VECTOR_DB_PORT") or os.getenv("DB_PORT", "5432")
    username = os.getenv("VECTOR_DB_USERNAME") or os.getenv("DB_USERNAME", "cognee")
    password = os.getenv("VECTOR_DB_PASSWORD") or os.getenv("DB_PASSWORD", "cognee")
    name = os.getenv("VECTOR_DB_NAME") or os.getenv("DB_NAME", "cognee_db")
    return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{name}"


PG_URL = _pg_url()

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not PG_URL,
        reason="VECTOR_DB_PROVIDER is not pgvector and TEST_PGVECTOR_URL is unset",
    ),
    pytest.mark.skipif(not HAS_PGVECTOR, reason="postgres extra (asyncpg, pgvector) not installed"),
]

FIXED_VECTOR = [0.1, 0.2, 0.3]


class _FakeEmbeddingEngine:
    """Deterministic stand-in: one fixed vector per text, no API call."""

    def get_vector_size(self) -> int:
        return len(FIXED_VECTOR)

    def get_batch_size(self) -> int:
        return 100

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [list(FIXED_VECTOR) for _ in texts]


class _LexicalOnlyPoint(DataPoint):
    """Text kept only in the keyword index (``CachedDocument``'s shape): no embedding."""

    text: str
    metadata: dict = {"index_fields": [], "lexical_fields": ["text"]}


class _EmbeddedPoint(DataPoint):
    """Text that is embedded and keyword-indexed in the same collection."""

    text: str
    metadata: dict = {"index_fields": ["text"], "lexical_fields": ["text"]}


@pytest.fixture(scope="module")
def pg_url() -> str:
    """The database URL, after proving the server answers (else skip)."""
    import asyncpg

    url = make_url(PG_URL)

    async def probe():
        connection = await asyncpg.connect(
            host=url.host,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            database=url.database,
            timeout=5,
        )
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await connection.close()

    try:
        asyncio.run(probe())
    except (OSError, asyncio.TimeoutError, asyncpg.PostgresError) as error:
        # Refused or timed-out connections, bad credentials, a server without
        # the vector extension: no usable Postgres here.
        pytest.skip(
            f"Postgres at {url.render_as_string(hide_password=True)} is unreachable: "
            f"{type(error).__name__}: {error}"
        )
    return PG_URL


@asynccontextmanager
async def _adapter_with_collection(pg_url: str):
    """An adapter on the live database and a fresh type name; drops its tables after."""
    adapter = PGVectorAdapter(
        connection_string=pg_url, api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    type_name = f"LexicalPoint{uuid4().hex[:8]}"
    collection = f"{type_name}_text"
    try:
        yield adapter, type_name, collection
    finally:
        docs_table, terms_table = lexical_table_names(collection)
        async with adapter.get_async_session() as session:
            for table_name in (terms_table, docs_table, collection):
                await session.execute(sql_text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
            await session.commit()
        await adapter.close()


async def _scalar(adapter, statement: str, **params):
    async with adapter.get_async_session() as session:
        return (await session.execute(sql_text(statement), params)).scalar_one()


async def _count_postings(adapter, collection: str, row_id: UUID) -> tuple[int, int]:
    """(docs rows, terms rows) stored for ``row_id``."""
    docs_table, terms_table = lexical_table_names(collection)
    docs = await _scalar(
        adapter, f'SELECT count(*) FROM "{docs_table}" WHERE row_id = :row_id', row_id=row_id
    )
    terms = await _scalar(
        adapter,
        f'SELECT count(*) FROM "{terms_table}" t JOIN "{docs_table}" d '
        "ON d.doc_key = t.doc_key WHERE d.row_id = :row_id",
        row_id=row_id,
    )
    return docs, terms


def _fixture_points():
    return [
        _LexicalOnlyPoint(text="apple banana zebra"),
        _LexicalOnlyPoint(text="apple banana cherry"),
        _LexicalOnlyPoint(text="apple cherry date and the of"),
    ]


async def test_owned_rows_have_no_vector_and_are_found(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        points = _fixture_points()
        await adapter.create_lexical_index(type_name, "text")
        await adapter.index_lexical_data_points(type_name, "text", points, owns_rows=True)

        rows = await _scalar(adapter, f'SELECT count(*) FROM "{collection}"')
        without_vector = await _scalar(
            adapter, f'SELECT count(*) FROM "{collection}" WHERE vector IS NULL'
        )
        assert (rows, without_vector) == (3, 3)

        results = await adapter.lexical_search(collection, "zebra")

        assert [result.id for result in results] == [points[0].id]
        assert results[0].payload["text"] == "apple banana zebra"
        assert results[0].payload["id"] == str(points[0].id), "payload is the point's dump"
        assert results[0].score > 0

        stored = await adapter.retrieve(collection, [str(point.id) for point in points])
        assert len(stored) == 3


async def test_rare_term_outranks_common_one_and_scores_match_the_reference(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        points = _fixture_points()
        await adapter.index_lexical_data_points(type_name, "text", points, owns_rows=True)

        results = await adapter.lexical_search(collection, "zebra apple", limit=10)

        assert results[0].id == points[0].id, "the only row with the rare term ranks first"
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True), "best first"
        assert scores[0] > scores[1]
        assert scores[1] == pytest.approx(scores[2]), "the two common-only rows tie"

        # The statement's arithmetic against the Python reference on the same tokens.
        corpus = {}
        for passage in build_passage_rows(points, "text"):
            counts: dict[str, int] = {}
            for token in tokenize_words(passage.text, DEFAULT_STOP_WORDS):
                counts[token] = counts.get(token, 0) + 1
            corpus[passage.doc_id] = counts
        expected = bm25_scores(corpus, tokenize_words("zebra apple", DEFAULT_STOP_WORDS))
        for result in results:
            assert result.score == pytest.approx(expected[f"{result.id}:0"], abs=1e-9)

        assert len(await adapter.lexical_search(collection, "zebra apple", limit=1)) == 1
        assert len(await adapter.lexical_search(collection, "apple", limit=None)) == 3


async def test_rows_written_by_the_embedding_path_keep_their_vector(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        points = [_EmbeddedPoint(text="quantum zebra"), _EmbeddedPoint(text="classical apple")]
        await adapter.create_lexical_index(type_name, "text")
        # What index_data_points does for a field in both index_fields and
        # lexical_fields: embed first, then index the text without owning rows.
        await adapter.index_data_points(type_name, "text", points)
        await adapter.index_lexical_data_points(type_name, "text", points, owns_rows=False)

        with_vector = await _scalar(
            adapter, f'SELECT count(*) FROM "{collection}" WHERE vector IS NOT NULL'
        )
        assert with_vector == 2
        stored_vector = await _scalar(
            adapter,
            f'SELECT CAST(vector AS text) FROM "{collection}" WHERE id = :row_id',
            row_id=points[0].id,
        )
        assert stored_vector == "[0.1,0.2,0.3]"

        results = await adapter.lexical_search(collection, "zebra")
        assert [result.id for result in results] == [points[0].id]
        assert results[0].payload["text"] == "quantum zebra", "the embedding path's payload"

        # The vector search over the same collection is unaffected.
        hits = await adapter.search(collection, query_vector=FIXED_VECTOR, limit=None)
        assert sorted(hit.id for hit in hits) == sorted(point.id for point in points)
        assert all(hit.score == pytest.approx(0.0, abs=1e-6) for hit in hits)


async def test_adding_a_lexical_index_backfills_existing_embeddings(pg_url):
    """Adding lexical_fields to an existing type must not hide its older rows."""
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        points = [_EmbeddedPoint(text=f"pelican archive {index}") for index in range(35)]
        await adapter.create_vector_index(type_name, "text")
        await adapter.index_data_points(type_name, "text", points)
        assert not await adapter.has_lexical_index(collection)

        await adapter.create_lexical_index(type_name, "text")

        assert await adapter.has_lexical_index(collection)
        found = await adapter.lexical_search(collection, "pelican", limit=None)
        assert {hit.id for hit in found} == {point.id for point in points}
        # Repeated creation must neither duplicate postings nor touch vectors.
        await adapter.create_lexical_index(type_name, "text")
        assert len(await adapter.lexical_search(collection, "pelican", limit=None)) == len(points)
        assert await _scalar(
            adapter, f'SELECT count(*) FROM "{collection}" WHERE vector IS NOT NULL'
        ) == len(points)


async def test_delete_data_points_removes_the_postings(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        points = _fixture_points()
        await adapter.index_lexical_data_points(type_name, "text", points, owns_rows=True)
        assert await _count_postings(adapter, collection, points[0].id) == (1, 3)

        await adapter.delete_data_points(collection, [points[0].id])

        assert await _count_postings(adapter, collection, points[0].id) == (0, 0)
        assert await adapter.lexical_search(collection, "zebra") == []
        remaining = await adapter.lexical_search(collection, "cherry")
        assert {result.id for result in remaining} == {points[1].id, points[2].id}


async def test_missing_collection_returns_nothing(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, _type_name, _collection):
        assert await adapter.lexical_search(f"Missing{uuid4().hex[:8]}_text", "zebra") == []

        # A collection without a lexical index is "missing" for lexical search too.
        plain = _EmbeddedPoint(text="no keyword index here")
        plain_type = f"Plain{uuid4().hex[:8]}"
        try:
            await adapter.index_data_points(plain_type, "text", [plain])
            assert await adapter.lexical_search(f"{plain_type}_text", "keyword") == []
        finally:
            async with adapter.get_async_session() as session:
                await session.execute(sql_text(f'DROP TABLE IF EXISTS "{plain_type}_text"'))
                await session.commit()


async def test_reindexing_a_row_replaces_its_postings(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        point = _LexicalOnlyPoint(text="apple banana zebra", belongs_to_set=["DatasetA"])
        await adapter.index_lexical_data_points(type_name, "text", [point], owns_rows=True)
        assert await _count_postings(adapter, collection, point.id) == (1, 3)

        edited = _LexicalOnlyPoint(id=point.id, text="kiwi", belongs_to_set=["DatasetB"])
        await adapter.index_lexical_data_points(type_name, "text", [edited], owns_rows=True)

        assert await _count_postings(adapter, collection, point.id) == (1, 1)
        assert await adapter.lexical_search(collection, "zebra") == []
        results = await adapter.lexical_search(collection, "kiwi")
        assert [result.id for result in results] == [point.id]
        assert results[0].payload["text"] == "kiwi"
        assert sorted(results[0].payload["belongs_to_set"]) == ["DatasetA", "DatasetB"], (
            "tags are united on conflict, as the embedding write path does"
        )
        assert await _scalar(adapter, f'SELECT count(*) FROM "{collection}"') == 1


async def test_long_text_is_indexed_as_passages_and_collapsed_per_row(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        filler = " ".join(f"word{i}" for i in range(2400))
        point = _LexicalOnlyPoint(text=f"{filler} zebra")
        other = _LexicalOnlyPoint(text="a short zebra note")
        passage_count = len(split_passages(point.text))
        assert passage_count == 3, "fixture must span several passages"

        await adapter.index_lexical_data_points(type_name, "text", [point, other], owns_rows=True)

        docs, _terms = await _count_postings(adapter, collection, point.id)
        assert docs == passage_count

        results = await adapter.lexical_search(collection, "zebra")
        assert sorted(result.id for result in results) == sorted([point.id, other.id])
        assert len(results) == 2, "one result per row, not per passage"
        # The short passage is the better one for the shared term.
        assert results[0].id == other.id


async def test_simple_language_drops_the_shared_stop_words(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        point = _LexicalOnlyPoint(text="the cat and the hat")
        await adapter.index_lexical_data_points(type_name, "text", [point], owns_rows=True)

        _docs_table, terms_table = lexical_table_names(collection)
        async with adapter.get_async_session() as session:
            terms = (
                await session.execute(
                    sql_text(f'SELECT term, tf FROM "{terms_table}" ORDER BY term')
                )
            ).all()
        assert [tuple(row) for row in terms] == [("cat", 1), ("hat", 1)]
        doc_len = await _scalar(
            adapter,
            f'SELECT doc_len FROM "{collection}__lexical_docs" WHERE row_id = :row_id',
            row_id=point.id,
        )
        assert doc_len == 2

        assert await adapter.lexical_search(collection, "the and") == []
        assert [r.id for r in await adapter.lexical_search(collection, "The Hat")] == [point.id]


async def test_english_language_stems_and_uses_postgres_stop_words(pg_url, monkeypatch):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        monkeypatch.setattr(adapter, "_lexical_language", lambda: "english")
        point = _LexicalOnlyPoint(text="The runners were running quickly")
        await adapter.index_lexical_data_points(type_name, "text", [point], owns_rows=True)

        assert [r.id for r in await adapter.lexical_search(collection, "run")] == [point.id]
        assert await adapter.lexical_search(collection, "the were") == []

    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        point = _LexicalOnlyPoint(text="The runners were running quickly")
        await adapter.index_lexical_data_points(type_name, "text", [point], owns_rows=True)

        assert await adapter.lexical_search(collection, "run") == [], "simple: no stemming"
        assert [r.id for r in await adapter.lexical_search(collection, "running")] == [point.id]


async def test_create_lexical_index_is_idempotent_and_detagging_cascades(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        await adapter.create_lexical_index(type_name, "text")
        await adapter.create_lexical_index(type_name, "text")
        adapter.reset_metadata_cache()
        await adapter.create_lexical_index(type_name, "text")

        docs_table, terms_table = lexical_table_names(collection)
        for table_name in (collection, docs_table, terms_table):
            exists = await _scalar(
                adapter, "SELECT to_regclass(:name) IS NOT NULL", name=f'"{table_name}"'
            )
            assert exists, table_name

        point = _LexicalOnlyPoint(text="apple zebra", belongs_to_set=["DatasetA"])
        await adapter.index_lexical_data_points(type_name, "text", [point], owns_rows=True)

        # Detagging walks every collection table; the posting tables are skipped,
        # and the row emptied of tags is deleted together with its postings.
        await adapter.remove_belongs_to_set_tags(["DatasetA"])

        assert await _scalar(adapter, f'SELECT count(*) FROM "{collection}"') == 0
        assert await _count_postings(adapter, collection, point.id) == (0, 0)
        assert await adapter.lexical_search(collection, "zebra") == []


async def test_index_data_points_task_drives_both_lexical_paths(pg_url):
    """The driver passes (type name, field) exactly as it does for embeddings, so a
    lexical-only type lands in ``<Type>_text`` with NULL vectors and a type that is
    embedded too keeps its vectors; both answer ``lexical_search`` on that table."""
    from pydantic import create_model

    from cognee.tasks.storage.index_data_points import index_data_points

    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        lexical_only = create_model(
            type_name,
            __base__=DataPoint,
            text=(str, ...),
            metadata=(dict, {"index_fields": [], "lexical_fields": ["text"]}),
        )
        points = [lexical_only(text="apple banana zebra"), lexical_only(text="apple cherry")]

        await index_data_points(points, vector_engine=adapter)

        without_vector = await _scalar(
            adapter, f'SELECT count(*) FROM "{collection}" WHERE vector IS NULL'
        )
        assert without_vector == 2
        results = await adapter.lexical_search(collection, "zebra")
        assert [result.id for result in results] == [points[0].id]

    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        embedded = create_model(
            type_name,
            __base__=DataPoint,
            text=(str, ...),
            metadata=(dict, {"index_fields": ["text"], "lexical_fields": ["text"]}),
        )
        points = [embedded(text="quantum zebra"), embedded(text="classical apple")]

        await index_data_points(points, vector_engine=adapter)

        with_vector = await _scalar(
            adapter, f'SELECT count(*) FROM "{collection}" WHERE vector IS NOT NULL'
        )
        assert with_vector == 2
        results = await adapter.lexical_search(collection, "zebra")
        assert [result.id for result in results] == [points[0].id]
        assert results[0].payload["text"] == "quantum zebra"


async def test_has_lexical_index_follows_the_posting_tables(pg_url):
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        assert await adapter.has_lexical_index(collection) is False
        await adapter.create_lexical_index(type_name, "text")
        assert await adapter.has_lexical_index(collection) is True


async def test_text_with_nul_characters_indexes_and_is_found(pg_url):
    """PDF extractions carry NUL, which Postgres text rejects; the batch must not fail."""
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        clean = _LexicalOnlyPoint(text="the pelican harbour report")
        with_nul = _LexicalOnlyPoint(text="binary\x00ish notes about the pelican colony")
        await adapter.create_lexical_index(type_name, "text")
        await adapter.index_lexical_data_points(
            type_name, "text", [clean, with_nul], owns_rows=True
        )

        results = await adapter.lexical_search(collection, "pelican")
        assert {result.id for result in results} == {clean.id, with_nul.id}
        assert len(await adapter.lexical_search(collection, "pel\x00ican")) == 2


async def test_a_megabyte_whitespace_free_run_indexes_and_is_found(pg_url):
    """An inline base64 image exceeded the 1 MB tsvector limit as one passage."""
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        blob = "QUJD" * 400_000
        image_note = _LexicalOnlyPoint(
            text=f"A photo of the pelican.\n\n![photo](data:image/png;base64,{blob})"
        )
        await adapter.create_lexical_index(type_name, "text")
        await adapter.index_lexical_data_points(type_name, "text", [image_note], owns_rows=True)

        results = await adapter.lexical_search(collection, "pelican")
        assert [result.id for result in results] == [image_note.id]


async def test_an_index_keeps_the_language_it_was_built_with(pg_url, monkeypatch):
    """Switching LEXICAL_SEARCH_LANGUAGE does not re-tokenize stored postings, so
    queries and later writes keep the recorded language instead of missing matches."""
    async with _adapter_with_collection(pg_url) as (adapter, type_name, collection):
        first = _LexicalOnlyPoint(text="The runners were running quickly")
        await adapter.index_lexical_data_points(type_name, "text", [first], owns_rows=True)

        monkeypatch.setattr(adapter, "_lexical_language", lambda: "english")
        later = _LexicalOnlyPoint(text="Running again tomorrow")
        await adapter.index_lexical_data_points(type_name, "text", [later], owns_rows=True)

        found = await adapter.lexical_search(collection, "running")
        assert {result.id for result in found} == {first.id, later.id}
        assert await adapter.lexical_search(collection, "run") == [], "still simple: no stemming"
