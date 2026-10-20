"""BM25 posting index for a pgvector collection, in plain SQL.

A collection table ``"<collection>"`` (``id``, ``payload``, ``vector``) gets two
posting tables next to it, created by :func:`create_lexical_tables`:

* ``"<collection>__lexical_docs"`` -- one row per *passage* of an indexed row
  (``doc_id = "<row_id>:<passage_index>"``) with its length in terms. ``row_id``
  references the collection row ``ON DELETE CASCADE``, so a plain ``DELETE`` on
  the collection drops the postings with it.
* ``"<collection>__lexical_terms"`` -- one row per (term, passage) with the term
  frequency; ``doc_id`` references the docs table ``ON DELETE CASCADE``.

Text is tokenized by Postgres itself (``to_tsvector``) with the regconfig of the
configured language: ``simple`` lower-cases without stemming and the shared
``DEFAULT_STOP_WORDS`` are filtered after the fact, so the index holds the same
tokens as the in-memory scorer and the LanceDB index; ``english`` stems and
drops Postgres's own English stop words. Scoring is BM25 (``k1 = 1.2``,
``b = 0.75``) per passage, computed in the search statement from the live
tables (no stats table); a row's score is its best passage.

Everything here is a pure function of its arguments: statements are built from
a collection name, and the ``async`` runners take the SQLAlchemy connection or
session they run on. No adapter state is read, so a forked copy of the adapter
can import this module as is.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

from cognee.infrastructure.databases.vector.lexical import (
    DEFAULT_PASSAGE_TOKENS,
    DEFAULT_STOP_WORDS,
    LEXICAL_SEARCH_LANGUAGES,
    split_passages,
)

BM25_K1 = 1.2
BM25_B = 0.75

# Marker every posting table name carries; anything that enumerates a schema's
# tables and treats each as a collection must skip names containing it.
LEXICAL_TABLE_MARKER = "__lexical_"
DOCS_TABLE_SUFFIX = "__lexical_docs"
TERMS_TABLE_SUFFIX = "__lexical_terms"

# Type of the collection table's ``id`` column (``DataPoint.id``), repeated on
# ``docs.row_id`` and in the casts of the statements below.
ROW_ID_SQL_TYPE = "uuid"

# Passages per INSERT statement. A statement carries four arrays of this many
# elements (one of them the passage texts), so this bounds the parameter size.
PASSAGE_BATCH_SIZE = 500

# Rows per owned-rows upsert statement.
ROW_BATCH_SIZE = 1000

# A passage longer than this many characters is cut into pieces of at most
# this length. split_passages caps passages by words, and one whitespace-free
# run (an inline base64 image, minified JSON) can be megabytes long; Postgres
# refuses a tsvector over 1 MB, and at this length one cannot reach it.
MAX_PASSAGE_CHARS = 65_536

# Postgres truncates identifiers longer than this (NAMEDATALEN - 1) silently.
_MAX_IDENTIFIER_BYTES = 63


def fit_identifier(name: str) -> str:
    """``name`` if Postgres keeps it whole, else a prefix plus a hash of the full name.

    Postgres cuts identifiers to 63 bytes without an error, so the docs and
    terms tables of a very long collection name would both be cut to the same
    prefix and collide; the hash keeps them distinct.
    """
    if len(name.encode("utf-8")) <= _MAX_IDENTIFIER_BYTES:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    budget = _MAX_IDENTIFIER_BYTES - len(digest) - 1
    prefix = name.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return f"{prefix}_{digest}"


def lexical_table_names(collection_name: str) -> tuple[str, str]:
    """The (docs, terms) posting table names of ``collection_name``."""
    return (
        fit_identifier(f"{collection_name}{DOCS_TABLE_SUFFIX}"),
        fit_identifier(f"{collection_name}{TERMS_TABLE_SUFFIX}"),
    )


def is_lexical_table_name(table_name: str) -> bool:
    """True for a posting table (as opposed to a collection table)."""
    return LEXICAL_TABLE_MARKER in table_name


def quote_identifier(name: str) -> str:
    """Double-quote an SQL identifier, escaping embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def tokenizer_settings(language: str) -> tuple[str, list[str]]:
    """The ``to_tsvector`` regconfig and the stop words filtered after it, per language.

    ``simple``: the ``simple`` regconfig (lower-case, no stemming, keeps every
    token) and the shared ``DEFAULT_STOP_WORDS`` list, filtered in SQL.
    ``english``: the ``english`` regconfig, which stems and drops Postgres's own
    English stop words, and no extra filter (an empty list).
    """
    if language == "simple":
        return "simple", sorted(DEFAULT_STOP_WORDS)
    if language == "english":
        return "english", []
    raise ValueError(
        f"Unsupported lexical search language {language!r}; expected one of "
        f"{LEXICAL_SEARCH_LANGUAGES}"
    )


# --- passages -----------------------------------------------------------------


@dataclass(frozen=True)
class PassageRow:
    """One docs-table row to write: a passage of a collection row's text."""

    doc_id: str
    row_id: str
    passage_index: int
    text: str


def passage_doc_id(row_id: str, passage_index: int) -> str:
    """The docs-table key of passage ``passage_index`` of row ``row_id``."""
    return f"{row_id}:{passage_index}"


def clean_text(value: str) -> str:
    """``value`` without NUL characters, which Postgres text values cannot hold."""
    return value.replace("\x00", "")


def cut_passage(passage: str, max_chars: int = MAX_PASSAGE_CHARS) -> list[str]:
    """``passage`` in pieces of at most ``max_chars`` characters, cut at whitespace when it can."""
    pieces: list[str] = []
    rest = passage
    while len(rest) > max_chars:
        cut = rest.rfind(" ", max_chars // 2, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def build_passage_rows(
    data_points: Iterable[Any], field_name: str, max_tokens: int = DEFAULT_PASSAGE_TOKENS
) -> list[PassageRow]:
    """Split ``field_name`` of every data point into the passages to index.

    A point whose field is ``None`` or has no non-blank text yields no passage
    (its stale postings are still removed, since the caller deletes by row id).
    """
    rows: list[PassageRow] = []
    for data_point in data_points:
        value = getattr(data_point, field_name, None)
        if value is None:
            continue
        row_id = str(data_point.id)
        pieces = [
            piece
            for passage in split_passages(clean_text(str(value)), max_tokens)
            for piece in cut_passage(passage)
        ]
        for passage_index, passage in enumerate(pieces):
            rows.append(
                PassageRow(
                    doc_id=passage_doc_id(row_id, passage_index),
                    row_id=row_id,
                    passage_index=passage_index,
                    text=passage,
                )
            )
    return rows


def dedupe_owned_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows that share an ``id`` (last one wins), uniting ``belongs_to_set``.

    ``ON CONFLICT DO UPDATE`` refuses to touch one row twice in a statement, and
    a tag carried by only one duplicate must not be dropped (the same merge the
    embedding write path does).
    """
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        existing = deduped.get(row_id)
        if existing is None:
            deduped[row_id] = row
            continue
        existing_tags = (existing.get("payload") or {}).get("belongs_to_set") or []
        incoming_payload = dict(row.get("payload") or {})
        incoming_tags = incoming_payload.get("belongs_to_set") or []
        if existing_tags or incoming_tags:
            incoming_payload["belongs_to_set"] = list(
                dict.fromkeys(list(existing_tags) + list(incoming_tags))
            )
        deduped[row_id] = {**row, "payload": incoming_payload}
    return list(deduped.values())


# --- DDL ----------------------------------------------------------------------


def create_lexical_tables_statements(collection_name: str) -> list[TextClause]:
    """The idempotent DDL for the two posting tables of ``collection_name``.

    The collection table must already exist: ``docs.row_id`` references it, and
    that cascade is what keeps the index consistent with row deletion.
    """
    docs_table, terms_table = lexical_table_names(collection_name)
    docs, terms, collection = (
        quote_identifier(docs_table),
        quote_identifier(terms_table),
        quote_identifier(collection_name),
    )
    docs_row_index = quote_identifier(fit_identifier(f"{docs_table}_row_id_idx"))
    terms_doc_index = quote_identifier(fit_identifier(f"{terms_table}_doc_id_idx"))
    return [
        text(
            f"CREATE TABLE IF NOT EXISTS {docs} ("
            "doc_id TEXT PRIMARY KEY, "
            f"row_id {ROW_ID_SQL_TYPE.upper()} NOT NULL "
            f"REFERENCES {collection} (id) ON DELETE CASCADE, "
            "passage_index INTEGER NOT NULL, "
            "doc_len INTEGER NOT NULL)"
        ),
        text(f"CREATE INDEX IF NOT EXISTS {docs_row_index} ON {docs} (row_id)"),
        text(
            f"CREATE TABLE IF NOT EXISTS {terms} ("
            "term TEXT NOT NULL, "
            f"doc_id TEXT NOT NULL REFERENCES {docs} (doc_id) ON DELETE CASCADE, "
            "tf INTEGER NOT NULL, "
            "PRIMARY KEY (term, doc_id))"
        ),
        text(f"CREATE INDEX IF NOT EXISTS {terms_doc_index} ON {terms} (doc_id)"),
    ]


async def create_lexical_tables(connection, collection_name: str) -> None:
    """Run :func:`create_lexical_tables_statements` on ``connection`` (an open transaction)."""
    for statement in create_lexical_tables_statements(collection_name):
        await connection.execute(statement)


# --- writes -------------------------------------------------------------------


def upsert_owned_rows_statement(collection_name: str) -> TextClause:
    """Upsert collection rows that have no embedding: ``vector`` NULL, payload replaced.

    Binds: ``ids`` (text[]), ``payloads`` (text[] of JSON documents). On
    conflict only ``payload`` is written -- a vector stored by the embedding
    path is never touched -- and the ``belongs_to_set`` tags of the stored and
    the incoming payload are united, as the embedding write path does.
    """
    collection = quote_identifier(collection_name)
    return text(
        f"""
        INSERT INTO {collection} (id, payload, vector)
        SELECT CAST(r.id AS {ROW_ID_SQL_TYPE}), CAST(r.payload AS json), NULL
        FROM unnest(CAST(:ids AS text[]), CAST(:payloads AS text[])) AS r(id, payload)
        ON CONFLICT (id) DO UPDATE SET payload = jsonb_set(
            CAST(EXCLUDED.payload AS jsonb),
            '{{belongs_to_set}}',
            (
                SELECT COALESCE(jsonb_agg(DISTINCT val), '[]'::jsonb)
                FROM jsonb_array_elements_text(
                    COALESCE(CAST({collection}.payload AS jsonb)->'belongs_to_set', '[]'::jsonb)
                    || COALESCE(CAST(EXCLUDED.payload AS jsonb)->'belongs_to_set', '[]'::jsonb)
                ) AS val
            )
        )::json
        """
    )


async def upsert_owned_rows(session, collection_name: str, rows: Sequence[dict[str, Any]]) -> None:
    """Write ``rows`` (``{"id", "payload"}`` dicts, payload JSON-serializable) with NULL vectors."""
    rows = dedupe_owned_rows(rows)
    statement = upsert_owned_rows_statement(collection_name)
    for start in range(0, len(rows), ROW_BATCH_SIZE):
        batch = rows[start : start + ROW_BATCH_SIZE]
        await session.execute(
            statement,
            {
                "ids": [str(row["id"]) for row in batch],
                "payloads": [json.dumps(row["payload"]) for row in batch],
            },
        )


def delete_postings_statement(collection_name: str) -> TextClause:
    """Delete the docs rows (and, by cascade, the terms) of the rows in ``row_ids`` (text[])."""
    docs_table, _ = lexical_table_names(collection_name)
    return text(
        f"DELETE FROM {quote_identifier(docs_table)} "
        f"WHERE row_id = ANY(CAST(CAST(:row_ids AS text[]) AS {ROW_ID_SQL_TYPE}[]))"
    )


def insert_passages_statement(collection_name: str) -> TextClause:
    """Index a batch of passages: one statement writing docs and terms.

    Binds: ``doc_ids``, ``row_ids``, ``texts`` (text[]), ``passage_indexes``
    (integer[]), ``config`` (the regconfig name) and ``stop_words`` (text[],
    may be empty). Each passage is tokenized once with ``to_tsvector``; the
    term rows are the lexemes left after the stop-word filter with their
    position counts, and the docs row's ``doc_len`` is their sum (0 for a
    passage with no term left).
    """
    docs_table, terms_table = lexical_table_names(collection_name)
    docs, terms = quote_identifier(docs_table), quote_identifier(terms_table)
    return text(
        f"""
        WITH passages AS (
            SELECT p.doc_id, p.row_id, p.passage_index, p.body
            FROM unnest(
                CAST(:doc_ids AS text[]),
                CAST(:row_ids AS text[]),
                CAST(:passage_indexes AS integer[]),
                CAST(:texts AS text[])
            ) AS p(doc_id, row_id, passage_index, body)
        ),
        passage_terms AS (
            SELECT p.doc_id, v.lexeme AS term, cardinality(v.positions) AS tf
            FROM passages p
            CROSS JOIN LATERAL unnest(
                to_tsvector(CAST(CAST(:config AS text) AS regconfig), p.body)
            ) AS v(lexeme, positions, weights)
            WHERE v.lexeme <> ALL(CAST(:stop_words AS text[]))
        ),
        passage_lengths AS (
            SELECT doc_id, sum(tf) AS doc_len FROM passage_terms GROUP BY doc_id
        ),
        inserted_docs AS (
            INSERT INTO {docs} (doc_id, row_id, passage_index, doc_len)
            SELECT
                p.doc_id,
                CAST(p.row_id AS {ROW_ID_SQL_TYPE}),
                p.passage_index,
                CAST(COALESCE(l.doc_len, 0) AS integer)
            FROM passages p
            LEFT JOIN passage_lengths l ON l.doc_id = p.doc_id
            RETURNING doc_id
        )
        INSERT INTO {terms} (term, doc_id, tf)
        SELECT t.term, t.doc_id, t.tf
        FROM passage_terms t
        JOIN inserted_docs d ON d.doc_id = t.doc_id
        """
    )


async def replace_postings(
    session,
    collection_name: str,
    row_ids: Sequence[str],
    passages: Sequence[PassageRow],
    *,
    language: str,
) -> None:
    """Replace the postings of ``row_ids`` with ``passages``, inside the caller's transaction.

    Deletes every posting of the given rows first (cascade), then inserts the
    passages in batches of ``PASSAGE_BATCH_SIZE``, one statement each.
    """
    if not row_ids:
        return
    config, stop_words = tokenizer_settings(language)
    await session.execute(
        delete_postings_statement(collection_name), {"row_ids": [str(r) for r in row_ids]}
    )
    statement = insert_passages_statement(collection_name)
    for start in range(0, len(passages), PASSAGE_BATCH_SIZE):
        batch = passages[start : start + PASSAGE_BATCH_SIZE]
        await session.execute(
            statement,
            {
                "doc_ids": [p.doc_id for p in batch],
                "row_ids": [p.row_id for p in batch],
                "passage_indexes": [p.passage_index for p in batch],
                "texts": [p.text for p in batch],
                "config": config,
                "stop_words": stop_words,
            },
        )


# --- search -------------------------------------------------------------------


def query_terms_statement() -> TextClause:
    """Tokenize a query as the passages were: the distinct lexemes left after the stop words.

    Binds ``config``, ``query`` and ``stop_words``. Run on its own, so the
    search statement gets the terms as a bound array: the planner cannot
    estimate an inline ``to_tsvector`` and would scan every posting.
    """
    return text(
        """
        SELECT DISTINCT v.lexeme
        FROM unnest(
            to_tsvector(CAST(CAST(:config AS text) AS regconfig), CAST(:query AS text))
        ) AS v(lexeme, positions, weights)
        WHERE v.lexeme <> ALL(CAST(:stop_words AS text[]))
        ORDER BY v.lexeme
        """
    )


def search_statement(collection_name: str) -> TextClause:
    """Rank the collection's rows by BM25 relevance to the query terms, best first.

    Binds: ``terms`` (text[], from :func:`query_terms_statement`), ``k1``,
    ``b`` (double precision) and ``limit`` (integer; NULL for no limit). The
    terms table's ``(term, doc_id)`` key finds the postings of the query
    terms only; ``N`` and ``avgdl`` come from the docs table, ``df`` from the
    terms table (both computed once: ``MATERIALIZED``, since an inlined CTE is
    re-evaluated per posting), each passage sums the distinct query terms it
    contains, and a row keeps its best passage. Rows are joined to the collection table for the
    payload. Ties break on ``row_id`` so the order is deterministic.
    """
    docs_table, terms_table = lexical_table_names(collection_name)
    docs, terms, collection = (
        quote_identifier(docs_table),
        quote_identifier(terms_table),
        quote_identifier(collection_name),
    )
    return text(
        f"""
        WITH corpus AS MATERIALIZED (
            SELECT
                CAST(count(*) AS double precision) AS n,
                CAST(COALESCE(avg(doc_len), 0) AS double precision) AS avgdl
            FROM {docs}
        ),
        term_df AS MATERIALIZED (
            SELECT t.term, CAST(count(*) AS double precision) AS df
            FROM {terms} t
            WHERE t.term = ANY(CAST(:terms AS text[]))
            GROUP BY t.term
        ),
        passage_scores AS (
            SELECT
                d.row_id,
                sum(
                    ln(1 + (c.n - f.df + 0.5) / (f.df + 0.5))
                    * (CAST(t.tf AS double precision) * (CAST(:k1 AS double precision) + 1))
                    / (
                        CAST(t.tf AS double precision)
                        + CAST(:k1 AS double precision) * (
                            1 - CAST(:b AS double precision)
                            + CAST(:b AS double precision)
                            * COALESCE(CAST(d.doc_len AS double precision) / NULLIF(c.avgdl, 0), 0)
                        )
                    )
                ) AS score
            FROM {terms} t
            JOIN term_df f ON f.term = t.term
            JOIN {docs} d ON d.doc_id = t.doc_id
            CROSS JOIN corpus c
            WHERE t.term = ANY(CAST(:terms AS text[]))
            GROUP BY d.doc_id, d.row_id
        ),
        row_scores AS (
            SELECT row_id, max(score) AS score FROM passage_scores GROUP BY row_id
        )
        SELECT r.row_id, r.score, c.payload
        FROM row_scores r
        JOIN {collection} c ON c.id = r.row_id
        ORDER BY r.score DESC, r.row_id ASC
        LIMIT CAST(:limit AS integer)
        """
    )


async def search_postings(
    session,
    collection_name: str,
    query_text: str,
    limit: int | None,
    *,
    language: str,
) -> list[tuple[Any, float, dict[str, Any] | None]]:
    """Tokenize the query, then run :func:`search_statement`; ``(row_id, score, payload)``
    tuples, best first. A query with no term left after tokenizing has no hit."""
    config, stop_words = tokenizer_settings(language)
    terms_result = await session.execute(
        query_terms_statement(),
        {"config": config, "query": clean_text(query_text), "stop_words": stop_words},
    )
    terms = [term for (term,) in terms_result.all()]
    if not terms:
        return []
    result = await session.execute(
        search_statement(collection_name),
        {"terms": terms, "k1": BM25_K1, "b": BM25_B, "limit": limit},
    )
    rows = []
    for row_id, score, payload in result.all():
        if isinstance(payload, (str, bytes, bytearray)):
            payload = json.loads(payload)
        rows.append((row_id, float(score), payload))
    return rows


# --- reference arithmetic -----------------------------------------------------


def bm25_term_score(
    tf: int, df: int, n: int, doc_len: int, avgdl: float, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    """One query term's BM25 contribution to one passage; the formula the SQL computes.

    ``idf = ln(1 + (N - df + 0.5) / (df + 0.5))`` times the saturated,
    length-normalized term frequency ``tf * (k1 + 1) / (tf + k1 * (1 - b + b *
    doc_len / avgdl))``.
    """
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    length_ratio = doc_len / avgdl if avgdl else 0.0
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length_ratio))


def bm25_scores(
    term_frequencies: dict[str, dict[str, int]], query_terms: Iterable[str]
) -> dict[str, float]:
    """Reference BM25 over a corpus of passages, ``{doc_id: {term: tf}}``.

    Returns ``{doc_id: score}`` for the passages that contain at least one
    query term, summing :func:`bm25_term_score` over the distinct query terms
    with ``N`` the number of passages and ``avgdl`` their mean length. Tests
    check the search statement against it.
    """
    n = len(term_frequencies)
    doc_lens = {doc_id: sum(tfs.values()) for doc_id, tfs in term_frequencies.items()}
    avgdl = sum(doc_lens.values()) / n if n else 0.0
    terms = set(query_terms)
    document_frequency = {
        term: sum(1 for tfs in term_frequencies.values() if term in tfs) for term in terms
    }
    scores: dict[str, float] = {}
    for doc_id, tfs in term_frequencies.items():
        score = 0.0
        matched = False
        for term in terms:
            tf = tfs.get(term, 0)
            if tf <= 0:
                continue
            matched = True
            score += bm25_term_score(tf, document_frequency[term], n, doc_lens[doc_id], avgdl)
        if matched:
            scores[doc_id] = score
    return scores
