"""BROAD search type: questions that need a whole dataset, answered exactly (SDK-324).

Ordinary search shows the model the passages most similar to the question, which is
enough for "what" and not for "how many": a count, a total or a list needs every record.
BROAD builds a record store of the whole dataset (``broad_store.py``: tables for records,
a ``lines`` table for the rest of the text, the graph's entities) and makes one decision
call:

- the dataset fits one prompt: the model sees all of it and either answers, or writes a
  SQL query when the answer needs many rows counted, summed or compared;
- it does not fit: the model writes a SQL query over the store, or says the question
  needs the text read; then the chunks most similar to the question fill one prompt.

A query runs read-only in SQLite, and the model phrases its result. Code appends how the
answer was reached, and the full result when it has more than one row.
"""

import json
import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from itertools import groupby, pairwise
from typing import Any, Literal
from uuid import UUID

from fastapi import status
from pydantic import BaseModel, Field
from sqlalchemy import select

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.broad_store import BroadQueryError, RecordStore
from cognee.modules.retrieval.broad_worker import BroadWorker
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")

# A dataset up to this many tokens is shown to the model whole; a larger one is answered
# through the record store.
BROAD_CONTEXT_TOKENS = 30_000
# The size of the prompt the reading route fills with the parts most similar to the question.
BROAD_READING_TOKENS = 12_000
# Megabytes of text held in memory at once while the store is built.
BROAD_BATCH_MB = 256
# A dataset holding up to this many megabytes (million characters) of text is loaded into
# the store whole; a larger one is represented by the parts most similar to the question,
# up to this size.
BROAD_STORE_MB = 512
# The least text a chunk is assumed to hold, when deciding how many to ask the vector
# store for; a chunk is rarely shorter.
_MIN_CHUNK_CHARS = 1_000
# Chunks whose text one call fetches while the most similar parts are loaded.
_SIMILAR_FETCH_CHUNKS = 1_000
# Rows of a query result shown to the answer model; the full result is appended by code.
BROAD_RESULT_ROWS_SHOWN = 200
BROAD_TEXT_NODE_TYPES = ("DocumentChunk", "DltRow")
# Text longer than this many characters per token of the budget cannot fit in any script,
# so its tokens are not counted.
_MAX_CHARS_PER_TOKEN = 6
# Chunks each reading search returns: enough to fill the budget when a chunk holds at
# least this many tokens.
_MIN_CHUNK_TOKENS = 256


class BroadLimitError(CogneeValidationError):
    """The dataset exceeds a BROAD limit; the message names the setting to raise. Not
    logged when raised: recall catches it on a routed guess and answers another way."""

    def __init__(
        self,
        message: str,
        name: str = "BroadLimitError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code, log=False)


class BroadNoTextError(NoDataError):
    """The dataset holds no document text for BROAD to read. Narrower than an empty
    graph: code or custom DataPoints may still be there for other search types."""


class DecisionSmall(BaseModel):
    """The whole dataset is shown: answer from it, or compute the answer with SQL."""

    answer: str | None = None
    sql: str | None = None


class DecisionLarge(BaseModel):
    """Only the store's schema is shown: compute the answer with SQL, or ask to read.
    ``search_queries`` are phrasings of the passage that would answer the question, used
    to find the parts read."""

    sql: str | None = None
    needs_reading: bool = False
    search_queries: list[str] = Field(default_factory=list)


@dataclass
class Unit:
    id: str
    text: str
    kind: str
    document: str


@dataclass
class BroadContext:
    route: Literal["answer", "sql", "reading"]
    # What the answer model reads; empty when the decision call already answered.
    text: str
    # How the answer was reached, appended to it by code.
    note: str
    # What the store held when the dataset was too large to load whole; empty otherwise.
    coverage: str = ""
    answer: str | None = None
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


def _read_prompt(name: str) -> str:
    prompt = read_query_prompt(name)
    if prompt is None:
        raise FileNotFoundError(f"BROAD prompt {name!r} could not be read.")
    return prompt


def _units_of(nodes: list, edges: list, names: dict[str, str]) -> list[Unit]:
    """The chunks and DLT rows among graph nodes, grouped by document, in chunk order."""
    document_of = {
        str(source): str(target)
        for source, target, relation, _ in edges
        if relation == "is_part_of"
    }
    units = []
    for node_id, props in nodes:
        document = document_of.get(str(node_id))
        if props.get("type") in BROAD_TEXT_NODE_TYPES and props.get("text") and document:
            order = (names.get(document, document), document, int(props.get("chunk_index") or 0))
            units.append((order, Unit(str(node_id), props["text"], props["type"], document)))
    return [unit for _, unit in sorted(units, key=lambda pair: pair[0])]


@dataclass
class Corpus:
    """A dataset's documents, fetched from the graph in batches of whole documents so the
    text held at once stays near ``batch_chars``. Batches are planned from each document's
    stored size, scaled by how much text a stored byte turned out to hold. The batches
    end early once the text read passes ``store_chars``: the dataset is then too large to
    load whole (``over_store``). A document larger than a batch is left out and raises
    once the rest is read, so only when the dataset would otherwise load whole."""

    graph: Any
    documents: list[str]
    names: dict[str, str]
    sizes: dict[str, int]
    batch_chars: int
    store_chars: int
    worker: BroadWorker
    # The text read so far, in characters, and the stored bytes of the documents it came from.
    read_chars: int = 0
    read_stored: int = 0

    @property
    def over_store(self) -> bool:
        return self.read_chars > self.store_chars

    def estimated_chars(self) -> int | None:
        """The whole dataset's text in characters, estimated from the text the documents
        read so far held per stored byte; None when their stored sizes are unknown."""
        if not self.read_stored:
            return None
        return round(self.read_chars * sum(self.sizes.values()) / self.read_stored)

    async def fetch(self, documents: list[str]) -> list[Unit]:
        nodes, edges = await self.graph.get_neighborhood(documents, depth=1)
        return await self.worker.run(_units_of, nodes, edges, self.names)

    async def batches(self) -> AsyncIterator[list[Unit]]:
        ratio, start = 1.0, 0
        # The first document larger than a batch, and its text in characters.
        oversized: tuple[str, int] | None = None
        while start < len(self.documents):
            batch = [self.documents[start]]
            planned = self.sizes.get(batch[0], 0) * ratio
            while start + len(batch) < len(self.documents):
                size = self.sizes.get(self.documents[start + len(batch)])
                if not size or planned + size * ratio > self.batch_chars:
                    break
                batch.append(self.documents[start + len(batch)])
                planned += size * ratio
            units = await self.fetch(batch)
            text = sum(len(unit.text) for unit in units)
            stored = sum(self.sizes.get(document, 0) for document in batch)
            self.read_chars += text
            self.read_stored += stored
            if self.over_store:
                return
            if stored:
                ratio = max(ratio, text / stored)
            start += len(batch)
            if len(batch) == 1 and text > self.batch_chars:
                oversized = oversized or (batch[0], text)
                continue
            yield units
        if oversized:
            document, text = oversized
            raise BroadLimitError(
                f"BROAD: document {self.names.get(document, document)} holds {text:,} "
                f"characters of text, more than one batch of {self.batch_chars:,} "
                "(batch_mb). Raise batch_mb in retriever_specific_config."
            )


def _documents_of(units: list[Unit]) -> list[tuple[str, list[Unit]]]:
    return [(document, list(parts)) for document, parts in groupby(units, key=lambda u: u.document)]


def _store_batch(
    store: RecordStore, units: list[Unit], names: dict[str, str], chars: int, context_chars: int
) -> tuple[list[str], int]:
    """Join and parse a corpus batch on the store's worker."""
    whole = []
    for document, parts in _documents_of(units):
        name = names.get(document, document)
        rows = [u.text for u in parts if u.kind == "DltRow"]
        if rows:
            store.add_dlt_rows(rows)
            text = "\n\n".join(rows)
        else:
            # Chunks partition the document exactly; preserve their concatenation.
            text = "".join(u.text for u in parts)
            store.add_document(name, text)
        chars += len(text)
        if chars <= context_chars:
            whole.append(f"=== Document: {name} ===\n{text}")
    return whole, chars


def _best_first(ranked: list[tuple[str, list]]) -> list[tuple[str, str]]:
    """The results of one search per collection as one list of (kind, id), best first."""
    found = [(result.score, kind, str(result.id)) for kind, results in ranked for result in results]
    found.sort(key=lambda scored: scored[0])
    return [(kind, point_id) for _, kind, point_id in found]


@dataclass
class _SimilarParts:
    """The chunks most similar to the question, taken best first until ``budget``
    characters are held. They are kept by document, because a document is parsed from
    all of its chosen chunks together."""

    budget: int
    chars: int = 0
    parts: int = 0
    by_document: dict[tuple[str, str], list[tuple[int, str, str]]] = field(default_factory=dict)

    def take(self, batch: list[tuple[str, str]], payloads: dict[tuple[str, str], dict]) -> bool:
        """Keep the chunks of one fetched batch that fit. False once one did not: the
        budget is spent, so no further batch needs fetching."""
        room = True
        for kind, point_id in batch:
            payload = payloads.get((kind, point_id)) or {}
            text = str(payload.get("text", ""))
            if not text:
                continue
            if self.chars + len(text) > self.budget:
                room = False
                continue
            self.chars += len(text)
            self.parts += 1
            document = str(payload.get("document_id") or point_id)
            name = str(payload.get("document_name") or document)
            index = int(payload.get("chunk_index") or 0)
            self.by_document.setdefault((name, document), []).append((index, text, kind))
        return room

    def fill(self, store: RecordStore) -> None:
        """Parse the chosen text into the store, a document at a time, letting go of each
        document's text once it is stored."""
        for key in sorted(self.by_document):
            pieces = sorted(self.by_document.pop(key))
            rows = [text for _, text, kind in pieces if kind == "DltRow"]
            if rows:
                store.add_dlt_rows(rows)
                continue
            # Parts that are not adjacent in their document are separated by a line break.
            text = pieces[0][1] + "".join(
                ("" if current[0] == previous[0] + 1 else "\n") + current[1]
                for previous, current in pairwise(pieces)
            )
            store.add_document(key[0], text)


def _reading_parts(ranked: list[list], tokenizer, reading_tokens: int) -> list[str]:
    parts, used, seen = [], 0, set()
    for result in _merged(ranked):
        if result.id in seen:
            continue
        seen.add(result.id)
        text = str((result.payload or {}).get("text", ""))
        tokens = len(tokenizer.extract_tokens(text))
        if text and used + tokens <= reading_tokens:
            parts.append(text)
            used += tokens
    return parts


def _merged(ranked: list[list]) -> list:
    """Several ranked lists as one: the first of each, then the second of each, ..."""
    return (
        [r for tier in zip(*ranked) for r in tier]
        if len(ranked) > 1
        else ranked[0]
        if ranked
        else []
    )


def _reads_text(sql: str | None) -> bool:
    return bool(sql) and re.search(r"\blines\b", sql, re.IGNORECASE) is not None


def _found_nothing(rows: list[tuple]) -> bool:
    """No rows, or one row of nothing but zeros and NULLs."""
    return not rows or (len(rows) == 1 and all(v in (None, 0, "", "0") for v in rows[0]))


def _result_table(columns: list[str], rows: list[tuple], limit: int) -> str:
    lines = [" | ".join(columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)


class BroadRetriever(CompletionRetriever):
    """Answers questions over a whole dataset (counts, totals, lists, lookups).

    Settings (optional, passed through ``retriever_specific_config``):

    - ``context_tokens``: a dataset up to this size is shown to the model whole; above it,
      the question is answered through the record store.
    - ``reading_tokens``: the size of the prompt the reading route fills with the parts of
      the text most similar to the question.
    - ``batch_mb``: megabytes of text held in memory at once while the store is built.
    - ``store_mb``: megabytes (million characters) of text the in-memory SQLite store holds
      (default 512). A dataset with up to this much text is loaded whole. A larger one is
      represented by the parts most similar to the question, up to the same size, and the
      answer says so. The text is measured as it is read, so a dataset over the limit is
      read up to the limit once before its most similar parts are loaded. This is a
      coverage threshold, not a bound on process RAM.
    """

    def __init__(
        self,
        context_tokens: int = BROAD_CONTEXT_TOKENS,
        reading_tokens: int = BROAD_READING_TOKENS,
        batch_mb: float = BROAD_BATCH_MB,
        store_mb: float = BROAD_STORE_MB,
        **kwargs,
    ):
        super().__init__(**kwargs)
        settings = {
            "context_tokens": context_tokens,
            "reading_tokens": reading_tokens,
            "batch_mb": batch_mb,
            "store_mb": store_mb,
        }
        for name, value in settings.items():
            # bool is an int subclass; True must not pass as 1.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"BROAD {name} must be a number, not {value!r}.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"BROAD {name} must be a positive, finite number, not {value!r}.")
        self.context_tokens = context_tokens
        self.reading_tokens = reading_tokens
        self.batch_chars = int(batch_mb * 1_000_000)
        self.store_chars = int(store_mb * 1_000_000)
        self.tokenizer = TikTokenTokenizer()

    async def get_retrieved_objects(self, query: str) -> BroadContext:
        engine = await get_unified_engine()
        foreign_keys = await self.dlt_foreign_keys(engine.graph)
        async with BroadWorker(RecordStore) as store:
            await store.run(store.store.add_dlt_tables, foreign_keys)
            corpus = await self.open_corpus(engine.graph, store)
            whole, chars = await self.build(store, corpus)
            if not corpus.over_store:
                if not chars:
                    raise BroadNoTextError("BROAD found no document text in this dataset.")
                fits = (
                    bool(whole)
                    and len(await store.run(self.tokenizer.extract_tokens, whole))
                    <= self.context_tokens
                )
                logger.info("BROAD dataset: %d characters, shown whole: %s", chars, fits)
                return await self.decide(query, store, whole if fits else None)
        # That store holds the first documents by name, not the ones the question needs,
        # so the most similar parts are loaded into a new one.
        async with BroadWorker(RecordStore) as store:
            await store.run(store.store.add_dlt_tables, foreign_keys)
            chars, parts = await self.build_most_similar(store, engine.vector, query)
            if not chars:
                raise BroadNoTextError("BROAD found no document text in this dataset.")
            total = corpus.estimated_chars()
            of_total = f" of about {total / 1e6:,.1f} MB" if total is not None else ""
            coverage = (
                f"the {parts:,} parts of the dataset most similar to the question "
                f"({chars / 1e6:.1f} MB{of_total} of text; the rest was not searched)"
            )
            logger.warning(
                "BROAD dataset holds more text than store_mb (%.1f MB): the store holds %s. "
                "Raise store_mb in retriever_specific_config to search more.",
                self.store_chars / 1e6,
                coverage,
            )
            found = await self.decide(query, store, None)
            found.coverage = coverage
            if found.route == "sql":
                found.note = found.note.replace("over the whole dataset", f"over {coverage}")
            return found

    # --- loading ----------------------------------------------------------------------

    async def dlt_foreign_keys(self, graph_engine) -> dict[str, list[str]]:
        """Each DLT table's foreign keys, read from its SchemaTable nodes.

        Rows are shown as one table per table name, so the keys of every source's
        table of that name are listed together. The keys are JSON cognee wrote
        itself, so a node whose keys do not parse is a bug and raises.
        """
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["SchemaTable"]}])
        foreign_keys: dict[str, list[str]] = {}
        for _, props in nodes:
            lines = [
                f"{fk['column']} references {fk['ref_table']}.{fk['ref_column']}"
                for fk in json.loads(props["foreign_keys"])
            ]
            listed = foreign_keys.setdefault(str(props["name"]), [])
            listed.extend(line for line in lines if line not in listed)
        return foreign_keys

    async def open_corpus(self, graph_engine, worker: BroadWorker) -> Corpus:
        """The dataset's documents, their names and stored sizes; text is fetched later."""
        document_types = [cls.__name__ for cls in get_all_subclasses(Document)]
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": document_types}])
        names = {str(node_id): str(props.get("name") or node_id) for node_id, props in nodes}
        documents = sorted(names, key=lambda d: (names[d], d))
        return Corpus(
            graph=graph_engine,
            documents=documents,
            names=names,
            sizes=await self.document_sizes(documents),
            batch_chars=self.batch_chars,
            store_chars=self.store_chars,
            worker=worker,
        )

    async def document_sizes(self, documents: list[str]) -> dict[str, int]:
        """Each document's stored size in bytes (a document's graph id is its data id)."""
        sizes: dict[str, int] = {}
        async with get_relational_engine().get_async_session() as session:
            for start in range(0, len(documents), 5_000):
                ids = [UUID(document) for document in documents[start : start + 5_000]]
                rows = await session.execute(
                    select(Data.id, Data.data_size).where(Data.id.in_(ids))
                )
                sizes.update({str(data_id): size or 0 for data_id, size in rows})
        return sizes

    async def build(self, store: BroadWorker, corpus: Corpus) -> tuple[str, int]:
        """Fill the store batch by batch; return the dataset's text when it may fit one
        prompt (else "") and its size in characters. The batches stop once the text passes
        store_mb (``corpus.over_store``); the store is then incomplete."""
        whole: list[str] = []
        chars = 0
        async for units in corpus.batches():
            texts, chars = await store.run(
                _store_batch,
                store.store,
                units,
                corpus.names,
                chars,
                self.context_tokens * _MAX_CHARS_PER_TOKEN,
            )
            whole.extend(texts)
        fits = chars <= self.context_tokens * _MAX_CHARS_PER_TOKEN
        return ("\n\n".join(whole) if fits else ""), chars

    async def build_most_similar(self, store: BroadWorker, vector, query: str) -> tuple[int, int]:
        """Fill the store with the chunks most similar to the question, up to store_mb of
        text. One vector search per text collection ranks the chunks without their text;
        the text is then fetched best first, a batch at a time, until the size is reached,
        so what is read does not grow with the dataset. A chunk's payload names its
        document and position, so no graph call is needed. Returns the characters and
        parts loaded."""
        limit = max(1, self.store_chars // _MIN_CHUNK_CHARS)
        ranked = []
        for kind in BROAD_TEXT_NODE_TYPES:
            try:
                results = await vector.search(f"{kind}_text", query, query_vector=None, limit=limit)
            except CollectionNotFoundError:
                continue
            ranked.append((kind, results))
        order = await store.run(_best_first, ranked)
        del ranked
        similar = _SimilarParts(budget=self.store_chars)
        for start in range(0, len(order), _SIMILAR_FETCH_CHUNKS):
            batch = order[start : start + _SIMILAR_FETCH_CHUNKS]
            payloads = {}
            for kind in dict.fromkeys(kind for kind, _ in batch):
                ids = [point_id for batch_kind, point_id in batch if batch_kind == kind]
                for result in await vector.retrieve(f"{kind}_text", ids):
                    payloads[(kind, str(result.id))] = result.payload
            if not await store.run(similar.take, batch, payloads):
                break
        await store.run(similar.fill, store.store)
        return similar.chars, similar.parts

    # --- deciding and answering ---------------------------------------------------------

    async def decide(self, query: str, store: BroadWorker, whole: str | None) -> BroadContext:
        schema = await store.run(store.store.describe, query)
        if whole is not None:
            prompt = (
                f"Question: {query}\n\nThe whole dataset:\n{whole}\n\nIts record store:\n{schema}"
            )
            decision = await self._ask(prompt, "broad_decide_small.txt", DecisionSmall)
            if decision.sql is None:
                if not decision.answer:
                    raise BroadQueryError("BROAD decision returned neither an answer nor a query")
                return BroadContext(
                    "answer", "", "(Answered from the whole dataset.)", answer=decision.answer
                )
        else:
            prompt = f"Question: {query}\n\nThe record store of the dataset:\n{schema}"
            decision = await self._ask(prompt, "broad_decide_large.txt", DecisionLarge)
            phrasings = [query, *decision.search_queries]
            if decision.sql is None or decision.needs_reading:
                return await self.read_most_similar(query, store, phrasings)
            found = await self.run_query(query, prompt, decision.sql, store, False)
            if found.route == "sql" and _reads_text(found.sql) and _found_nothing(found.rows):
                # A word match that finds nothing does not show the text lacks the answer.
                logger.info("BROAD query over the text found nothing; reading instead")
                reading = await self.read_most_similar(query, store, phrasings)
                reading.note = f"(A query over the text found nothing: {found.sql}) " + reading.note
                return reading
            return found
        return await self.run_query(query, prompt, decision.sql, store, whole is not None)

    async def run_query(
        self, query: str, prompt: str, sql: str, store: BroadWorker, small: bool
    ) -> BroadContext:
        """Run the query; a failing one is sent back once with the error."""
        try:
            columns, rows, cut = await store.run(store.store.run, sql)
        except BroadQueryError as error:
            logger.info("BROAD query failed (%s); asking once more", error)
            retry = (
                f"{prompt}\n\nYour query failed: {error}\nQuery: {sql}\nReturn a corrected query."
            )
            model = DecisionSmall if small else DecisionLarge
            name = "broad_decide_small.txt" if small else "broad_decide_large.txt"
            decision = await self._ask(retry, name, model)
            if small and not decision.sql and decision.answer:
                note = "(Answered from the whole dataset.)"
                return BroadContext("answer", "", note, answer=decision.answer)
            if not decision.sql:
                raise BroadQueryError(f"BROAD could not write a working query: {error}") from error
            sql = decision.sql
            columns, rows, cut = await store.run(store.store.run, sql)
        logger.info("BROAD query: %s -> %d rows", sql, len(rows))
        shown = await store.run(_result_table, columns, rows, BROAD_RESULT_ROWS_SHOWN)
        more = " (more rows exist; only the first were returned)" if cut else ""
        text = (
            f"Question: {query}\n\nSQL computed over every record of the dataset:\n{sql}\n\n"
            f"Result ({len(rows)} rows{more}):\n{shown}"
        )
        return BroadContext(
            "sql",
            text,
            f"(Computed by a SQL query over the whole dataset: {sql})",
            sql=sql,
            columns=columns,
            rows=rows,
        )

    async def read_most_similar(
        self, query: str, store: BroadWorker, phrasings: list[str] | None = None
    ) -> BroadContext:
        """Fill one prompt with the chunks most similar to the question: one bounded vector
        search per phrasing, their results merged rank by rank."""
        limit = max(1, self.reading_tokens // _MIN_CHUNK_TOKENS)
        vector = (await get_unified_engine()).vector
        try:
            ranked = [
                await vector.search(
                    "DocumentChunk_text", text, query_vector=None, limit=limit, include_payload=True
                )
                for text in dict.fromkeys(t.strip() for t in (phrasings or [query]) if t.strip())
            ]
        except CollectionNotFoundError as error:
            raise BroadLimitError(
                "BROAD: the question needs the text read, the dataset is larger than "
                "context_tokens, and it has no chunk embeddings to choose the most relevant "
                "parts. Raise context_tokens in retriever_specific_config."
            ) from error
        parts = await store.run(_reading_parts, ranked, self.tokenizer, self.reading_tokens)
        logger.warning(
            "BROAD cannot read the whole dataset: only the %d chunks most similar to the "
            "question fit reading_tokens (%d). To read more, raise reading_tokens in "
            "retriever_specific_config.",
            len(parts),
            self.reading_tokens,
        )
        note = (
            f"(Answered from the {len(parts)} parts of the dataset most similar to the "
            "question; the rest of the dataset was not read.)"
        )
        text = "\n\n---\n\n".join(parts)
        return BroadContext("reading", f"Question: {query}\n\nParts of the dataset:\n{text}", note)

    async def _ask(self, text_input: str, prompt: str, response_model: type[Any]) -> Any:
        return await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=_read_prompt(prompt),
            response_model=response_model,
        )

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        context: BroadContext = retrieved_objects
        return context.text or context.answer or ""

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Any,
        context: Any | None = None,
        effective_query: str | None = None,
        turn_preparation=None,
    ) -> list[Any]:
        """The decision's own answer, or a completion over the result or the parts read;
        code appends how it was reached, and a result of several rows in full."""
        found: BroadContext = retrieved_objects
        if found.answer is not None:
            completions: list[Any] = [found.answer]
        else:
            completions = await super().get_completion_from_context(
                query,
                retrieved_objects,
                context=context,
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
        tail = [found.note]
        if len(found.rows) > 1:
            tail.append(f"Result:\n{_result_table(found.columns, found.rows, len(found.rows))}")
        suffix = "\n\n".join(tail)
        return [f"{c.rstrip()}\n\n{suffix}" if isinstance(c, str) else c for c in completions]

    def extract_context_object_ids(self, retrieved_objects: Any) -> dict[str, list[str]] | None:
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        return []

    async def append_references(self, completions: list[Any], retrieved_objects: Any) -> list[Any]:
        return completions
