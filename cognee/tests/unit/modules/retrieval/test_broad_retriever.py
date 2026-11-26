"""BROAD's routes: a small dataset shown whole, a large one queried with SQL, and reading
the most similar chunks when a query cannot express the question (SDK-324)."""

import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import (
    BroadLimitError,
    BroadNoTextError,
    BroadRetriever,
    DecisionLarge,
    DecisionSmall,
)
from cognee.modules.retrieval.broad_store import BroadQueryError, RecordStore


class _Graph:
    """One document per text, each a single chunk."""

    def __init__(self, texts: dict[str, str]):
        self.nodes = []
        self.edges = []
        for n, (name, text) in enumerate(texts.items()):
            self.nodes += [
                (f"d{n}", {"type": "TextDocument", "name": name}),
                (f"c{n}", {"type": "DocumentChunk", "chunk_index": 0, "text": text}),
            ]
            self.edges.append((f"c{n}", f"d{n}", "is_part_of", {}))

    async def get_filtered_graph_data(self, filters):
        types = filters[0]["type"]
        return [node for node in self.nodes if node[1]["type"] in types], []

    async def get_neighborhood(self, ids, depth=1, edge_types=None):
        self.lookups = getattr(self, "lookups", []) + [(list(ids), edge_types)]
        keep = set(ids)
        keep |= {s for s, t, _, _ in self.edges if t in ids} | {
            t for s, t, _, _ in self.edges if s in ids
        }
        return [n for n in self.nodes if n[0] in keep], [e for e in self.edges if e[0] in keep]


def _use(monkeypatch, texts: dict[str, str], vector=None, size: int = 10):
    engine = SimpleNamespace(graph=_Graph(texts), vector=vector)

    async def unified():
        return engine

    async def sizes(self, documents):
        return {document: size for document in documents}

    monkeypatch.setattr(broad_retriever, "get_unified_engine", unified)
    monkeypatch.setattr(BroadRetriever, "document_sizes", sizes)


def _llm(monkeypatch, answers: list):
    """The LLM returns ``answers`` in order; the calls are recorded."""
    calls = []

    async def fake(text_input, system_prompt, response_model, **kwargs):
        calls.append((response_model, text_input))
        return answers.pop(0)

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)
    return calls


def _scores(rows: int) -> str:
    return "\n".join(["player,goals"] + [f"P{i},{i % 3}" for i in range(rows)])


@pytest.mark.asyncio
async def test_a_small_dataset_is_shown_whole_and_answered_in_one_call(monkeypatch):
    _use(monkeypatch, {"notes": "The meeting moved to Tuesday."})
    calls = _llm(monkeypatch, [DecisionSmall(answer="Tuesday.")])

    found = await BroadRetriever().get_retrieved_objects("When is the meeting?")
    [answer] = await BroadRetriever().get_completion_from_context("q", found)

    assert found.route == "answer" and len(calls) == 1
    assert "The meeting moved to Tuesday." in calls[0][1]
    assert answer.startswith("Tuesday.") and "whole dataset" in answer


@pytest.mark.asyncio
async def test_a_small_dataset_can_still_be_counted_by_sql(monkeypatch):
    _use(monkeypatch, {"scores": _scores(30)})
    _llm(monkeypatch, [DecisionSmall(sql='SELECT COUNT(*) FROM "scores" WHERE "goals" = 2')])

    found = await BroadRetriever().get_retrieved_objects("How many players scored twice?")

    assert found.route == "sql" and found.rows == [(10,)]


@pytest.mark.asyncio
async def test_a_large_dataset_is_answered_from_the_schema_by_sql(monkeypatch):
    _use(monkeypatch, {"scores": _scores(3_000)})
    calls = _llm(
        monkeypatch,
        [DecisionLarge(sql='SELECT "goals", COUNT(*) FROM "scores" GROUP BY 1 ORDER BY 1')],
    )

    found = await BroadRetriever(context_tokens=200).get_retrieved_objects("Goals per player?")

    assert found.route == "sql" and found.rows == [(0, 1000), (1, 1000), (2, 1000)]
    assert calls[0][0] is DecisionLarge and "P2999" not in calls[0][1]  # schema, not the text


@pytest.mark.asyncio
async def test_a_failing_query_is_sent_back_once_with_its_error(monkeypatch):
    _use(monkeypatch, {"scores": _scores(3_000)})
    calls = _llm(
        monkeypatch,
        [
            DecisionLarge(sql='SELECT COUNT(*) FROM "nope"'),
            DecisionLarge(sql='SELECT COUNT(*) FROM "scores"'),
        ],
    )

    found = await BroadRetriever(context_tokens=200).get_retrieved_objects("How many?")

    assert found.rows == [(3000,)] and "no such table" in calls[1][1]


@pytest.mark.asyncio
async def test_on_a_small_dataset_the_retry_may_answer_instead(monkeypatch):
    _use(monkeypatch, {"notes": "Three apples."})
    _llm(monkeypatch, [DecisionSmall(sql="SELECT x FROM nope"), DecisionSmall(answer="3")])

    found = await BroadRetriever().get_retrieved_objects("How many apples?")

    assert (found.route, found.answer) == ("answer", "3")


@pytest.mark.asyncio
async def test_a_decision_with_neither_answer_nor_query_is_a_query_error(monkeypatch):
    _use(monkeypatch, {"notes": "Three apples."})
    _llm(monkeypatch, [DecisionSmall()])

    with pytest.raises(BroadQueryError, match="neither an answer nor a query"):
        await BroadRetriever().get_retrieved_objects("How many apples?")


@pytest.mark.asyncio
async def test_a_dataset_without_document_text_is_a_no_text_error(monkeypatch):
    _use(monkeypatch, {})

    with pytest.raises(BroadNoTextError) as error:
        await BroadRetriever().get_retrieved_objects("How many apples?")

    assert error.value.status_code == 404


class _Vector:
    """Every search returns the same nine chunks; the searches are recorded."""

    def __init__(self):
        self.searches = []

    async def search(self, collection, query_text, query_vector=None, limit=None, **kw):
        self.searches.append((collection, query_text, limit, kw.get("include_payload")))
        return [
            SimpleNamespace(id=f"c{n}", payload={"text": f"part {n} " + "word " * 30})
            for n in range(9)
        ]


@pytest.mark.asyncio
async def test_reading_fills_one_prompt_from_bounded_vector_searches(monkeypatch):
    vector = _Vector()
    _use(monkeypatch, {"story": "word " * 5_000}, vector=vector)
    _llm(monkeypatch, [DecisionLarge(needs_reading=True, search_queries=["a storm", "a wreck"])])

    found = await BroadRetriever(context_tokens=100, reading_tokens=100).get_retrieved_objects(
        "What happens?"
    )

    assert found.route == "reading"
    assert [(q, limit, p) for _, q, limit, p in vector.searches] == [
        ("What happens?", 1, True),
        ("a storm", 1, True),
        ("a wreck", 1, True),
    ]
    assert found.text.count("part 0") == 1 and "part 5" not in found.text  # merged, no repeats
    assert "most similar" in found.note


@pytest.mark.asyncio
async def test_a_text_query_that_finds_nothing_falls_back_to_reading(monkeypatch):
    vector = _Vector()
    _use(monkeypatch, {"story": "word " * 5_000}, vector=vector)
    _llm(
        monkeypatch,
        [
            DecisionLarge(
                sql="SELECT COUNT(*) FROM lines WHERE text LIKE '%married%'",
                search_queries=["a wedding"],
            )
        ],
    )

    found = await BroadRetriever(context_tokens=100, reading_tokens=100).get_retrieved_objects(
        "Who married?"
    )

    assert found.route == "reading" and len(vector.searches) == 2
    assert (
        found.note.startswith("(A query over the text found nothing:")
        and "most similar" in found.note
    )


@pytest.mark.asyncio
async def test_a_table_query_that_finds_nothing_is_an_answer(monkeypatch):
    _use(monkeypatch, {"scores": _scores(3_000)})
    _llm(monkeypatch, [DecisionLarge(sql='SELECT COUNT(*) FROM "scores" WHERE "goals" = 9')])

    found = await BroadRetriever(context_tokens=200).get_retrieved_objects("Nine goals?")

    assert found.route == "sql" and found.rows == [(0,)]


@pytest.mark.asyncio
async def test_a_result_of_several_rows_is_appended_in_full(monkeypatch):
    _use(monkeypatch, {"scores": _scores(3_000)})
    _llm(monkeypatch, [DecisionLarge(sql='SELECT "player" FROM "scores" LIMIT 3')])
    retriever = BroadRetriever(context_tokens=200)
    found = await retriever.get_retrieved_objects("Name three players.")

    async def phrase(self, query, retrieved_objects, context=None, **kwargs):
        return ["Three players."]

    monkeypatch.setattr(broad_retriever.CompletionRetriever, "get_completion_from_context", phrase)
    [answer] = await retriever.get_completion_from_context("q", found)

    assert "P0\nP1\nP2" in answer and "SQL query" in answer


def test_the_settings_are_positive():
    with pytest.raises(ValueError):
        BroadRetriever(context_tokens=0)
    with pytest.raises(ValueError):
        BroadRetriever(reading_tokens=0)


@pytest.mark.parametrize("setting", ["context_tokens", "reading_tokens", "batch_mb", "store_mb"])
@pytest.mark.parametrize("value", ["abc", None, True, float("inf"), float("nan"), -1])
def test_a_setting_that_is_not_a_positive_finite_number_is_named(setting, value):
    with pytest.raises(ValueError, match=f"BROAD {setting} must be"):
        BroadRetriever(**{setting: value})


class _OneChunkVector:
    """Ranks the given chunks best first and serves their text; each is its own document."""

    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    async def search(self, collection, *args, **kwargs):
        if collection != "DocumentChunk_text":
            raise CollectionNotFoundError("none")
        return [
            SimpleNamespace(id=name, score=n / 10, payload=None)
            for n, name in enumerate(self.texts)
        ]

    async def retrieve(self, collection, ids):
        return [
            SimpleNamespace(
                id=name,
                score=0,
                payload={"text": self.texts[name], "document_id": name, "document_name": name},
            )
            for name in ids
        ]


def test_the_default_store_holds_512_mb_of_text():
    assert BroadRetriever().store_chars == 512_000_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_chars", "expected_rows", "partial"), [(22, 2, False), (21, 2, False), (20, 1, True)]
)
@pytest.mark.parametrize("stored_bytes", [1, 10, 1_000_000_000])
async def test_store_mb_is_compared_with_the_text_not_the_stored_bytes(
    monkeypatch, store_chars, expected_rows, partial, stored_bytes
):
    """21 characters of text: loaded whole into a store of 21, whatever the files weigh
    (a CSV holds more text than bytes, a PDF far less)."""
    texts = {"a": "Alpha only.", "b": "Beta only."}
    _use(monkeypatch, texts, vector=_OneChunkVector(texts), size=stored_bytes)
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])
    retriever = BroadRetriever(context_tokens=1)
    retriever.store_chars = store_chars

    found = await retriever.get_retrieved_objects("How many lines?")

    assert found.rows == [(expected_rows,)]
    assert bool(found.coverage) is partial


@pytest.mark.asyncio
async def test_a_store_just_under_the_text_holds_nearly_all_of_it(monkeypatch):
    """One character short of the dataset costs one part, not most of the store: the
    threshold and the size of the most similar parts are the same characters."""
    texts = {f"doc{n}": f"row {n:02d}\n" for n in range(40)}  # 7 characters each, 280 in all
    _use(monkeypatch, texts, vector=_OneChunkVector(texts), size=1)  # 40 bytes stored
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])
    retriever = BroadRetriever(context_tokens=1)
    retriever.store_chars = 279

    found = await retriever.get_retrieved_objects("How many rows?")

    assert found.rows == [(39,)]
    assert "39 parts of the dataset most similar" in found.coverage


@pytest.mark.asyncio
async def test_partial_coverage_states_the_text_held_and_the_estimated_text_of_the_dataset(
    monkeypatch,
):
    """Both sizes are characters of text. The total is estimated from what the documents
    read before the store was full held per stored byte."""
    document = "".join(f"row {n:05d}\n" for n in range(1_000))  # 10,000 characters
    texts = {f"doc{n}": document for n in range(40)}  # 0.4 MB of text
    _use(monkeypatch, texts, vector=_OneChunkVector(texts), size=2_500)  # 0.1 MB stored
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])

    found = await BroadRetriever(
        context_tokens=1, batch_mb=0.05, store_mb=0.1
    ).get_retrieved_objects("How many lines?")

    assert found.rows == [(10_000,)]
    assert "(0.1 MB of about 0.4 MB of text; the rest was not searched)" in found.coverage
    assert found.coverage in found.note


@pytest.mark.asyncio
async def test_a_document_over_a_batch_in_a_dataset_over_the_store_is_not_an_error(monkeypatch):
    """The store limit is met first: the most similar parts are loaded, and batch_mb,
    which bounds the whole load only, does not raise."""
    texts = {"a": "alpha line\n" * 3}  # 33 characters, over both limits
    _use(monkeypatch, texts, vector=_OneChunkVector({"c": "alpha line\n"}), size=33)
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])

    found = await BroadRetriever(
        context_tokens=1, batch_mb=0.00002, store_mb=0.00003
    ).get_retrieved_objects("How many lines?")

    assert found.rows == [(1,)]
    assert found.coverage


@pytest.mark.asyncio
async def test_a_document_over_a_batch_before_the_store_limit_is_met_is_not_an_error(monkeypatch):
    """The large document comes first by name, while the text read is still under
    store_mb; the documents after it take the dataset over, so the most similar parts
    are loaded."""
    line = "alpha line\n"  # 11 characters
    texts = {"a": line * 4, **{f"b{n:02d}": line for n in range(20)}}  # 264 characters
    _use(monkeypatch, texts, vector=_OneChunkVector({"c": line}), size=30)
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])

    found = await BroadRetriever(
        context_tokens=1, batch_mb=0.00003, store_mb=0.0001
    ).get_retrieved_objects("How many lines?")

    assert found.rows == [(1,)]
    assert found.coverage


@pytest.mark.asyncio
async def test_a_document_over_a_batch_in_a_dataset_under_the_store_names_batch_mb(monkeypatch):
    line = "alpha line\n"
    _use(monkeypatch, {"a": line * 4, "b": line}, size=30)  # 55 characters

    with pytest.raises(BroadLimitError, match="document a holds 44 .*batch_mb"):
        await BroadRetriever(
            context_tokens=1, batch_mb=0.00003, store_mb=0.0001
        ).get_retrieved_objects("How many lines?")


def test_a_broad_limit_error_is_not_logged(caplog):
    """recall catches it on a routed guess and answers another way."""
    with caplog.at_level("DEBUG"):
        BroadLimitError("BROAD: over a limit.")

    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, RuntimeError("decision failed"), asyncio.CancelledError()])
async def test_record_store_is_closed_after_success_failure_or_cancellation(monkeypatch, error):
    _use(monkeypatch, {"notes": "The meeting moved to Tuesday."})
    closed = []

    class TrackedStore(RecordStore):
        def close(self):
            super().close()
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                self.connection.execute("SELECT 1")
            closed.append(True)

    async def decide(self, *args):
        if error is not None:
            raise error
        return DecisionSmall(answer="Tuesday.")

    monkeypatch.setattr(broad_retriever, "RecordStore", TrackedStore)
    monkeypatch.setattr(BroadRetriever, "_ask", decide)

    if error is None:
        found = await BroadRetriever().get_retrieved_objects("When is the meeting?")
        assert found.answer == "Tuesday."
    else:
        with pytest.raises(type(error)):
            await BroadRetriever().get_retrieved_objects("When is the meeting?")

    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["add_document", "describe", "run"])
async def test_server_handles_other_requests_during_broad_cpu_work(monkeypatch, stage):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    _use(monkeypatch, {"notes": "One line."})
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*) FROM lines")])
    started, release = threading.Event(), threading.Event()
    threads = []
    main_thread = threading.get_ident()

    class BusyStore(RecordStore):
        def __init__(self):
            threads.append(threading.get_ident())
            super().__init__()

        def close(self):
            threads.append(threading.get_ident())
            super().close()

    original = getattr(RecordStore, stage)

    def busy(self, *args):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(5), "The event loop could not release the BROAD worker"
        return original(self, *args)

    monkeypatch.setattr(BusyStore, stage, busy)
    monkeypatch.setattr(broad_retriever, "RecordStore", BusyStore)
    app = FastAPI()

    @app.get("/broad")
    async def broad():
        found = await BroadRetriever(context_tokens=1).get_retrieved_objects("Count lines")
        return {"rows": found.rows}

    @app.get("/health")
    async def health():
        return {"ok": True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request = asyncio.create_task(client.get("/broad"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            response = await asyncio.wait_for(client.get("/health"), timeout=1)
            assert response.json() == {"ok": True}
            assert not request.done()
        finally:
            release.set()
            response = await request
    assert response.json() == {"rows": [[1]]}
    assert len(threads) == 3 and len(set(threads)) == 1 and threads[0] != main_thread


@pytest.mark.asyncio
async def test_cancellation_interrupts_running_sql_and_closes_on_its_worker(monkeypatch):
    _use(monkeypatch, {"notes": "One line."})
    started = threading.Event()
    closed = []
    sql = (
        "WITH RECURSIVE numbers(n) AS (VALUES(1) UNION ALL "
        "SELECT n+1 FROM numbers WHERE n < 100000000) SELECT SUM(observe(n)) FROM numbers"
    )
    _llm(monkeypatch, [DecisionLarge(sql=sql)])

    class SlowStore(RecordStore):
        def __init__(self):
            super().__init__()
            self.thread = threading.get_ident()

            def observe(value):
                started.set()
                return value

            self.connection.create_function("observe", 1, observe)

        def close(self):
            assert threading.get_ident() == self.thread
            super().close()
            closed.append(True)

    monkeypatch.setattr(broad_retriever, "RecordStore", SlowStore)
    request = asyncio.create_task(
        BroadRetriever(context_tokens=1).get_retrieved_objects("Count lines")
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
    finally:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=2)
    assert closed == [True]


@pytest.mark.asyncio
async def test_dlt_rows_are_loaded_through_the_worker(monkeypatch):
    _use(monkeypatch, {"orders": "Table: orders\nRow Data:\n  id: 7\n  amount: 12"})
    engine = await broad_retriever.get_unified_engine()
    engine.graph.nodes[1][1]["type"] = "DltRow"
    _llm(monkeypatch, [DecisionLarge(sql="SELECT id, amount FROM orders")])
    found = await BroadRetriever(context_tokens=1).get_retrieved_objects("Orders")
    assert found.rows == [(7, 12)]


@pytest.mark.asyncio
async def test_a_dataset_over_store_mb_is_represented_by_its_most_similar_parts(monkeypatch):
    texts = {"a": "alpha line\n" * 3, "b": "beta line\n" * 3}

    class Vector:
        async def search(self, collection, query_text, query_vector=None, limit=None, **kw):
            if collection != "DocumentChunk_text":
                raise CollectionNotFoundError("none")
            return [  # b is the better match (lower distance); the 40-char store holds one chunk
                SimpleNamespace(id="b", score=0.1, payload=None),
                SimpleNamespace(id="a", score=0.5, payload=None),
            ]

        async def retrieve(self, collection, ids):
            return [  # in no particular order, as the adapters return them
                SimpleNamespace(
                    id=doc,
                    score=0,
                    payload={
                        "text": texts[doc],
                        "document_id": f"d-{doc}",
                        "document_name": doc,
                        "chunk_index": 0,
                    },
                )
                for doc in sorted(ids)
            ]

    _use(monkeypatch, texts, vector=Vector(), size=30)  # 66 characters > 40-char store
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*), MIN(document) FROM lines")])

    found = await BroadRetriever(store_mb=0.00004).get_retrieved_objects("beta?")

    assert found.route == "sql" and found.rows == [(3, "b")]  # only b's three lines were loaded
    assert "1 parts of the dataset most similar" in found.coverage
    assert "over the 1 parts of the dataset most similar" in found.note


@pytest.mark.asyncio
async def test_the_most_similar_parts_are_ranked_without_text_and_fetched_until_full(monkeypatch):
    """The search returns ids and scores only; text is fetched best first in batches, and
    no batch is fetched after the store is full."""
    texts = {f"doc{n}": f"line {n}\n" for n in range(6)}  # 7 characters each
    searches, fetched = [], []

    class Vector:
        async def search(self, collection, query_text, query_vector=None, limit=None, **kw):
            searches.append((collection, kw.get("include_payload", False)))
            if collection != "DocumentChunk_text":
                raise CollectionNotFoundError("none")
            return [
                SimpleNamespace(id=doc, score=n / 10, payload=None) for n, doc in enumerate(texts)
            ]

        async def retrieve(self, collection, ids):
            fetched.append((collection, list(ids)))
            return [
                SimpleNamespace(
                    id=doc,
                    score=0,
                    payload={"text": texts[doc], "document_id": doc, "document_name": doc},
                )
                for doc in reversed(ids)
            ]

    _use(monkeypatch, texts, vector=Vector(), size=30)
    monkeypatch.setattr(broad_retriever, "_SIMILAR_FETCH_CHUNKS", 2)
    _llm(monkeypatch, [DecisionLarge(sql="SELECT document FROM lines ORDER BY document")])

    # 42 characters > the 20-character store, which holds two 7-character chunks.
    found = await BroadRetriever(store_mb=0.00002).get_retrieved_objects("which lines?")

    assert searches == [("DocumentChunk_text", False), ("DltRow_text", False)]
    # The second batch holds the chunk that no longer fits; the third is never fetched.
    assert fetched == [
        ("DocumentChunk_text", ["doc0", "doc1"]),
        ("DocumentChunk_text", ["doc2", "doc3"]),
    ]
    assert found.rows == [("doc0",), ("doc1",)]
    assert "2 parts of the dataset most similar" in found.coverage


def test_broad_reports_no_structured_evidence_as_an_empty_list():
    retriever = BroadRetriever.__new__(BroadRetriever)
    assert retriever.get_context_evidence({"route": "small"}, dataset_id="d") == []


class _SchemaGraph:
    """SchemaTable nodes only, as the DLT route writes them."""

    def __init__(self, tables: list[tuple[str, str]]):
        self.nodes = [
            (f"t{n}", {"type": "SchemaTable", "name": name, "foreign_keys": foreign_keys})
            for n, (name, foreign_keys) in enumerate(tables)
        ]

    async def get_filtered_graph_data(self, filters):
        return [node for node in self.nodes if node[1]["type"] in filters[0]["type"]], []


@pytest.mark.asyncio
async def test_dlt_foreign_keys_are_shown_with_their_table():
    """Keys of every table of one name are listed together, once each."""
    from cognee.modules.retrieval.broad_store import RecordStore

    orders_fk = '[{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}]'
    other_orders_fk = '[{"column": "shop_id", "ref_table": "shops", "ref_column": "id"}]'
    graph = _SchemaGraph(
        [("orders", orders_fk), ("orders", other_orders_fk), ("orders", orders_fk), ("notes", "[]")]
    )
    foreign_keys = await BroadRetriever().dlt_foreign_keys(graph)
    assert foreign_keys == {
        "orders": ["customer_id references customers.id", "shop_id references shops.id"],
        "notes": [],
    }

    store = RecordStore()
    try:
        store.add_dlt_tables(foreign_keys)
        store.add_dlt_rows(["Table: orders\nRow Data:\n  id: 7\n  customer_id: 3"])
        assert "customer_id references customers.id; shop_id references shops.id" in (
            store.describe("orders")
        )
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("foreign_keys", "error"),
    [("not json", ValueError), ('[{"column": "a"}]', KeyError), (None, TypeError)],
)
async def test_a_table_node_with_unreadable_keys_raises(foreign_keys, error):
    """cognee writes these keys itself, so one that does not parse is a bug, not data to skip."""
    with pytest.raises(error):
        await BroadRetriever().dlt_foreign_keys(_SchemaGraph([("broken", foreign_keys)]))
