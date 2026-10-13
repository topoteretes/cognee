"""BROAD's routes: a small dataset shown whole, a large one queried with SQL, and reading
the most similar chunks when a query cannot express the question (SDK-324)."""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import (
    BroadRetriever,
    DecisionLarge,
    DecisionSmall,
)


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


@pytest.mark.asyncio
async def test_a_dataset_over_store_mb_is_represented_by_its_most_similar_parts(monkeypatch):
    texts = {"a": "alpha line\n" * 3, "b": "beta line\n" * 3}

    class Vector:
        async def search(self, collection, query_text, query_vector=None, limit=None, **kw):
            if collection != "DocumentChunk_text":
                raise CollectionNotFoundError("none")
            payload = lambda doc: {
                "text": texts[doc],
                "document_id": f"d-{doc}",
                "document_name": doc,
                "chunk_index": 0,
            }
            return [  # b is the better match (lower distance); the 40-char store holds one chunk
                SimpleNamespace(id="c1", score=0.1, payload=payload("b")),
                SimpleNamespace(id="c0", score=0.5, payload=payload("a")),
            ]

    _use(monkeypatch, texts, vector=Vector(), size=30)  # 60 bytes stored > 40-char store

    async def no_dataset(self):
        return None  # fall back to the documents' stored sizes

    monkeypatch.setattr(BroadRetriever, "dataset_size", no_dataset)
    _llm(monkeypatch, [DecisionLarge(sql="SELECT COUNT(*), MIN(document) FROM lines")])

    found = await BroadRetriever(store_mb=0.00004).get_retrieved_objects("beta?")

    assert found.route == "sql" and found.rows == [(3, "b")]  # only b's three lines were loaded
    assert "1 parts of the dataset most similar" in found.coverage
    assert "over the 1 parts of the dataset most similar" in found.note
