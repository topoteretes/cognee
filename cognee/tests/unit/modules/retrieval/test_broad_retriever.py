"""BROAD search type: plan once, count in code (SDK-324).

The LLM is stubbed: these tests pin the parts that must hold regardless of the
model — which counter runs, sharding, de-duplication, name merging, the document
context line, and how honestly the result is worded.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.retrieval import broad_retriever
from cognee.modules.retrieval.broad_retriever import (
    BroadRetriever,
    CountPlan,
    CountResult,
    ExtractedItem,
    NameGroups,
    ShardItems,
    Unit,
)
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType


def _stub_llm(monkeypatch, respond):
    """Route every structured-output call to ``respond(response_model, text_input)``."""

    async def fake(text_input, system_prompt, response_model, **kwargs):
        return respond(response_model, text_input)

    monkeypatch.setattr(broad_retriever.LLMGateway, "acreate_structured_output", fake)


def _units(count: int, words: int = 50) -> list[Unit]:
    return [Unit(id=f"u{i}", text=f"unit {i} " + "word " * words) for i in range(count)]


class _FakeGraph:
    """Answers get_filtered_graph_data for entity and text-unit node types."""

    def __init__(self):
        self.entity_nodes = [
            ("t-person", {"type": "EntityType", "name": "person"}),
            ("t-place", {"type": "EntityType", "name": "place"}),
            ("e1", {"type": "Entity", "name": "natasha rostov", "description": "a Rostov"}),
            ("e2", {"type": "Entity", "name": "pierre", "description": "a count"}),
            ("e3", {"type": "Entity", "name": "count ilya rostov", "description": "her father"}),
            ("e4", {"type": "Entity", "name": "moscow", "description": "a city"}),
        ]
        self.entity_edges = [
            ("e1", "t-person", "is_a", {}),
            ("e2", "t-person", "is_a", {}),
            ("e3", "t-person", "is_a", {}),
            ("e4", "t-place", "is_a", {}),
            ("e1", "e2", "knows", {}),
        ]
        self.text_nodes = [
            ("c1", {"type": "DocumentChunk", "text": "Chapter one."}),
            ("r1", {"type": "DltRow", "text": "Row Data: assignee: ann"}),
            ("c2", {"type": "DocumentChunk", "text": ""}),
        ]
        self.text_edges = []

    async def get_filtered_graph_data(self, attribute_filters):
        types = attribute_filters[0]["type"]
        if "Entity" in types:
            return self.entity_nodes, self.entity_edges
        return [n for n in self.text_nodes if n[1]["type"] in types], self.text_edges


def _use_graph(monkeypatch, graph):
    fake_engine = SimpleNamespace(graph=graph, vector=None)

    async def unified():
        return fake_engine

    monkeypatch.setattr(broad_retriever, "get_unified_engine", unified)


# --- sources -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entities_are_grouped_by_their_type():
    entities = await BroadRetriever().load_entities(_FakeGraph())

    assert sorted(entities) == ["person", "place"]
    assert [unit.id for unit in entities["person"]] == ["e1", "e2", "e3"]


@pytest.mark.asyncio
async def test_text_units_are_every_chunk_and_table_row_with_text():
    units = await BroadRetriever().load_text_units(_FakeGraph())

    assert sorted(unit.id for unit in units) == ["c1", "r1"]


@pytest.mark.asyncio
async def test_mid_document_chunks_carry_the_documents_first_line(monkeypatch):
    """A CSV ingested as text has its header only in chunk 0; later chunks get it as context."""
    graph = _FakeGraph()
    graph.text_nodes = [
        ("c1", {"type": "DocumentChunk", "chunk_index": 1, "text": "2,bob,no,yes"}),
        (
            "c0",
            {"type": "DocumentChunk", "chunk_index": 0, "text": "id,who,review,main\n1,ann,yes,no"},
        ),
        ("d", {"type": "TextDocument", "name": "triage.txt"}),
    ]
    graph.text_edges = [("c0", "d", "is_part_of", {}), ("c1", "d", "is_part_of", {})]
    seen = []

    def respond(model, text_input):
        seen.append(text_input)
        return ShardItems(items=[])

    _stub_llm(monkeypatch, respond)
    retriever = BroadRetriever(shard_tokens=10_000)

    units = await retriever.load_text_units(graph)
    await retriever.count_by_reading(CountPlan(source="text", item="a row"), units)

    assert [unit.id for unit in units] == ["c0", "c1"]
    assert units[0].preamble == "" and units[1].preamble == "id,who,review,main"
    assert seen[0].count("id,who,review,main") == 2  # chunk 0 itself + one context line
    assert seen[0].count("[document start") == 1


# --- planning --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_entity_types_are_normalized_and_validated(monkeypatch):
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(source="entities", entity_types=["Person"], item="a person"),
    )

    plan = await BroadRetriever().plan("How many people?", {"person": _units(3)})

    assert plan.entity_types == ["person"]


@pytest.mark.asyncio
async def test_planner_choosing_a_missing_type_fails_fast(monkeypatch):
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(source="entities", entity_types=["Ghost"], item="a ghost"),
    )

    with pytest.raises(ValueError, match="not in the graph"):
        await BroadRetriever().plan("How many ghosts?", {"person": _units(1)})


# --- which counter runs -------------------------------------------------------------


@pytest.mark.asyncio
async def test_distinct_typed_things_are_counted_from_the_graph(monkeypatch):
    calls = []

    def respond(model, _):
        calls.append(model)
        return CountPlan(source="entities", entity_types=["person"], item="a person")

    _stub_llm(monkeypatch, respond)
    _use_graph(monkeypatch, _FakeGraph())

    result = await BroadRetriever().get_retrieved_objects("How many people?")

    assert (result.method, result.total) == ("graph", 3)
    assert calls == [CountPlan]  # only the planner ran


@pytest.mark.asyncio
async def test_a_name_filter_is_applied_exactly_by_code(monkeypatch):
    """ "People with Rostov in their name" is a substring match, not an LLM read."""
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(
            source="entities", entity_types=["person"], item="a person", name_contains="Rostov"
        ),
    )
    _use_graph(monkeypatch, _FakeGraph())

    result = await BroadRetriever().get_retrieved_objects("How many Rostovs?")

    assert (result.total, result.units) == (2, 3)


@pytest.mark.asyncio
async def test_an_entity_plan_with_a_condition_reads_the_text(monkeypatch):
    """Entities hold a name, not assignments or verdicts: such conditions scan the text."""

    def respond(model, _):
        if model is CountPlan:
            return CountPlan(
                source="entities", entity_types=["person"], item="an issue", condition="assigned"
            )
        return ShardItems(items=[ExtractedItem(unit=0, evidence="Row Data: assignee: ann")])

    _stub_llm(monkeypatch, respond)
    _use_graph(monkeypatch, _FakeGraph())

    result = await BroadRetriever().get_retrieved_objects("How many issues were assigned?")

    assert (result.method, result.plan.source, result.total) == ("reading", "text", 1)


@pytest.mark.asyncio
async def test_word_mentions_are_counted_by_code_across_every_unit(monkeypatch):
    """ "How many times is X mentioned" needs no reading: whole-word matches, no listing."""
    calls = []

    def respond(model, _):
        calls.append(model)
        return CountPlan(source="text", item="a mention of Moscow", literal_terms=["Moscow"])

    _stub_llm(monkeypatch, respond)
    graph = _FakeGraph()
    graph.text_nodes = [
        ("c1", {"type": "DocumentChunk", "text": "Moscow burned. They left Moscow's gates."}),
        ("c2", {"type": "DocumentChunk", "text": "Muscovites and MoscowRiver are not Moscow."}),
    ]
    _use_graph(monkeypatch, graph)

    result = await BroadRetriever().get_retrieved_objects("How many times is Moscow mentioned?")

    assert (result.method, result.total) == ("words", 3)
    assert calls == [CountPlan]


# --- counting by reading ---------------------------------------------------------------


def test_shards_respect_the_token_budget_and_keep_every_unit():
    retriever = BroadRetriever(shard_tokens=200)
    units = _units(20)

    shards, tokens = retriever.pack_shards(units)

    assert [unit for shard in shards for unit in shard] == units
    assert len(shards) > 1
    for shard in shards:
        size = sum(len(retriever.tokenizer.extract_tokens(u.text)) for u in shard)
        assert size <= 200 or len(shard) == 1
    assert tokens == sum(len(retriever.tokenizer.extract_tokens(u.text)) for u in units)


@pytest.mark.asyncio
async def test_every_shard_is_read(monkeypatch):
    seen = []

    def respond(model, text_input):
        seen.append(text_input)
        return ShardItems(items=[ExtractedItem(unit=0, evidence=f"item {len(seen)}")])

    _stub_llm(monkeypatch, respond)

    result = await BroadRetriever(shard_tokens=200).count_by_reading(
        CountPlan(source="text", item="x"), _units(20)
    )

    assert result.llm_calls == len(seen) > 1
    assert result.total == len(seen)
    assert all(f"unit {i} " in "".join(seen) for i in range(20))


@pytest.mark.asyncio
async def test_repeated_mentions_of_one_item_are_counted_once(monkeypatch):
    """A recap that repeats PR #7 must not count it twice (dedup by key)."""
    items = [
        ExtractedItem(unit=0, key="#7", group="ann", evidence="Ann opened PR #7"),
        ExtractedItem(unit=0, key="PR 8", group="bob", evidence="Bob opened PR 8"),
        ExtractedItem(unit=0, key="7", group="ann", evidence="recap: PR #7 by Ann"),
    ]

    def respond(model, _):
        if model is ShardItems:
            return ShardItems(items=items)
        return NameGroups(groups=[["ann"], ["bob"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="a PR", group_by="author", dedup_key="PR number")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.total == 2
    assert dict(result.groups) == {"ann": 1, "bob": 1}


@pytest.mark.asyncio
async def test_every_occurrence_counts_without_a_dedup_key(monkeypatch):
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, evidence="Moscow burned"),
            ExtractedItem(unit=0, evidence="left Moscow"),
            ExtractedItem(unit=0, evidence="Moscow burned"),  # same quote listed twice: one item
        ]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)

    result = await BroadRetriever().count_by_reading(
        CountPlan(source="text", item="a burning"), _units(1)
    )

    assert result.total == 2


@pytest.mark.asyncio
async def test_identical_quotes_from_different_rows_are_different_items(monkeypatch):
    """Table rows repeat values: "worth_reviewing: yes" on 40 rows is 40 items."""
    shard = ShardItems(
        items=[ExtractedItem(unit=index, evidence="worth_reviewing: yes") for index in range(40)]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(
        CountPlan(source="text", item="a row with worth_reviewing yes"), _units(40, words=2)
    )

    assert result.total == 40


@pytest.mark.asyncio
async def test_name_variants_are_merged_before_tallying(monkeypatch):
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, group="Akshats-git", evidence="a"),
            ExtractedItem(unit=0, group="Akshats", evidence="b"),
            ExtractedItem(unit=0, group="@Akshats-git", evidence="c"),
            ExtractedItem(unit=0, group="Megha-gbs", evidence="d"),
        ]
    )

    def respond(model, _):
        if model is ShardItems:
            return shard
        return NameGroups(groups=[["Akshats-git", "Akshats", "@Akshats-git"], ["Megha-gbs"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="an assignment", group_by="assignee")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.groups == [("Akshats-git", 3), ("Megha-gbs", 1)]


# --- how the result is worded -------------------------------------------------------


def _plan(**fields) -> CountPlan:
    return CountPlan(source="text", item="an item", **fields)


@pytest.mark.asyncio
async def test_a_count_by_reading_is_never_presented_as_exact():
    result = CountResult(
        plan=_plan(group_by="author"),
        method="reading",
        total=946,
        units=28,
        groups=[("Akshats-git", 43)],
        llm_calls=28,
        tokens_read=137095,
    )

    context = await BroadRetriever().get_context_from_objects("q", result)

    assert "TOTAL: 946" in context
    assert "reading can miss an item" in context
    assert "Exact." not in context
    assert "Akshats-git: 43" in context


@pytest.mark.asyncio
async def test_graph_and_word_counts_are_stated_as_exact():
    graph_result = CountResult(
        plan=CountPlan(source="entities", entity_types=["person"], item="a person"),
        method="graph",
        total=6756,
        units=6756,
    )
    words_result = CountResult(
        plan=_plan(literal_terms=["Moscow"]), method="words", total=720, units=172
    )

    graph_context = await BroadRetriever().get_context_from_objects("q", graph_result)
    words_context = await BroadRetriever().get_context_from_objects("q", words_result)

    assert "Exact." in graph_context and "TOTAL: 6756" in graph_context
    assert 'whole-word matches of "Moscow"' in words_context and "Exact for these" in words_context


# --- search wiring -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_builds_broad_with_its_scan_settings():
    retriever = await get_search_type_retriever_instance(
        SearchType.BROAD,
        "How many?",
        retriever_specific_config={"shard_tokens": 4000, "max_parallel_calls": 4},
    )

    assert isinstance(retriever, BroadRetriever)
    assert retriever.shard_tokens == 4000
    assert retriever.max_parallel_calls == 4


@pytest.mark.asyncio
async def test_distinct_names_are_counted_as_groups_and_keys_are_never_merged(monkeypatch):
    """ "How many different people": groups after merging spellings. Identifiers are keys and
    are never merged, even when their titles look alike."""
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="INC-1", group="Megha-gbs", evidence="a"),
            ExtractedItem(unit=1, key="INC-2", group="Megha", evidence="b"),
            ExtractedItem(unit=2, key="INC-3", group="Akshats-git", evidence="c"),
        ]
    )
    merge_inputs = []

    def respond(model, text_input):
        if model is ShardItems:
            return shard
        merge_inputs.append(text_input)
        return NameGroups(groups=[["Megha-gbs", "Megha"], ["Akshats-git"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text", item="an incident", group_by="responder", dedup_key="incident id"
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(3, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.total == 3  # three different incident ids survive
    assert "DISTINCT responder: 2" in context
    assert all("INC-" not in text for text in merge_inputs)  # keys never reach the merge step
