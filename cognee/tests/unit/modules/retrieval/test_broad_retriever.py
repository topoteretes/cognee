"""BROAD search type: count by listing in parallel, tally in code (SDK-324).

The LLM is stubbed: these tests pin the parts that must be exact regardless of
the model — planning validation, sharding, de-duplication, name merging, entity
counting from the graph, and the rendered context.
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

    entity_nodes = [
        ("t-person", {"type": "EntityType", "name": "person"}),
        ("t-place", {"type": "EntityType", "name": "place"}),
        ("e1", {"type": "Entity", "name": "natasha rostov", "description": "a Rostov"}),
        ("e2", {"type": "Entity", "name": "pierre", "description": "a count"}),
        ("e3", {"type": "Entity", "name": "count ilya rostov", "description": "her father"}),
        ("e4", {"type": "Entity", "name": "moscow", "description": "a city"}),
    ]
    entity_edges = [
        ("e1", "t-person", "is_a", {}),
        ("e2", "t-person", "is_a", {}),
        ("e3", "t-person", "is_a", {}),
        ("e4", "t-place", "is_a", {}),
        ("e1", "e2", "knows", {}),
    ]
    text_nodes = [
        ("c1", {"type": "DocumentChunk", "text": "Chapter one."}),
        ("r1", {"type": "DltRow", "text": "Row Data: assignee: ann"}),
        ("c2", {"type": "DocumentChunk", "text": ""}),
    ]

    async def get_filtered_graph_data(self, attribute_filters):
        types = attribute_filters[0]["type"]
        if "Entity" in types:
            return self.entity_nodes, self.entity_edges
        return [n for n in self.text_nodes if n[1]["type"] in types], []


def _use_fake_graph(monkeypatch):
    fake_engine = SimpleNamespace(graph=_FakeGraph(), vector=None)

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

    assert [unit.id for unit in units] == ["c1", "r1"]


# --- sharding ------------------------------------------------------------------


def test_shards_respect_the_token_budget_and_keep_every_unit():
    retriever = BroadRetriever()
    units = _units(20)

    shards, tokens = retriever.pack_shards(units, 200)

    assert [unit for shard in shards for unit in shard] == units
    assert len(shards) > 1
    for shard in shards:
        size = sum(len(retriever.tokenizer.extract_tokens(u.text)) for u in shard)
        assert size <= 200 or len(shard) == 1
    assert tokens == sum(len(retriever.tokenizer.extract_tokens(u.text)) for u in units)


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


# --- graph entities ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distinct_typed_things_are_counted_from_the_graph_without_a_scan(monkeypatch):
    calls = []

    def respond(model, _):
        calls.append(model)
        return CountPlan(source="entities", entity_types=["person"], item="a person")

    _stub_llm(monkeypatch, respond)
    _use_fake_graph(monkeypatch)

    result = await BroadRetriever().get_retrieved_objects("How many people?")

    assert result.total == 3
    assert calls == [CountPlan]  # only the planner ran


@pytest.mark.asyncio
async def test_a_name_filter_is_applied_exactly_by_code(monkeypatch):
    """ "People with Rostov in their name" is a substring match, not an LLM read."""
    calls = []

    def respond(model, _):
        calls.append(model)
        return CountPlan(
            source="entities", entity_types=["person"], item="a person", name_contains="Rostov"
        )

    _stub_llm(monkeypatch, respond)
    _use_fake_graph(monkeypatch)

    result = await BroadRetriever().get_retrieved_objects("How many Rostovs?")

    assert result.total == 2
    assert result.units_total == 3
    assert calls == [CountPlan]


# --- map / reduce ------------------------------------------------------------------


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

    result = await BroadRetriever().count_items(plan, _units(1), 1000)

    assert result.total == 2
    assert dict(result.groups) == {"ann": 1, "bob": 1}


@pytest.mark.asyncio
async def test_every_occurrence_counts_without_a_dedup_key(monkeypatch):
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, evidence="Moscow burned"),
            ExtractedItem(unit=0, evidence="left Moscow"),
            ExtractedItem(unit=0, evidence="Moscow burned"),  # same quote listed twice: one mention
        ]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)

    result = await BroadRetriever().count_items(
        CountPlan(source="text", item="a mention of Moscow"), _units(1), 1000
    )

    assert result.total == 2


@pytest.mark.asyncio
async def test_identical_quotes_from_different_rows_are_different_items(monkeypatch):
    """Table rows repeat values: "worth_reviewing: yes" on 40 rows is 40 items."""
    shard = ShardItems(
        items=[ExtractedItem(unit=index, evidence="worth_reviewing: yes") for index in range(40)]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)

    result = await BroadRetriever().count_items(
        CountPlan(source="text", item="a row with worth_reviewing yes"), _units(40, words=2), 10_000
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

    result = await BroadRetriever().count_items(plan, _units(1), 1000)

    assert result.groups == [("Akshats-git", 3), ("Megha-gbs", 1)]


@pytest.mark.asyncio
async def test_every_shard_is_read(monkeypatch):
    seen = []

    def respond(model, text_input):
        seen.append(text_input)
        return ShardItems(items=[ExtractedItem(unit=0, evidence=f"item {len(seen)}")])

    _stub_llm(monkeypatch, respond)

    result = await BroadRetriever().count_items(CountPlan(source="text", item="x"), _units(20), 200)

    assert result.llm_calls == len(seen) > 1
    assert result.total == len(seen)
    assert all(f"unit {i} " in "".join(seen) for i in range(20))


# --- context -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_states_the_exact_total_and_coverage():
    plan = CountPlan(source="text", item="a PR", group_by="author", dedup_key="PR number")
    result = CountResult(
        plan=plan,
        total=981,
        groups=[("Akshats-git", 43), ("Megha-gbs", 20)],
        units_scanned=31,
        units_total=31,
        evidence=["Akshats-git opened pull request #1"],
        llm_calls=12,
        tokens_read=140000,
    )

    context = await BroadRetriever().get_context_from_objects("q", result)

    assert "TOTAL: 981" in context
    assert "all 31 of 31" in context
    assert "Akshats-git: 43" in context


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
    fake_engine = SimpleNamespace(graph=graph, vector=None)

    async def unified():
        return fake_engine

    monkeypatch.setattr(broad_retriever, "get_unified_engine", unified)

    result = await BroadRetriever().get_retrieved_objects("How many times is Moscow mentioned?")

    assert result.total == 3
    assert calls == [CountPlan]


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
    _use_fake_graph(monkeypatch)

    result = await BroadRetriever().get_retrieved_objects("How many issues were assigned?")

    assert result.plan.source == "text"
    assert result.total == 1
    assert result.llm_calls == 1
