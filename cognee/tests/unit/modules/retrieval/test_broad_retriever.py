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
    TargetMatch,
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


def test_a_unit_longer_than_a_shard_is_split_at_paragraphs():
    """A 5k-token PDF chunk read whole is denser than the shard budget: cut it up."""
    retriever = BroadRetriever(shard_tokens=40)
    paragraphs = [f"paragraph {i} " + "word " * 20 for i in range(6)]
    unit = Unit(id="c7", text="\n\n".join(paragraphs), preamble="Header line")

    shards, _ = retriever.pack_shards([unit, Unit(id="c8", text="short")])
    pieces = [piece for shard in shards for piece in shard]

    assert len(pieces) > 2 and pieces[-1].id == "c8"
    assert all(
        piece.id.startswith("c7#") and piece.preamble == "Header line" for piece in pieces[:-1]
    )
    assert "\n".join(p.text for p in pieces[:-1]).split() == unit.text.split()  # nothing lost
    for piece in pieces[:-1]:
        assert len(retriever.tokenizer.extract_tokens(piece.text)) <= 40 or "\n" not in piece.text


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
        ExtractedItem(unit=0, key="8", group="bob", evidence="Bob opened PR 8"),
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


def _item(group, key, undone=False):
    return ExtractedItem(unit=0, group=group, key=key, undone=undone, evidence=f"{group}-{key}")


def test_a_relation_is_one_item_per_participant_and_a_removal_subtracts_it():
    """Arthur–Priya is Arthur's connection and Priya's; "Arthur removed Priya" ends both
    entries the removal names, and a request that was only declined was never listed."""
    items = [
        _item("Arthur", "Priya"),
        _item("Priya", "Arthur"),
        _item("Arthur", "Mei"),
        _item("Mei", "Arthur"),
        _item("Arthur", "Priya"),  # the same connection told again from Arthur's side
        _item("Arthur", "Mei", undone=True),
        _item("Mei", "Arthur", undone=True),
    ]

    kept = BroadRetriever.dedup(items, by_group=True)

    assert sorted((i.group, i.key) for i in kept) == [("Arthur", "Priya"), ("Priya", "Arthur")]


def test_an_item_with_several_group_values_is_one_entry_per_value():
    """A paper by three authors is one paper of each; the key stays the paper."""
    paper = ExtractedItem(unit=0, key="Paper 3", groups=["Ana", "Ben", "Chen"], evidence="p3")
    single = ExtractedItem(unit=0, key="Paper 4", group="Dana", evidence="p4")

    entries = broad_retriever._one_entry_per_group([paper, single], relation=False)

    assert [(e.group, e.key) for e in entries] == [
        ("Ana", "Paper 3"),
        ("Ben", "Paper 3"),
        ("Chen", "Paper 3"),
        ("Dana", "Paper 4"),
    ]


def test_a_relation_listed_once_becomes_every_ordered_pair():
    """A three-author paper is three co-authorships; each author gets the two others.
    A relation listed with one side in group and the other in key is completed too."""
    paper = ExtractedItem(unit=0, groups=["Ana", "Ben", "Chen"], evidence="p3")
    pair = ExtractedItem(unit=1, group="Helga", key="Arthur", evidence="accepted")

    entries = broad_retriever._one_entry_per_group([paper, pair], relation=True)

    assert sorted((e.group, e.key) for e in entries) == [
        ("Ana", "Ben"),
        ("Ana", "Chen"),
        ("Arthur", "Helga"),
        ("Ben", "Ana"),
        ("Ben", "Chen"),
        ("Chen", "Ana"),
        ("Chen", "Ben"),
        ("Helga", "Arthur"),
    ]


@pytest.mark.asyncio
async def test_a_relation_listed_under_one_side_counts_for_both(monkeypatch):
    """The model lists "Helga accepted a request from Art" under Helga only; code adds
    Arthur's side, spells "Art" as Arthur Bennett, and a removal ends both sides."""
    shard = ShardItems(
        items=[
            _item("Helga Menon", "Art"),
            _item("Arthur Bennett", "Priya Nair"),
            _item("Priya Nair", "Arthur Bennett"),  # the same connection from Priya's side
            _item("Mei Lin", "Arthur Bennett"),
            _item("Arthur Bennett", "Mei Lin", undone=True),
            _item("Helga Menon", "Priya Nair"),
        ],
        aliases=[["Arthur Bennett", "Art"]],
    )

    def respond(model, _):
        if model is ShardItems:
            return shard
        if model is TargetMatch:
            return TargetMatch(names=[])
        return NameGroups(groups=[["Arthur Bennett", "Art"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text",
        item="a connection",
        group_by="member",
        target="Art",  # the roster says "Arthur Bennett (usually called Art)"
        dedup_key="the other member",
        relation=True,
        reversible=True,
        list_items=True,
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(6, words=2))

    assert result.total == 2  # Helga and Priya; Mei was removed
    assert dict(result.groups) == {
        "Arthur Bennett": 2,
        "Helga Menon": 2,
        "Priya Nair": 2,
    }
    assert sorted(entry.split(":")[0] for entry in result.evidence) == ["Helga Menon", "Priya Nair"]


def test_a_key_keeps_its_kind_so_pr_42_and_issue_42_are_two_items():
    """Mixed contributions: "PR 42" and "issue 42" differ; "PR #42", "PR-42" and "pr 42"
    are one; a bare "#42" is the number alone."""
    normalize = broad_retriever._normalize_key

    assert normalize("PR #42") == normalize("PR-42") == normalize("pr 42") == "pr-42"
    assert normalize("issue 42") == "issue-42" and normalize("issue 42") != normalize("PR 42")
    assert normalize("#42") == normalize("42") == normalize("no. 42") == "42"
    assert normalize("EXP-1004") == "exp-1004" and normalize("2026-03-04") == "2026-03-04"


def test_a_key_made_of_several_words_keeps_them_all():
    """Match reports have no match number: the record is "Matchday 5: Harbour City v
    Glenmarsh City". Eight matches share a matchday, so the words must survive — 54
    draws once collapsed to the 25 matchdays they fell on."""
    normalize = broad_retriever._normalize_key

    assert normalize("Matchday 5: Harbour City v Glenmarsh City") != normalize(
        "Matchday 5: Ashvale v Fenwick"
    )
    assert normalize("Matchday 5: Harbour City v Glenmarsh City") == normalize(
        "matchday 5, harbour city v glenmarsh city"
    )
    assert normalize("Harbour City v Fenwick, 4th minute") != normalize(
        "Harbour City v Larkhill, 4th minute"
    )


def test_a_bare_number_adopts_the_kind_written_elsewhere_for_it():
    """Pavel's PR 1032 was keyed "PR 1032" once and "1032" once: one item. PR 42 and
    issue 42 stay two. A bare 7 with no labelled twin stays bare."""
    items = [
        _item("Pavel", "PR 1032"),
        _item("Pavel", "1032"),
        _item("Pavel", "PR 42"),
        _item("Pavel", "issue 42"),
        _item("Pavel", "7"),
    ]

    kept = BroadRetriever.dedup(items)

    assert [i.key for i in kept] == ["PR 1032", "PR 42", "issue 42", "7"]


def test_names_with_digits_are_not_reduced_to_their_digits():
    """ "raj921" and "RajdeepKushwaha5" are two members, and "@raj921" is raj921; only an
    item identifier ("PR #921") reduces to its digits."""
    items = [_item("Ann", "raj921"), _item("Ann", "RajdeepKushwaha5"), _item("Ann", "@raj921")]

    assert len(BroadRetriever.dedup(items, by_group=True)) == 2
    assert broad_retriever._loose_name("@raj921") == "raj921"
    assert broad_retriever._normalize_key("#921") == "921"


def test_the_latest_dated_entry_decides_an_items_state():
    """Ticket 1: opened, resolved, reopened -> open. Ticket 2: opened, resolved -> gone.
    Ticket 3: the resolve was read before the open (separate files, no document
    order), but its date is later -> gone. The open ticket is cited by its opening."""
    items = [
        ExtractedItem(unit=0, key="T-1", when="2026-02-02", evidence="opened"),
        ExtractedItem(unit=1, key="T-1", when="2026-02-05", undone=True, evidence="resolved"),
        ExtractedItem(unit=2, key="T-1", when="2026-02-09", evidence="reopened"),
        ExtractedItem(unit=3, key="T-2", when="2026-02-03", evidence="opened"),
        ExtractedItem(unit=4, key="T-2", when="2026-02-04", undone=True, evidence="resolved"),
        ExtractedItem(unit=5, key="T-3", when="2026-03-09", undone=True, evidence="resolved"),
        ExtractedItem(unit=6, key="T-3", when="2026-03-01", evidence="opened"),
    ]

    kept = BroadRetriever.dedup(items)

    assert [(i.key, i.evidence) for i in kept] == [("T-1", "opened")]


@pytest.mark.asyncio
async def test_a_reversible_plans_condition_never_filters_state_entries(monkeypatch):
    """The planner wrote "currently open, not resolved"; applied per entry that would
    skip the resolve entries the state machine needs."""
    seen = []

    def respond(model, text_input):
        seen.append(text_input)
        return ShardItems(items=[])

    _stub_llm(monkeypatch, respond)
    plan = _plan(condition="currently open, not resolved", reversible=True, dedup_key="ticket")

    await BroadRetriever().count_by_reading(plan, _units(1))

    assert "list EVERY state entry (opened, resolved, reopened, removed)" in seen[0]
    assert "code decides the state from the latest entry" in seen[0]


@pytest.mark.asyncio
async def test_a_percentage_is_two_counts_from_one_pass(monkeypatch):
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="T-1", matches=True, evidence="escalated"),
            ExtractedItem(unit=1, key="T-2", matches=False, evidence="not"),
            ExtractedItem(unit=2, key="T-3", matches=True, evidence="escalated"),
            ExtractedItem(unit=3, key="T-4", matches=False, evidence="not"),
        ]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)
    plan = _plan(ratio_condition="the ticket was escalated", dedup_key="the ticket number")

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(4, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert (result.total, result.denominator) == (2, 4)
    assert "out of 4 items in all: 50.0%" in context


def test_a_ratio_flag_holds_if_any_entry_of_the_item_says_so():
    """The escalation is a separate email from the ticket's opening."""
    items = [
        ExtractedItem(unit=0, key="T-1", matches=False, evidence="opened"),
        ExtractedItem(unit=1, key="T-1", matches=True, evidence="escalated"),
        ExtractedItem(unit=2, key="T-2", matches=False, evidence="opened"),
    ]

    kept = BroadRetriever.dedup(items)

    assert [(i.key, i.matches) for i in kept] == [("T-1", True), ("T-2", False)]


@pytest.mark.asyncio
async def test_a_wording_key_gets_the_records_date(monkeypatch):
    """ "The budget's title" repeats across meetings; the planner's key is widened by code."""
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(
            source="text", item="a budget line", measure="euros", dedup_key="the budget's title"
        ),
    )

    plan = await BroadRetriever().plan("What is the total budget approved?", {})

    assert plan.dedup_key == "the budget's title, together with the date it appears under"
    assert not broad_retriever._is_wording_key("the ticket number")
    assert not broad_retriever._is_wording_key("the meeting date and the decision's wording")


@pytest.mark.asyncio
async def test_an_average_divides_the_sum_by_the_items_with_an_amount(monkeypatch):
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="O-1", amount=10, evidence="10"),
            ExtractedItem(unit=1, key="O-2", amount=30, evidence="30"),
            ExtractedItem(unit=2, key="O-3", evidence="no amount"),
        ]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)
    plan = _plan(measure="order value", average=True, dedup_key="the order number")

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(3, words=2))

    assert result.total == 20


@pytest.mark.asyncio
async def test_an_unsupported_question_is_refused_with_the_reason(monkeypatch):
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(
            source="text", item="a customer", unsupported="absence across the corpus"
        ),
    )
    _use_graph(monkeypatch, _FakeGraph())

    result = await BroadRetriever().get_retrieved_objects("Who never wrote to support?")
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.method == "unsupported"
    assert "cannot be answered by counting: absence across the corpus" in context


def test_one_item_per_group_and_key_and_a_recap_finds_its_item():
    """A key written "#7" or "7" is one key. Paper 3 by Ann and by Bob is one item of
    each; a recap that names #7 without its group is the item already counted."""
    items = [
        _item("Ann", "#7"),
        _item(None, "7"),
        _item("Ann", "Paper 3"),
        _item("Bob", "Paper 3"),
        _item("Ann", "7"),
    ]

    kept = BroadRetriever.dedup(items)

    assert [(i.group, i.key) for i in kept] == [
        ("Ann", "#7"),
        ("Ann", "Paper 3"),
        ("Bob", "Paper 3"),
    ]


@pytest.mark.asyncio
async def test_a_paper_counts_for_every_author_it_lists(monkeypatch):
    """The extractor returns paper 34 once with three authors; Ana's tally includes it
    even though she is not the first author."""
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="34", groups=["Omar Reyes", "Ana Kovač"], evidence="p34"),
            ExtractedItem(unit=0, key="37", groups=["Ana Kovač", "Emeka Obi"], evidence="p37"),
            ExtractedItem(unit=1, key="35", groups=["Gil Park", "Iris Ng"], evidence="p35"),
        ]
    )

    def respond(model, _):
        if model is ShardItems:
            return shard
        return NameGroups(groups=[]) if model is NameGroups else TargetMatch(names=[])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text", item="a paper", group_by="author", target="Ana Kovač", dedup_key="paper"
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(2, words=2))

    assert result.total == 2 and dict(result.groups)["Ana Kovač"] == 2


@pytest.mark.asyncio
async def test_unkeyed_entries_count_and_bound_the_possible_overcount(monkeypatch):
    """An entry without its key cannot be checked for repeats. It counts, and the answer
    context states how many such entries there were: the most the total is over by."""
    shard = ShardItems(
        items=[
            _item(None, "1"),
            ExtractedItem(unit=0, evidence="an order whose number the model left out"),
            ExtractedItem(unit=1, evidence="another with no number given"),
        ]
    )
    _stub_llm(monkeypatch, lambda model, _: shard)
    plan = _plan(dedup_key="the order number")

    result = await BroadRetriever().count_by_reading(plan, _units(1))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.total == 3 and result.unkeyed == 2
    assert "2 counted entries carried no the order number" in context
    assert "over by up to that many" in context


def test_a_relation_without_a_target_counts_each_pair_once():
    """ "How many connections were removed": Arthur–Priya is one connection, listed
    for both of them."""
    items = [_item("Arthur", "Priya"), _item("Priya", "Arthur"), _item("Mei", "Priya")]
    plan = _plan(group_by="member", relation=True, dedup_key="the other member")

    kept = BroadRetriever.dedup(items, by_group=True)
    pairs = {frozenset((i.group, i.key)) for i in kept}

    assert len(kept) == 3 and len(pairs) == 2
    assert plan.relation  # the total uses the pair count; the groups keep the per-member tally


@pytest.mark.asyncio
async def test_stated_aliases_merge_by_code_even_when_the_model_does_not(monkeypatch):
    """The roster says "Arthur Bennett (usually called Art)". With no target to anchor
    on, the model merge may miss it; the stated alias joins the groups regardless."""
    shard = ShardItems(
        items=[
            _item("Arthur Bennett", "1"),
            _item("Art", "2"),
            _item("Yuki Sato", "3"),
        ],
        aliases=[["Arthur Bennett", "Art"]],
    )

    def respond(model, _):
        return shard if model is ShardItems else NameGroups(groups=[])

    _stub_llm(monkeypatch, respond)
    plan = _plan(group_by="member", dedup_key="the connection")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.groups == [("Arthur Bennett", 2), ("Yuki Sato", 1)]


@pytest.mark.asyncio
async def test_merged_names_are_tallied_under_the_spelling_the_corpus_uses_most(monkeypatch):
    """Match reports call Ciaran Rojas "the Lighthouse" 40% of the time; the roster
    states the alias. The tally must read "Ciaran Rojas", not the longer nickname."""
    shard = ShardItems(
        items=[
            _item("Ciaran Rojas", "1"),
            _item("Ciaran Rojas", "2"),
            _item("Ciaran Rojas", "3"),
            _item("The Lighthouse", "4"),
            _item("The Lighthouse", "5"),
            _item("Yusuf Demir", "6"),
        ],
        aliases=[["Ciaran Rojas", "the Lighthouse"]],
    )

    def respond(model, _):
        return shard if model is ShardItems else NameGroups(groups=[])

    _stub_llm(monkeypatch, respond)
    plan = _plan(group_by="scorer", dedup_key="the match and minute")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.groups == [("Ciaran Rojas", 5), ("Yusuf Demir", 1)]


def test_an_alias_returned_as_a_sentence_is_split_into_its_names():
    """The extractor sometimes returns the stating sentence as one string."""
    groups = broad_retriever._alias_groups(
        [
            ["Pavel Horák is Pav"],
            ["Akshats-git (usually called Akshats)"],
            ["Robert, known as Bob"],
            ["Maria Duarte", "Maria"],
            ["Lone Name"],
        ]
    )

    assert groups == [
        ["Pavel Horák", "Pav"],
        ["Akshats-git", "Akshats"],
        ["Robert", "Bob"],
        ["Maria Duarte", "Maria"],
    ]


@pytest.mark.asyncio
async def test_only_read_names_reach_the_model_as_stated_aliases(monkeypatch):
    """A roster's four alias sets came back as four sentence-strings in one group. None
    is a read name, so they must not be presented to the model as "the same": it
    would obey and fold four people into one (Esme Nkemelu, 155 contributions)."""
    seen: list[str] = []
    sentences = ["Pavel Horák is Pav", "Anna Lund is Annie", "Esme Nkemelu is Es"]
    shard = ShardItems(
        items=[_item("Pavel Horák", "1"), _item("Anna Lund", "2"), _item("Esme Nkemelu", "3")],
        aliases=[sentences],  # the sentences reach merge only if parsing failed on them
    )

    def respond(model, text):
        if model is NameGroups:
            seen.append(text)
            return NameGroups(groups=[])
        return shard

    _stub_llm(monkeypatch, respond)
    monkeypatch.setattr(broad_retriever, "_alias_groups", lambda raw: raw)  # force the raw form
    plan = _plan(group_by="contributor", dedup_key="the contribution's number")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert "Stated" not in seen[0]
    assert set(result.groups) == {("Pavel Horák", 1), ("Anna Lund", 1), ("Esme Nkemelu", 1)}


def test_a_long_paragraph_is_cut_at_sentence_ends_never_inside_one():
    retriever = BroadRetriever(shard_tokens=30)
    sentences = [f"Record {i} was opened by someone for issue {i}." for i in range(12)]
    unit = Unit(id="c1", text=" ".join(sentences))  # pypdf text: no blank lines at all

    pieces = retriever.split_oversized([unit])

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.text.endswith(".")
        assert all(s in " ".join(p.text for p in pieces) for s in sentences)


@pytest.mark.asyncio
async def test_undone_only_subtracts_when_the_plan_counts_items_in_effect(monkeypatch):
    """ "How many orders were cancelled" counts the cancellations: an extractor that
    marks them undone must not empty the count. Only a reversible plan subtracts."""
    shard = ShardItems(items=[_item(None, "WO-1", undone=True), _item(None, "WO-2", undone=True)])
    _stub_llm(monkeypatch, lambda model, _: shard)

    counted = await BroadRetriever().count_by_reading(
        _plan(condition="was cancelled", dedup_key="the order number"), _units(1)
    )
    in_effect = await BroadRetriever().count_by_reading(
        _plan(dedup_key="the order number", reversible=True), _units(1)
    )

    assert counted.total == 2 and in_effect.total == 0


@pytest.mark.asyncio
async def test_an_undone_entry_without_a_dedup_key_is_not_counted(monkeypatch):
    shard = ShardItems(items=[_item("Ann", None), _item("Bob", None, undone=True)])
    _stub_llm(monkeypatch, lambda model, _: shard if model is ShardItems else NameGroups(groups=[]))

    result = await BroadRetriever().count_by_reading(
        CountPlan(source="text", item="a sale", group_by="seller", reversible=True), _units(1)
    )

    assert result.total == 1


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
async def test_keyed_items_sharing_a_templated_quote_in_one_piece_all_count(monkeypatch):
    """Fifty experiments in one piece each say "The run failed": distinct keys, distinct
    items. Only a keyless repeat of the same quote in the same unit is one occurrence."""
    shard = ShardItems(
        items=[
            *[_item(None, f"EXP-{n}") for n in range(50)],
            ExtractedItem(unit=0, evidence="the same quote"),
            ExtractedItem(unit=0, evidence="the same quote"),
        ]
    )
    for n in range(50):
        shard.items[n].evidence = "The run failed."
    _stub_llm(monkeypatch, lambda model, _: shard)

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(
        _plan(dedup_key="the experiment id"), _units(1)
    )

    assert result.total == 51


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
        # The model may put the @ form first; code tallies under the plain, longest spelling.
        return NameGroups(groups=[["@Akshats-git", "Akshats", "Akshats-git"], ["Megha-gbs"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="an assignment", group_by="assignee")

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.groups == [("Akshats-git", 3), ("Megha-gbs", 1)]


@pytest.mark.asyncio
async def test_aliases_stated_in_the_text_reach_the_merge_step(monkeypatch):
    """A roster line ("Akshats-git (usually called Akshats)") is the evidence a merge needs,
    once both spellings were read."""
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, group="Akshats", evidence="Akshats took #1"),
            ExtractedItem(unit=0, group="Akshats-git", evidence="Akshats-git took #3"),
            ExtractedItem(unit=0, group="Megha-gbs", evidence="Megha-gbs took #2"),
        ],
        aliases=[["Akshats-git", "Akshats"]],
    )
    merge_inputs = []

    def respond(model, text_input):
        if model is ShardItems:
            return shard
        merge_inputs.append(text_input)
        return NameGroups(groups=[["Akshats-git", "Akshats"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="an assignment", group_by="assignee")

    await BroadRetriever().count_by_reading(plan, _units(1))

    assert "Akshats-git = Akshats" in merge_inputs[0]


# --- one named target -----------------------------------------------------------------


def _assignments(*pairs: tuple[str, str]) -> ShardItems:
    return ShardItems(
        items=[
            ExtractedItem(unit=index, key=f"#{issue}", group=name, evidence=f"#{issue} to {name}")
            for index, (issue, name) in enumerate(pairs)
        ]
    )


@pytest.mark.asyncio
async def test_a_named_target_counts_its_group_under_every_spelling(monkeypatch):
    """ "Issues assigned to Megha": the handle and the nickname are one person."""
    shard = _assignments(
        ("1", "Megha-gbs"), ("2", "Megha"), ("3", "Akshats-git"), ("4", "Megha-gbs")
    )
    match_inputs = []

    def respond(model, text_input):
        if model is ShardItems:
            return shard
        if model is TargetMatch:
            match_inputs.append(text_input)
            return TargetMatch(names=["Megha-gbs"])
        return NameGroups(groups=[["Megha-gbs", "Megha"], ["Akshats-git"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text", item="an issue", group_by="assignee", target="Megha", dedup_key="issue"
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(4, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.total == 3
    # "Megha" is a merged spelling of Megha-gbs: known by code, and the model is
    # asked only whether any other name is the same person, anchored on that.
    assert "already known to be it: Megha-gbs" in match_inputs[0]
    assert "Akshats-git" in match_inputs[0] and "\nMegha-gbs" not in match_inputs[0]
    assert "TOTAL: 3" in context
    assert 'assignee is Megha-gbs (the names matching "Megha")' in context
    assert all(f"#{issue}" in context for issue in ("1", "2", "4"))
    assert "#3" not in context  # only the target's items are listed


@pytest.mark.asyncio
async def test_a_target_the_text_declares_equal_needs_no_model_call(monkeypatch):
    """ "Ann-dev, usually called Ann": asking about Ann finds Ann-dev by code, even when
    the alias group came back as one comma-joined string, and even if the model adds
    nothing. When Ann-dev is the only name there is nothing left to ask the model."""
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="#1", group="Ann-dev", evidence="#1 to Ann-dev"),
            ExtractedItem(unit=1, key="#2", group="Bob", evidence="#2 to Bob"),
        ],
        aliases=[["Ann-dev, Ann"]],
    )
    models = []

    def respond(model, _):
        models.append(model)
        if model is ShardItems:
            return shard
        if model is TargetMatch:
            return TargetMatch(names=[])
        return NameGroups(groups=[["Ann-dev"], ["Bob"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="a ticket", group_by="assignee", target="Ann")
    retriever = BroadRetriever(shard_tokens=10_000)

    result = await retriever.count_by_reading(plan, _units(2, words=2))
    alone = await retriever.match_target("Ann", ["Ann-dev"], [["Ann-dev", "Ann"]], {})

    assert result.total == 1 and result.target_names == ["Ann-dev"]
    assert alone == ["Ann-dev"] and models.count(TargetMatch) == 1


@pytest.mark.asyncio
async def test_a_translated_target_is_resolved_by_the_model(monkeypatch):
    """ "Deutschland" is no spelling of Germany: only the model can join them."""
    shard = ShardItems(
        items=[ExtractedItem(unit=0, key="S1", group="Germany", amount=40, evidence="40 from DE")]
    )
    models = []

    def respond(model, _):
        models.append(model)
        return shard if model is ShardItems else TargetMatch(names=["Germany"])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text", item="a shipment", group_by="origin", target="Deutschland", measure="units"
    )

    result = await BroadRetriever().count_by_reading(plan, _units(1))

    assert result.total == 40 and TargetMatch in models


@pytest.mark.asyncio
async def test_every_item_of_a_named_target_is_listed(monkeypatch):
    """ "Which issues": all of them, past the cap on quotes for a whole-corpus count."""
    count = broad_retriever.BROAD_EVIDENCE_SHOWN + 10
    shard = _assignments(*[(str(issue), "Akshats-git") for issue in range(count)])
    _stub_llm(
        monkeypatch,
        lambda model, _: TargetMatch(names=["Akshats-git"]) if model is TargetMatch else shard,
    )
    plan = CountPlan(
        source="text", item="an issue", group_by="assignee", target="Akshats-git", dedup_key="issue"
    )

    result = await BroadRetriever(shard_tokens=100_000).count_by_reading(plan, _units(count, 2))

    assert result.total == count
    assert len(result.evidence) == count


@pytest.mark.asyncio
async def test_a_named_target_nobody_matches_counts_zero(monkeypatch):
    """A name the matcher returns that was never read cannot create items."""
    shard = _assignments(("1", "Megha-gbs"), ("2", "Akshats-git"))

    def respond(model, _):
        if model is ShardItems:
            return shard
        if model is TargetMatch:
            return TargetMatch(names=["Ashkatosh"])
        return NameGroups(groups=[["Megha-gbs"], ["Akshats-git"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(source="text", item="an issue", group_by="assignee", target="Ashkatosh")

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(2, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.total == 0 and result.evidence == []
    assert "no listed item has assignee Ashkatosh" in context


@pytest.mark.asyncio
async def test_distinct_without_a_grouping_counts_the_items(monkeypatch):
    """ "How many people have the role SRE" planned as distinct with nothing to group
    by: the answer is the number of items, not the number of (zero) groups."""
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(source="text", item="a person with role SRE", distinct=True),
    )

    plan = await BroadRetriever().plan("How many people have the role SRE?", {})

    assert plan.distinct is False


@pytest.mark.asyncio
async def test_a_target_without_its_attribute_gets_one_retry_then_fails(monkeypatch):
    inputs = []

    def respond(model, text_input):
        inputs.append(text_input)
        group_by = "reagent" if "without group_by" in text_input else None
        return CountPlan(source="text", item="a run", target="RX-7", group_by=group_by)

    _stub_llm(monkeypatch, respond)

    plan = await BroadRetriever().plan("How many milligrams of RX-7 were used?", {})

    assert plan.group_by == "reagent" and len(inputs) == 2

    _stub_llm(monkeypatch, lambda model, _: CountPlan(source="text", item="a run", target="RX-7"))
    with pytest.raises(ValueError, match="target"):
        await BroadRetriever().plan("How many milligrams of RX-7 were used?", {})


@pytest.mark.asyncio
async def test_a_named_target_counts_its_items_not_different_values(monkeypatch):
    _stub_llm(
        monkeypatch,
        lambda model, _: CountPlan(
            source="text", item="an issue", group_by="assignee", target="Megha", distinct=True
        ),
    )

    plan = await BroadRetriever().plan("How many issues does Megha have?", {})

    assert plan.distinct is False


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


# --- summing a measure -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_measure_is_summed_per_item_and_per_group(monkeypatch):
    """ "How many units were imported": 40 + 12 + 30 units, not 3 shipments."""
    shard = ShardItems(
        items=[
            ExtractedItem(unit=0, key="SHP-1", group="Germany", amount=40, evidence="40 cameras"),
            ExtractedItem(unit=1, key="SHP-2", group="Germany", amount=12, evidence="12 lenses"),
            ExtractedItem(unit=2, key="SHP-3", group="Italy", amount=30, evidence="30 crates"),
            ExtractedItem(unit=3, key="SHP-4", group="Italy", evidence="olive oil, amount unclear"),
            ExtractedItem(unit=4, key="SHP-1", group="Germany", amount=40, evidence="recap SHP-1"),
        ]
    )

    def respond(model, _):
        if model is ShardItems:
            return shard
        return NameGroups(groups=[["Germany"], ["Italy"]])

    _stub_llm(monkeypatch, respond)
    plan = CountPlan(
        source="text",
        item="a shipment",
        group_by="origin country",
        measure="quantity in units",
        dedup_key="shipment id",
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(5, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.total == 82  # the recap of SHP-1 is not added twice
    assert dict(result.groups) == {"Germany": 52, "Italy": 30}
    assert result.amounts_missing == 1
    assert "TOTAL: 82" in context and "sum of quantity in units" in context
    assert "1 listed items stated no amount" in context


@pytest.mark.asyncio
async def test_a_measure_is_never_counted_as_word_mentions(monkeypatch):
    plan = _plan(literal_terms=["Germany"], measure="quantity")
    retriever = BroadRetriever()
    called = []

    async def reading(plan, units):
        called.append("reading")
        return CountResult(plan=plan, method="reading", total=0, units=0)

    async def plan_stub(query, entities_by_type):
        return plan

    async def units(graph_engine):
        return _units(1)

    monkeypatch.setattr(retriever, "count_by_reading", reading)
    monkeypatch.setattr(retriever, "plan", plan_stub)
    monkeypatch.setattr(retriever, "load_text_units", units)
    _use_graph(monkeypatch, _FakeGraph())

    await retriever.get_retrieved_objects("How many units came from Germany?")

    assert called == ["reading"]


# --- listing the counted items ---------------------------------------------------------


def _stub_answer(monkeypatch, text: str):
    """The answer LLM (the base class's completion step) returns ``text``."""

    async def fake(self, query, retrieved_objects, context=None, **kwargs):
        return [text]

    monkeypatch.setattr(broad_retriever.CompletionRetriever, "get_completion_from_context", fake)


@pytest.mark.asyncio
async def test_a_listing_request_appends_every_item_by_code(monkeypatch):
    """ "List them": all 981 items, written by code so none can be dropped."""
    _stub_answer(monkeypatch, "981 pull requests were found.")
    result = CountResult(
        plan=_plan(list_items=True),
        method="reading",
        total=981,
        units=28,
        evidence=[f"#{number}: opened PR #{number}" for number in range(981)],
    )

    [answer] = await BroadRetriever().get_completion_from_context("q", result, context="c")

    assert answer.startswith("981 pull requests were found.")
    assert all(f"- #{number}: opened PR #{number}" in answer for number in range(981))


@pytest.mark.asyncio
async def test_no_list_is_appended_unless_asked(monkeypatch):
    _stub_answer(monkeypatch, "42 rows.")
    result = CountResult(plan=_plan(), method="reading", total=42, units=6, evidence=["a", "b"])

    assert await BroadRetriever().get_completion_from_context("q", result, context="c") == [
        "42 rows."
    ]


@pytest.mark.asyncio
async def test_nothing_counted_appends_no_empty_list(monkeypatch):
    _stub_answer(monkeypatch, "0 issues were assigned to Ashkatosh.")
    result = CountResult(plan=_plan(list_items=True), method="reading", total=0, units=34)

    [answer] = await BroadRetriever().get_completion_from_context("q", result, context="c")
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert answer == "0 issues were assigned to Ashkatosh."
    assert "appended below" not in context


@pytest.mark.asyncio
async def test_listing_different_things_lists_the_groups(monkeypatch):
    _stub_answer(monkeypatch, "2 people.")
    result = CountResult(
        plan=_plan(group_by="assignee", distinct=True, list_items=True),
        method="reading",
        total=2,
        units=1,
        groups=[("Megha-gbs", 3), ("Akshats-git", 1)],
        evidence=["a", "b", "c", "d"],
    )

    [answer] = await BroadRetriever().get_completion_from_context("q", result, context="c")

    assert "- Megha-gbs (3)" in answer and "- Akshats-git (1)" in answer
    assert "- a" not in answer


@pytest.mark.asyncio
async def test_the_answer_context_defers_the_full_list_to_code():
    count = broad_retriever.BROAD_EVIDENCE_SHOWN + 25
    result = CountResult(
        plan=_plan(list_items=True),
        method="reading",
        total=count,
        units=3,
        evidence=[f"quote {number}" for number in range(count)],
    )

    context = await BroadRetriever().get_context_from_objects("q", result)

    assert "appended below your answer by code" in context
    assert f"quote {broad_retriever.BROAD_EVIDENCE_SHOWN - 1}" in context
    assert f"quote {count - 1}" not in context  # the context carries a sample only


@pytest.mark.asyncio
async def test_a_graph_count_keeps_every_matching_name(monkeypatch):
    count = broad_retriever.BROAD_EVIDENCE_SHOWN + 10
    entities = {
        "person": [
            Unit(id=f"p{i}", text=f"rostov {i}: x", name=f"rostov {i}") for i in range(count)
        ]
    }
    plan = CountPlan(source="entities", entity_types=["person"], item="a person")

    result = BroadRetriever().count_entities(plan, entities)

    assert len(result.evidence) == count


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
        source="text",
        item="a responder",
        group_by="responder",
        distinct=True,
        dedup_key="incident id",
    )

    result = await BroadRetriever(shard_tokens=10_000).count_by_reading(plan, _units(3, words=2))
    context = await BroadRetriever().get_context_from_objects("q", result)

    assert result.items_listed == 3  # three different incident ids survive the dedup
    assert result.total == 2  # the answer is the number of different responders
    assert "TOTAL: 2" in context and "number of different responder values" in context
    assert all("INC-" not in text for text in merge_inputs)  # keys never reach the merge step
