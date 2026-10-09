import asyncio
import importlib
import re
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models.FactConflict import FactConflict
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.review_conflicts import read_facts
from cognee.tasks.memify.review_conflicts.models import ReviewScope

review_facts = importlib.import_module("cognee.tasks.memify.review_conflicts.review_facts")
schema = importlib.import_module("cognee.tasks.memify.review_conflicts.schema")
trim_facts_to_budget = importlib.import_module(
    "cognee.tasks.memify.review_conflicts.trim_facts_to_budget"
)


class WordTokenizer:
    def count_tokens(self, text):
        return len(text.split())


class CountingTokenizer(WordTokenizer):
    """Counts measurements of the whole multi-line prompt, not of single lines."""

    def __init__(self):
        self.whole_text_counts = 0

    def count_tokens(self, text):
        self.whole_text_counts += "\n" in text
        return super().count_tokens(text)


def fact(fact_id, target, date, *, relationship="has_ceo", text=None, source=None):
    return {
        "id": fact_id,
        "source": source or ("acme" if relationship != "contains" else f"chunk-{fact_id}"),
        "target": target,
        "relationship": relationship,
        "properties": {"edge_text": text or f"{target} is Acme's CEO", "conflict_marks": []},
        "sources": [
            {
                "chunk_id": f"chunk-{fact_id}",
                "data_id": f"doc-{fact_id}",
                "document": f"{fact_id}.txt",
                "effective_date": date,
            }
        ],
        "effective_date": date,
        "observed_at": None,
    }


def competing_context(scope):
    """Alice competes with Bob for Acme's CEO, so Alice's own facts become Acme's context."""
    scope.facts["e5"] = fact("e5", "english", "2026-06-01", relationship="speaks", source="alice")
    scope.facts["e6"] = fact("e6", "german", "2026-06-02", relationship="speaks", source="alice")
    scope.facts["e7"] = fact("e7", "acme", "2019-01-01", relationship="contains")


@pytest.fixture
def scope():
    entities = {
        name: {
            "id": name,
            "name": name.title(),
            "types": ["Company" if name == "acme" else "Person"],
        }
        for name in ("acme", "alice", "bob")
    }
    return ReviewScope(
        dataset_id="dataset",
        entities=entities,
        facts={"e1": fact("e1", "alice", "2020-05-01"), "e2": fact("e2", "bob", "2026-01-10")},
        nodes=entities,
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(review_facts, "get_llm_tokenizer", lambda: WordTokenizer())
    monkeypatch.setattr(review_facts, "get_llm_token_ceiling", lambda: 20_000)


def response(**conflict):
    return schema.ReviewOutput(
        descriptions=[
            schema.ReviewedEntity(entity=f"n{i}", description=name)
            for i, name in enumerate(("Acme", "Alice", "Bob"), 1)
        ],
        conflicts=[
            schema.ReviewedConflict(
                about="n1",
                attribute="ceo",
                kind="time_varying",
                text="Bob succeeded Alice.",
                current=["r2"],
                superseded=["r1"],
                **conflict,
            )
        ],
    )


def validate(scope, output):
    call = review_facts.ReviewCall(scope, list(scope.entities))
    return review_facts.validate_review_output(output, call, call.render())


def test_dated_conflict_uses_relationship_identity_and_preserves_facts(scope):
    before = deepcopy(scope.facts)
    batch = validate(scope, response())
    accepted = batch.conflicts[0]
    assert accepted.conflict.attribute == "has_ceo"
    assert accepted.conflict.status == "resolved"
    assert accepted.conflict.values == ["alice", "bob"]
    assert accepted.conflict.sources == ["chunk-e1", "chunk-e2"]
    assert accepted.fact_statuses == {"e2": "current", "e1": "superseded"}
    assert batch.scope is scope
    assert scope.facts == before


@pytest.mark.parametrize(
    "case", ["undated", "time_words", "fixed", "restored", "reversed", "mixed"]
)
def test_status_rules(scope, case):
    output = response()
    result = output.conflicts[0]
    expected = "resolved"
    if case in {"undated", "time_words"}:
        for value in scope.facts.values():
            value["effective_date"] = None
            value["sources"][0]["effective_date"] = None
        if case == "undated":
            result.current = result.superseded = []
            result.conflicting = ["r1", "r2"]
            expected = "unresolved"
        else:
            scope.facts["e1"]["properties"]["edge_text"] = "Alice was the former CEO."
    elif case == "fixed":
        result.kind = "fixed"
        expected = "unresolved"
    elif case == "restored":
        scope.facts["e1"]["effective_date"] = "2027-03-01"
        result.current, result.superseded = ["r1"], ["r2"]
    else:
        scope.facts["e1"]["effective_date"] = "2027-03-01"
        expected = "unresolved"
        if case == "mixed":
            scope.facts["e1"]["relationship"] = "led_by"
            result.attribute = "CEO's Office"
    accepted = validate(scope, output).conflicts[0]
    assert accepted.conflict.status == expected
    if expected == "unresolved":
        assert set(accepted.fact_statuses.values()) == {"conflicting"}
    if case in {"reversed", "mixed"}:
        assert "dates disagree" in accepted.conflict.text
    if case == "mixed":
        assert accepted.conflict.attribute == "ceos office"


@pytest.mark.parametrize(
    "case",
    ["unknown", "overlap", "unrelated", "duplicate", "no_descriptions", "unknown_subject", "empty"],
)
def test_invalid_output_rejects_call(scope, case):
    output = response()
    if case == "unknown":
        output.conflicts[0].current = ["r99"]
    elif case == "overlap":
        output.conflicts[0].current = ["r1"]
    elif case == "unrelated":
        output.conflicts[0].about = "n2"
    elif case == "duplicate":
        output.conflicts.append(output.conflicts[0])
    elif case == "no_descriptions":
        output.descriptions.clear()
    elif case == "unknown_subject":
        output.descriptions[0].entity = "n99"
    else:
        output.conflicts[0].current = output.conflicts[0].superseded = []
    with pytest.raises(ValueError):
        validate(scope, output)


def test_conflict_about_an_undescribed_subject_is_not_accepted(scope):
    output = response()
    output.descriptions = [item for item in output.descriptions if item.entity != "n1"]
    batch = validate(scope, output)
    assert batch.conflicts == []
    assert batch.unreviewed_entity_ids == ["acme"]


def existing_conflict(scope):
    conflict = FactConflict(
        dataset_id="dataset",
        about_id="acme",
        attribute="has_ceo",
        kind="time_varying",
        status="resolved",
        text="Old explanation",
        values=["alice", "bob"],
        sources=["chunk-e1", "chunk-e2"],
    )
    scope.conflicts[str(conflict.id)] = conflict.model_dump(mode="json")
    for value in scope.facts.values():
        value["properties"]["conflict_marks"] = [
            {"conflict_id": str(conflict.id), "status": "current"}
        ]
    return str(conflict.id)


def test_drop_checks_all_surviving_support_and_replacement_collision(scope):
    conflict_id = existing_conflict(scope)
    output = response()
    output.drop = ["f1"]
    with pytest.raises(ValueError, match="replaced drop"):
        validate(scope, output)
    output.conflicts = []
    assert validate(scope, output).dropped_conflict_ids == [conflict_id]
    call = review_facts.ReviewCall(scope, list(scope.entities))
    for ids in call.shown_facts.values():
        ids.remove("e1")
    assert (
        review_facts.validate_review_output(output, call, call.render()).dropped_conflict_ids == []
    )


def test_partial_answer_writes_nothing_for_an_undescribed_subject(scope):
    existing_conflict(scope)
    output = response()
    output.conflicts = []
    output.drop = ["f1"]
    output.descriptions = [item for item in output.descriptions if item.entity != "n1"]
    batch = validate(scope, output)
    assert set(batch.descriptions) == {"alice", "bob"}
    assert batch.unreviewed_entity_ids == ["acme"]
    assert batch.dropped_conflict_ids == []


def test_prompt_uses_sources_and_types_without_stored_descriptions(scope):
    scope.entities["acme"]["description"] = "STALE_DESCRIPTION"
    existing_conflict(scope)
    call = review_facts.ReviewCall(scope, ["acme"])
    text = call.render().text
    assert "e1.txt (2020-05-01)" in text
    assert "Company" in text and "Competing relationships:" in text
    assert "STALE_DESCRIPTION" not in text and "Old explanation" not in text
    assert "[f1] has_ceo" in text


def test_existing_action_labels_belong_only_to_batch_subjects(scope):
    conflict_id = existing_conflict(scope)
    call = review_facts.ReviewCall(scope, ["alice"])
    text = call.render().text
    assert "Existing has_ceo (value)" in text
    assert call.render().conflicts == {}
    assert "[f1]" not in text
    output = schema.ReviewOutput(
        descriptions=[schema.ReviewedEntity(entity="n1", description="Alice's facts")],
        drop=["f1"],
    )
    with pytest.raises(ValueError, match="Unknown, repeated, or replaced drop"):
        review_facts.validate_review_output(output, call, call.render())

    owned_call = review_facts.ReviewCall(scope, ["acme", "alice"])
    assert owned_call.render().conflicts == {"f1": conflict_id}


def test_pending_subject_header_requests_reconciliation_without_auto_acceptance(scope):
    conflict_id = existing_conflict(scope)
    scope.conflicts[conflict_id]["review_pending"] = True
    call = review_facts.ReviewCall(scope, ["acme"])
    assert "Existing [f1] has_ceo (subject; pending write)" in call.render().text
    value_call = review_facts.ReviewCall(scope, ["alice"])
    assert "pending write" not in value_call.render().text

    # Incomplete evidence still permits omission, and cannot authorize a drop.
    call.shown_facts["acme"].remove("e1")
    rendered = call.render()
    output = schema.ReviewOutput(
        descriptions=[schema.ReviewedEntity(entity="n1", description="Shown facts")],
        drop=["f1"],
    )
    batch = review_facts.validate_review_output(output, call, rendered)
    assert batch.conflicts == [] and batch.dropped_conflict_ids == []
    assert scope.conflicts[conflict_id]["review_pending"] is True


def test_legacy_textless_relationship_and_unnamed_sources(scope):
    del scope.facts["e1"]["properties"]["edge_text"]
    scope.facts["e1"]["sources"][0]["document"] = None
    scope.facts["e2"]["properties"]["edge_text"] = None
    scope.facts["e2"]["sources"][0].update(document=None, data_id=None)
    before = deepcopy(scope.facts)
    text = review_facts.ReviewCall(scope, ["acme"]).render().text
    assert "| has_ceo | doc-e1 (2020-05-01)" in text
    assert "| has_ceo | unnamed document (2026-01-10)" in text
    assert scope.facts == before


@pytest.mark.parametrize("cap,shown", [(15, ["e4"]), (25, ["e4", "e1"]), (40, ["e4", "e1", "e2"])])
def test_trim_priority_and_marks_remain_intact(scope, monkeypatch, cap, shown):
    existing_conflict(scope)
    scope.facts["e2"]["properties"]["conflict_marks"] = []
    scope.conflicts.clear()
    scope.facts["e3"] = fact("e3", "acme", "2026-09-01", relationship="contains")
    scope.facts["e4"] = fact("e4", "acme", "2020-01-01", relationship="contains")
    scope.facts["e4"]["observed_at"] = datetime(2026, 9, 2, tzinfo=timezone.utc)
    scope.since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    # Existing conflict support has a citation and keeps its mark when omitted.
    scope.conflicts["saved"] = {
        "id": "saved",
        "about_id": "acme",
        "attribute": "title",
        "sources": ["chunk-e1"],
        "values": [],
    }
    call = review_facts.ReviewCall(scope, ["acme"])
    assert call.shown_facts["acme"] == ["e4", "e1", "e2", "e3"]
    before = deepcopy(scope.facts)
    monkeypatch.setattr(trim_facts_to_budget, "FACT_TOKENS_PER_ENTITY", cap)
    trim_facts_to_budget.fit_to_budget(
        call, trim_facts_to_budget.TokenBudget(WordTokenizer(), 0, 10_000)
    )
    assert call.shown_facts["acme"] == shown
    assert scope.facts == before
    scope.since = None
    call = review_facts.ReviewCall(scope, ["acme"])
    assert call.shown_facts["acme"] == ["e1", "e2", "e3", "e4"]


def test_context_facts_rank_below_the_subjects_own_facts(scope):
    # validate_review_output only accepts a fact that touches the subject, so context can never be one
    # of its conflict sides. Ranking by priority alone gave ["e6", "e5", "e2", "e1", "e7"].
    competing_context(scope)
    call = review_facts.ReviewCall(scope, ["acme"])
    assert call.shown_facts["acme"] == ["e2", "e1", "e7", "e6", "e5"]


def test_chunk_statements_rank_below_ungrouped_relationship_facts(scope):
    # Only e5 and e6 move: one tier held both, ordered by date, so the newer statement won.
    scope.facts["e5"] = fact("e5", "leeds", "2026-02-01", relationship="has_headquarters")
    scope.facts["e6"] = fact("e6", "acme", "2026-03-01", relationship="contains")
    call = review_facts.ReviewCall(scope, ["acme"])
    assert call.shown_facts["acme"] == ["e2", "e1", "e5", "e6"]


def test_undated_competing_facts_outrank_a_newer_ungrouped_fact(scope):
    # Both sides of a fixed-attribute conflict are often undated. The competing tier is
    # what keeps them in the section; without it the date tiebreak sorts them last.
    for value in scope.facts.values():
        value["effective_date"] = None
        value["sources"][0]["effective_date"] = None
    scope.facts["e5"] = fact("e5", "leeds", "2026-02-01", relationship="has_headquarters")
    call = review_facts.ReviewCall(scope, ["acme"])
    assert call.shown_facts["acme"] == ["e1", "e2", "e5"]


def test_whole_call_trim_drops_context_before_any_direct_fact(scope, monkeypatch):
    # Zeta's only fact is an older statement, so a run-level comparator donates it first.
    competing_context(scope)
    scope.entities["zeta"] = {"id": "zeta", "name": "Zeta", "types": []}
    scope.facts["e8"] = fact("e8", "zeta", "2018-01-01", relationship="contains")
    monkeypatch.setattr(trim_facts_to_budget, "FACT_TOKENS_PER_ENTITY", 10_000)
    tokenizer = WordTokenizer()
    call = review_facts.ReviewCall(scope, ["zeta", "alice"])
    before = {subject: list(ids) for subject, ids in call.shown_facts.items()}
    trim_facts_to_budget.fit_to_budget(
        call,
        trim_facts_to_budget.TokenBudget(
            tokenizer, 0, tokenizer.count_tokens(call.render().text) - 1
        ),
    )
    assert [
        (subject, fact_id)
        for subject, ids in before.items()
        for fact_id in ids
        if fact_id not in call.shown_facts[subject]
    ] == [("alice", "e7")]


def test_trim_keeps_every_fact_the_cap_allows(scope, monkeypatch):
    """The per-subject cut is the longest prefix that fits, never a rounded batch."""
    for index in range(3, 16):
        scope.facts[f"e{index}"] = fact(f"e{index}", f"person{index}", "2020-01-01")
    tokenizer = WordTokenizer()
    call = review_facts.ReviewCall(scope, ["acme"])
    order = list(call.shown_facts["acme"])
    lines = call.render().fact_lines["acme"]
    monkeypatch.setattr(
        trim_facts_to_budget,
        "FACT_TOKENS_PER_ENTITY",
        sum(tokenizer.count_tokens(line) for line in lines[:9]),
    )
    trim_facts_to_budget.fit_to_budget(call, trim_facts_to_budget.TokenBudget(tokenizer, 0, 10_000))
    assert call.shown_facts["acme"] == order[:9]


def test_trim_stays_cheap_when_the_call_is_barely_over_budget(scope, monkeypatch):
    """One fact over budget must not pay a search across the whole fact list."""
    for index in range(3, 203):
        scope.facts[f"e{index}"] = fact(f"e{index}", f"person{index}", "2020-01-01")
    monkeypatch.setattr(trim_facts_to_budget, "FACT_TOKENS_PER_ENTITY", 10_000)
    tokenizer = CountingTokenizer()
    call = review_facts.ReviewCall(scope, ["acme"])
    limit = tokenizer.count_tokens(call.render().text) - 1
    before = tokenizer.whole_text_counts
    rendered = trim_facts_to_budget.fit_to_budget(
        call, trim_facts_to_budget.TokenBudget(tokenizer, 0, limit)
    )
    # Two: the budget probes zero drops, then one. The per-subject cut renders but
    # measures only single lines, which this tokenizer does not count.
    assert tokenizer.whole_text_counts - before <= 4
    assert len(call.shown_facts["acme"]) == 201
    assert tokenizer.count_tokens(rendered.text) <= limit


def test_trim_measures_the_whole_prompt_a_bounded_number_of_times(scope):
    """A drop must not cost a re-render and a re-count of the whole call."""
    for index in range(3, 203):
        scope.facts[f"e{index}"] = fact(f"e{index}", f"person{index}", "2020-01-01")
    tokenizer = CountingTokenizer()
    call = review_facts.ReviewCall(scope, ["acme"])
    limit = tokenizer.count_tokens(call.render().text) // 4
    rendered = trim_facts_to_budget.fit_to_budget(
        call, trim_facts_to_budget.TokenBudget(tokenizer, 0, limit)
    )
    # Seventeen: the limit, the per-subject cut, then the doubling probes and the bisect.
    # Measuring once per dropped fact needed 154 here.
    assert tokenizer.whole_text_counts <= 30
    assert tokenizer.count_tokens(rendered.text) <= limit


def test_competing_line_names_only_labels_in_that_subjects_section(scope):
    # A label trimmed out of this section is not a candidate here, even though its fact
    # line survives under another subject: validate_review_output rejects it and discards the batch.
    competing_context(scope)
    call = review_facts.ReviewCall(scope, ["acme", "alice"])
    call.shown_facts["acme"] = ["e2", "e7"]
    acme_section, alice_section = call.render().text.split("\n\n")[:2]
    assert "Competing relationships" not in acme_section
    assert "Competing relationships: r1, r2" in alice_section


async def descriptions_only(**kwargs):
    labels = re.findall(r"Subject \[(n\d+)\]", kwargs["text_input"])
    return schema.ReviewOutput(
        descriptions=[
            schema.ReviewedEntity(entity=label, description="Surviving facts.") for label in labels
        ]
    )


@pytest.mark.asyncio
async def test_generator_auto_drop_before_calls_and_no_output_cap(scope, monkeypatch):
    scope.drop_conflict_ids = ["orphan"]
    llm = AsyncMock(side_effect=descriptions_only)
    monkeypatch.setattr(review_facts.LLMGateway, "acreate_structured_output", llm)
    stream = review_facts.review_entities(scope)
    assert (await anext(stream)).dropped_conflict_ids == ["orphan"]
    llm.assert_not_awaited()
    batches = [batch async for batch in stream]
    assert set(batches[0].descriptions) == set(scope.entities)
    assert batches[0].conflicts == []  # Multivalued or compatible facts need no conflict.
    assert batches[-1].final
    assert batches[-1].unreviewed_entity_ids == []
    assert "max_completion_tokens" not in llm.call_args.kwargs


@pytest.mark.asyncio
async def test_failed_call_splits_once_and_continues(scope, monkeypatch):
    llm = AsyncMock(
        side_effect=[
            ValueError("bad batch"),
            ValueError("bad first half"),
            await descriptions_only(text_input="Subject [n1] Alice\nSubject [n2] Bob"),
        ]
    )
    monkeypatch.setattr(review_facts.LLMGateway, "acreate_structured_output", llm)
    batches = [batch async for batch in review_facts.review_entities(scope)]
    assert llm.await_count == 3
    assert batches[-1].unreviewed_entity_ids == ["acme"]
    assert set(batches[0].descriptions) == {"alice", "bob"}
    assert batches[-1].final


@pytest.mark.asyncio
async def test_partial_descriptions_reach_the_final_batch(scope, monkeypatch):
    async def omit_first(**kwargs):
        output = await descriptions_only(**kwargs)
        output.descriptions = output.descriptions[1:]
        return output

    monkeypatch.setattr(
        review_facts.LLMGateway, "acreate_structured_output", AsyncMock(side_effect=omit_first)
    )
    batches = [batch async for batch in review_facts.review_entities(scope)]
    assert set(batches[0].descriptions) == {"alice", "bob"}
    assert batches[0].unreviewed_entity_ids == ["acme"]
    assert batches[-1].final and batches[-1].unreviewed_entity_ids == ["acme"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [LLMPaymentRequiredError(), asyncio.CancelledError()])
async def test_budget_and_cancellation_propagate_without_retry(scope, monkeypatch, error):
    llm = AsyncMock(side_effect=error)
    monkeypatch.setattr(review_facts.LLMGateway, "acreate_structured_output", llm)
    with pytest.raises(type(error)):
        _ = [batch async for batch in review_facts.review_entities(scope)]
    assert llm.await_count == 1


@pytest.mark.asyncio
async def test_oversized_header_skipped_and_call_budget_enforced(scope, monkeypatch):
    scope.entities["acme"]["name"] = "very long name " * 10_000
    monkeypatch.setattr(review_facts, "get_llm_token_ceiling", lambda: 2_000)
    llm = AsyncMock(side_effect=descriptions_only)
    monkeypatch.setattr(review_facts.LLMGateway, "acreate_structured_output", llm)
    batches = [batch async for batch in review_facts.review_entities(scope)]
    assert batches[-1].unreviewed_entity_ids == ["acme"]
    assert set(batches[0].descriptions) == {"alice", "bob"}
    kwargs = llm.call_args.kwargs
    tokens = WordTokenizer().count_tokens(kwargs["system_prompt"] + kwargs["text_input"])
    tokens += WordTokenizer().count_tokens(
        review_facts.json.dumps(schema.ReviewOutput.model_json_schema(), sort_keys=True)
    )
    assert tokens <= 1_000


# Every review call pays words(prompt) + words(schema) before its first fact line,
# so growth here drops facts silently instead of failing. The budget test above
# gives a call 1_000 words, and the scope fixture's alice and bob need 70
# for headers and facts; past 990 they stop sharing a call and that test fails as
# a lost subject.
PROMPT_OVERHEAD_WORDS = 1_000 - 70


def test_system_prompt_leaves_room_for_facts():
    overhead = WordTokenizer().count_tokens(
        render_prompt("review_conflicts.txt", {})
        + " "
        + review_facts.json.dumps(schema.ReviewOutput.model_json_schema(), sort_keys=True)
    )
    assert overhead <= PROMPT_OVERHEAD_WORDS


@pytest.mark.asyncio
async def test_pipeline_tasks_preserve_reader_object_and_writer_batch(monkeypatch):
    entity = {"id": "acme", "name": "Acme", "type": "Entity", "description": "Old"}
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            side_effect=lambda filters: ([("acme", entity)] if "id" in filters[0] else [], [])
        ),
        get_nodes=AsyncMock(return_value=[entity]),
        get_neighborhood=AsyncMock(return_value=([("acme", entity)], [])),
    )
    monkeypatch.setattr(read_facts, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        read_facts,
        "get_improve_config",
        lambda: SimpleNamespace(effective_date_key="effective_date"),
    )
    monkeypatch.setattr(
        review_facts.LLMGateway,
        "acreate_structured_output",
        AsyncMock(side_effect=descriptions_only),
    )
    written = []

    async def write_batch(data):
        assert isinstance(data, list) and len(data) == 1
        assert isinstance(data[0], review_facts.ReviewBatch)
        written.append(data[0])

    reader = Task(read_facts.read_entity_facts, entity_ids=["acme"], needs_llm=False)
    reviewer = Task(review_facts.review_entities)
    writer = Task(write_batch, batch_size=1, needs_llm=False)
    ctx = PipelineContext(dataset=SimpleNamespace(id=uuid4()))
    async for scope in reader.execute([[{}]], {"ctx": ctx}):
        assert isinstance(scope, ReviewScope)
        async for batches in reviewer.execute([scope], {}, writer.task_config["batch_size"]):
            async for _ in writer.execute([batches], {}):
                pass
    assert written[0].descriptions == {"acme": "Surviving facts."}
    assert written[1].final


def test_reused_index_reads_support_changed_by_previous_batch(scope):
    scope.facts["e3"] = fact("e3", "acme", "2026-09-01", relationship="contains")
    fact_index = review_facts.FactIndex(scope.facts)
    assert review_facts.ReviewCall(scope, ["acme"], fact_index).shown_facts["acme"] == [
        "e2",
        "e1",
        "e3",
    ]

    # The writer updates the shared input between yielded batches. Neither marks
    # nor citations may be cached with the immutable fact index.
    scope.conflicts["saved"] = {
        "id": "saved",
        "about_id": "acme",
        "attribute": "title",
        "sources": ["chunk-e1"],
        "values": [],
    }
    scope.facts["e3"]["properties"]["conflict_marks"] = [
        {"conflict_id": "saved", "status": "current"}
    ]
    assert review_facts.ReviewCall(scope, ["acme"], fact_index).shown_facts["acme"] == [
        "e3",
        "e1",
        "e2",
    ]

    scope.facts["e3"]["properties"]["conflict_marks"].clear()
    scope.conflicts["saved"]["sources"] = ["chunk-e2"]
    assert review_facts.ReviewCall(scope, ["acme"], fact_index).shown_facts["acme"] == [
        "e2",
        "e1",
        "e3",
    ]


@pytest.mark.asyncio
async def test_batch_planning_builds_fact_context_only_for_llm_calls(monkeypatch):
    entities = {str(i): {"id": str(i), "name": f"Entity {i}"} for i in range(23)}
    data = ReviewScope(dataset_id="dataset", entities=entities, facts={})
    calls = []
    original_init = review_facts.ReviewCall.__init__

    def record_call(self, scope, subjects, fact_index=None):
        calls.append((list(subjects), fact_index))
        original_init(self, scope, subjects, fact_index)

    monkeypatch.setattr(review_facts.ReviewCall, "__init__", record_call)
    monkeypatch.setattr(
        review_facts.LLMGateway,
        "acreate_structured_output",
        AsyncMock(side_effect=descriptions_only),
    )
    batches = [batch async for batch in review_facts.review_entities(data)]
    assert [len(batch.descriptions) for batch in batches[:-1]] == [10, 10, 3]
    assert len(calls) == 3
    assert calls[0][1] is calls[1][1] is calls[2][1]


def test_validation_uses_the_snapshot_sent_to_the_llm(scope):
    call = review_facts.ReviewCall(scope, list(scope.entities))
    sent = trim_facts_to_budget.fit_to_budget(
        call, trim_facts_to_budget.TokenBudget(WordTokenizer(), 0, 10_000)
    )
    for facts in call.shown_facts.values():
        facts.remove("e1")
    later = call.render()
    assert later.facts != sent.facts
    batch = review_facts.validate_review_output(response(), call, sent)
    assert batch.shown_fact_ids == {"e1", "e2"}
    assert batch.conflicts[0].fact_statuses == {"e2": "current", "e1": "superseded"}


@pytest.mark.parametrize(
    "other, kept",
    [("2015-05-12", True), ("Bristol", False)],
    ids=["two dates compete", "a date beside a place does not"],
)
def test_a_date_competes_only_with_another_date(scope, other, kept):
    for name in ("2014-05-12", other):
        scope.entities[name] = {"id": name, "name": name, "types": []}
        scope.nodes[name] = scope.entities[name]
    scope.facts = {
        "e1": fact("e1", "2014-05-12", None, relationship="incorporated_on"),
        "e2": fact("e2", other, "2026-01-10", relationship="incorporated_on"),
    }
    output = schema.ReviewOutput(
        descriptions=[
            schema.ReviewedEntity(entity=f"n{i}", description=name)
            for i, name in enumerate(scope.entities, 1)
        ],
        conflicts=[
            schema.ReviewedConflict(
                about="n1",
                attribute="incorporated_on",
                kind="fixed",
                text="Two incorporation dates are recorded.",
                conflicting=["r1", "r2"],
            )
        ],
    )
    batch = validate(scope, output)
    assert [item.conflict.values for item in batch.conflicts] == (
        [sorted(["2014-05-12", other])] if kept else []
    )
