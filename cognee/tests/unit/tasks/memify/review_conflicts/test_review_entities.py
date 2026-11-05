"""Which sections share a review call, what an oversized one loses, and how calls run."""

import asyncio
import importlib
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models.FactConflict import FactConflict
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.review_conflicts import read_facts
from cognee.tasks.memify.review_conflicts.build_review_prompt import build_review_prompt
from cognee.tasks.memify.review_conflicts.models import ConflictsToFinalize, ReviewScope
from cognee.tasks.memify.review_conflicts.select_review_facts import find_fact_comparisons
from cognee.tests.unit.tasks.memify.review_conflicts.review_builders import (
    alice_group_and_lone_statement,
    build_prompt,
    existing_conflict,
    fact,
    response,
)

review_entities = importlib.import_module("cognee.tasks.memify.review_conflicts.review_entities")
select_review_facts = importlib.import_module(
    "cognee.tasks.memify.review_conflicts.select_review_facts"
)
schema = importlib.import_module("cognee.tasks.memify.review_conflicts.schema")


class WordTokenizer:
    def count_tokens(self, text):
        return len(text.split())


def words(scope, subjects):
    """One section's size as the planner measures it, in words."""
    return WordTokenizer().count_tokens(build_prompt(scope, subjects).text)


def plan(scope, token_budget):
    """The calls the planner packs, each as the facts its prompt shows per subject."""
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    calls = review_entities._plan_calls(
        scope, comparisons, list(scope.entities), WordTokenizer(), token_budget, []
    )
    return [build_review_prompt(scope, comparisons, call).shown_facts for call in calls]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(review_entities, "get_llm_tokenizer", lambda: WordTokenizer())
    monkeypatch.setattr(review_entities, "get_llm_token_ceiling", lambda: 16_384)


@pytest.fixture
def every_entity_a_subject(monkeypatch):
    """Review every entity of the scope, whether or not it has something to compare."""
    monkeypatch.setattr(
        review_entities, "select_subject_ids", lambda scope, comparisons: list(scope.entities)
    )


def test_calls_pack_whole_sections(scope):
    """A section is never cut: a subject moves to the next call rather than lose a fact."""
    alice_group_and_lone_statement(scope)
    acme, alice, bob = (words(scope, [subject]) for subject in ("acme", "alice", "bob"))
    assert plan(scope, acme + alice + bob) == [
        {"acme": ["e2", "e1"], "alice": ["e6", "e5"], "bob": []},
    ]
    assert plan(scope, max(acme, alice + bob)) == [
        {"acme": ["e2", "e1"]},
        {"alice": ["e6", "e5"], "bob": []},
    ]


def section_words(scope, fact_ids):
    """acme's section showing only these facts, as the planner measures it, in words."""
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    prompt = build_review_prompt(scope, comparisons, {"acme": fact_ids})
    return WordTokenizer().count_tokens(prompt.text)


def add_facts(scope, relationship, *people, date="2021-01-01"):
    """One fact per person, acme --relationship--> person, numbered after the existing ones."""
    scope.nodes = dict(scope.nodes)  # the fixture shares one dict for entities and nodes
    for person in people:
        scope.nodes[person] = {"id": person, "name": person.title(), "types": ["Person"]}
        fact_id = f"e{len(scope.facts) + 1}"
        scope.facts[fact_id] = fact(fact_id, person, date, relationship=relationship)


def test_oversized_section_loses_list_like_names_first(scope, monkeypatch):
    """works_with lists people and goes; the CEO facts still meet under their two names.

    A mark that another subject's conflict left on a works_with fact keeps nothing.
    """
    monkeypatch.setattr(select_review_facts, "MIN_LIST_VALUES", 3)
    scope.facts["e1"]["relationship"] = "is_ceo_of"
    scope.facts["e2"]["relationship"] = "ceo_of"
    add_facts(scope, "works_with", "carol", "dave", "erin")  # e3 to e5
    carols = FactConflict(
        dataset_id="dataset",
        about_id="carol",
        attribute="employer",
        kind="time_varying",
        status="resolved",
        text="Carol moved.",
        values=["acme", "bob"],
        sources=["chunk-e3"],
    )
    scope.conflicts[str(carols.id)] = carols.model_dump(mode="json")
    scope.facts["e3"]["properties"]["conflict_marks"] = [
        {"conflict_id": str(carols.id), "status": "superseded"}
    ]
    logger = Mock()
    monkeypatch.setattr(review_entities, "logger", logger)
    calls = plan(scope, section_words(scope, ["e1", "e2"]))
    assert [call["acme"] for call in calls if "acme" in call] == [["e2", "e1"]]
    assert "works_with" in str(logger.warning.call_args_list)


def test_oversized_section_loses_one_list_at_a_time_most_values_first(scope, monkeypatch):
    """works_with, four people, goes first; then advises, which ties with mentors at three
    and sorts first. mentors stays, because the section then fits."""
    monkeypatch.setattr(select_review_facts, "MIN_LIST_VALUES", 3)
    add_facts(scope, "works_with", "carol", "dave", "erin", "frank")  # e3 to e6
    add_facts(scope, "advises", "gina", "hank", "ivy")  # e7 to e9
    add_facts(scope, "mentors", "jack", "kate", "liam")  # e10 to e12
    kept = ["e1", "e2", "e10", "e11", "e12"]
    calls = plan(scope, section_words(scope, kept))
    assert [set(call["acme"]) for call in calls if "acme" in call] == [set(kept)]


def test_oversized_section_keeps_what_its_stored_conflict_needs(scope, monkeypatch):
    """Kept: has_ceo, which the stored conflict marks, with its new values; e4, which
    shares a cited chunk; and ceo_of, one value and so no list. Left out: e5 and e6."""
    monkeypatch.setattr(select_review_facts, "MIN_LIST_VALUES", 2)
    existing_conflict(scope)  # marks e1 and e2, cites chunk-e1 and chunk-e2
    add_facts(scope, "has_ceo", "carol", date="2027-01-01")  # e3
    add_facts(scope, "works_with", "dave", "erin", "frank")  # e4 to e6
    add_facts(scope, "has_ceo", "gina", date="2028-01-01")  # e7
    add_facts(scope, "ceo_of", "hank", date="2029-01-01")  # e8
    scope.facts["e4"]["sources"][0]["chunk_id"] = "chunk-e1"
    kept = ["e1", "e2", "e3", "e4", "e7", "e8"]
    calls = plan(scope, section_words(scope, kept))
    assert [set(call["acme"]) for call in calls if "acme" in call] == [set(kept)]


def test_statements_are_never_a_list(scope, monkeypatch):
    """Four paired statements cite four chunks, but a chunk is no value: they stay, and
    works_with, three people, goes."""
    monkeypatch.setattr(select_review_facts, "MIN_LIST_VALUES", 3)
    for fact_id, staff in (("e3", 120), ("e4", 150), ("e5", 180), ("e6", 210)):
        scope.facts[fact_id] = fact(
            fact_id, "acme", "2020-01-01", relationship="contains", text=f"Acme has {staff} staff"
        )
    add_facts(scope, "works_with", "carol", "dave", "erin")  # e7 to e9
    kept = ["e1", "e2", "e3", "e4", "e5", "e6"]
    calls = plan(scope, section_words(scope, kept))
    assert [set(call["acme"]) for call in calls if "acme" in call] == [set(kept)]


@pytest.mark.asyncio
async def test_section_that_only_fits_without_an_attribute_is_skipped(
    scope, monkeypatch, every_entity_a_subject
):
    """has_ceo and works_with have two values each, too few to be lists: neither is cut,
    so acme gets no call, and no retry stamp."""
    add_facts(scope, "works_with", "carol", "dave")
    budget = section_words(scope, ["e3", "e4"])
    assert words(scope, ["acme"]) > budget
    monkeypatch.setattr(review_entities, "_compute_user_prompt_token_budget", lambda *_: budget)
    llm = AsyncMock(side_effect=reviewed_only)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    _, failed = await review_all(scope)
    assert llm.await_count == 1
    assert not re.search(
        r"^Subject \[n\d+\] Acme ", llm.call_args.kwargs["text_input"], re.MULTILINE
    )
    assert failed == []


async def reviewed_only(**kwargs):
    labels = re.findall(r"Subject \[(n\d+)\]", kwargs["text_input"])
    return schema.ReviewResponse(reviewed=labels)


async def review_all(scope):
    """Every item the reviewer yields, and the subjects it reports it could not review."""
    failed = []
    items = [item async for item in review_entities.review_entities(scope, failed, [])]
    return items, failed


@pytest.mark.asyncio
async def test_generator_auto_drop_before_calls_and_no_output_cap(scope, monkeypatch):
    scope.conflict_ids_to_drop = ["orphan"]
    llm = AsyncMock(side_effect=reviewed_only)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    failed = []
    stream = review_entities.review_entities(scope, failed, [])
    assert (await anext(stream)).dropped_conflict_ids == ["orphan"]
    llm.assert_not_awaited()
    batches = [batch async for batch in stream]
    assert batches[0].accepted_conflicts == []  # Multivalued or compatible facts need no conflict.
    assert isinstance(batches[-1], ConflictsToFinalize)
    assert failed == []
    assert "max_completion_tokens" not in llm.call_args.kwargs


@pytest.mark.asyncio
async def test_failed_call_splits_once_and_continues(scope, monkeypatch, every_entity_a_subject):
    llm = AsyncMock(
        side_effect=[
            ValueError("bad batch"),
            ValueError("bad first half"),
            await reviewed_only(text_input="Subject [n1] Alice\nSubject [n2] Bob"),
        ]
    )
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    batches, failed = await review_all(scope)
    assert llm.await_count == 3
    assert failed == ["acme"]
    assert isinstance(batches[-1], ConflictsToFinalize)


@pytest.mark.asyncio
async def test_partial_answer_is_reported_to_the_caller(scope, monkeypatch, every_entity_a_subject):
    async def omit_first(**kwargs):
        output = await reviewed_only(**kwargs)
        output.reviewed = output.reviewed[1:]
        return output

    monkeypatch.setattr(
        review_entities.LLMGateway, "acreate_structured_output", AsyncMock(side_effect=omit_first)
    )
    batches, failed = await review_all(scope)
    assert batches[0].unreviewed_entity_ids == ["acme"]
    assert isinstance(batches[-1], ConflictsToFinalize)
    assert failed == ["acme"]


@pytest.mark.asyncio
async def test_review_end_lists_every_conflict_the_run_decided(
    scope, monkeypatch, every_entity_a_subject
):
    """The writer finalizes from ConflictsToFinalize alone: it must name each decided conflict."""
    scope.conflict_ids_to_drop = ["orphan"]
    monkeypatch.setattr(
        review_entities.LLMGateway, "acreate_structured_output", AsyncMock(return_value=response())
    )
    (drops, answer, end), failed = await review_all(scope)
    assert len(answer.accepted_conflicts) == 1
    assert end.accepted_conflicts == answer.accepted_conflicts
    assert end.dropped_conflict_ids == drops.dropped_conflict_ids == ["orphan"]
    assert failed == []


@pytest.mark.asyncio
async def test_next_call_selects_facts_after_the_last_write(
    scope, monkeypatch, every_entity_a_subject
):
    """Alice's section is selected after Acme's answer is written, so it shows what that write left.

    The write stores a conflict about Acme that marks the claim naming Alice as its CEO.
    The claim then supports a stored conflict, so Alice's section shows it.
    """
    monkeypatch.setattr(review_entities, "SUBJECTS_PER_CALL", 1)
    llm = AsyncMock(side_effect=reviewed_only)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    stream = review_entities.review_entities(scope, [], [])
    await anext(stream)  # Acme's answer
    existing_conflict(scope)  # what the writer leaves in the scope for a conflict about Acme
    _ = [item async for item in stream]
    alice_prompt = llm.call_args_list[1].kwargs["text_input"]
    assert alice_prompt.startswith("Subject [n1] Alice ")
    assert "alice is Acme's CEO" in alice_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [LLMPaymentRequiredError(), asyncio.CancelledError()])
async def test_budget_and_cancellation_propagate_without_retry(scope, monkeypatch, error):
    llm = AsyncMock(side_effect=error)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    with pytest.raises(type(error)):
        _ = [batch async for batch in review_entities.review_entities(scope, [], [])]
    assert llm.await_count == 1


@pytest.mark.asyncio
async def test_oversized_header_skipped_without_retry_and_call_budget_enforced(
    scope, monkeypatch, every_entity_a_subject
):
    """A header too large for a call is skipped like any section that cannot fit, and
    reported as skipped, not failed: a retry would only skip it again."""
    scope.entities["acme"]["name"] = "very long name " * 10_000
    monkeypatch.setattr(review_entities, "REVIEW_INPUT_TOKENS", 1_000)
    llm = AsyncMock(side_effect=reviewed_only)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    failed, skipped = [], []
    _ = [item async for item in review_entities.review_entities(scope, failed, skipped)]
    assert (failed, skipped) == ([], ["acme"])
    assert all("very long name" not in call.kwargs["text_input"] for call in llm.call_args_list)
    kwargs = llm.call_args.kwargs
    tokens = WordTokenizer().count_tokens(kwargs["system_prompt"] + kwargs["text_input"])
    tokens += WordTokenizer().count_tokens(
        review_entities.json.dumps(schema.ReviewResponse.model_json_schema(), sort_keys=True)
    )
    assert tokens <= 1_000


# Every review call pays words(prompt) + words(schema) before its first section, so
# growth here leaves less room for sections; nothing is dropped. The budget test above
# gives a call 1_000 words, and alice's and bob's headers take 10 together; past 990
# they stop sharing a call and past 995 that test loses them.
PROMPT_OVERHEAD_WORDS = 1_000 - 10


def test_system_prompt_leaves_room_for_facts():
    overhead = WordTokenizer().count_tokens(
        render_prompt("review_conflicts.txt", {})
        + " "
        + review_entities.json.dumps(schema.ReviewResponse.model_json_schema(), sort_keys=True)
    )
    assert overhead <= PROMPT_OVERHEAD_WORDS


def test_a_call_is_at_most_half_the_token_ceiling(monkeypatch):
    """A model whose token ceiling is under twice REVIEW_INPUT_TOKENS gets half its ceiling."""

    def budget(ceiling):
        monkeypatch.setattr(review_entities, "get_llm_token_ceiling", lambda: ceiling)
        return review_entities._compute_user_prompt_token_budget("prompt", WordTokenizer())

    assert budget(16_384) - budget(8_192) == review_entities.REVIEW_INPUT_TOKENS - 4_096


@pytest.mark.asyncio
async def test_pipeline_tasks_preserve_reader_object_and_skip_subjects_without_a_group(
    monkeypatch,
):
    entity = {"id": "acme", "name": "Acme", "type": "Entity"}
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            side_effect=lambda filters: ([("acme", entity)] if "id" in filters[0] else [], [])
        ),
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
    llm = AsyncMock()
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    written = []

    async def write_batch(data):
        assert isinstance(data, list) and len(data) == 1
        assert isinstance(data[0], (review_entities.ConflictsToWrite, ConflictsToFinalize))
        written.append(data[0])

    reader = Task(read_facts.read_entity_facts, entity_ids=["acme"], needs_llm=False)
    failed = []
    reviewer = Task(
        review_entities.review_entities, failed_entity_ids=failed, skipped_entity_ids=[]
    )
    writer = Task(write_batch, batch_size=1, needs_llm=False)
    ctx = PipelineContext(dataset=SimpleNamespace(id=uuid4()))
    async for scope in reader.execute([[{}]], {"ctx": ctx}):
        assert isinstance(scope, ReviewScope)
        async for batches in reviewer.execute([scope], {}, writer.task_config["batch_size"]):
            async for _ in writer.execute([batches], {}):
                pass
    # Acme has no value group and owns no conflict: no call is made, and it is not counted as failed.
    assert [type(item) for item in written] == [ConflictsToFinalize]
    assert failed == []
    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_calls_hold_at_most_subjects_per_call(monkeypatch, every_entity_a_subject):
    entities = {str(i): {"id": str(i), "name": f"Entity {i}"} for i in range(60)}
    data = ReviewScope(dataset_id="dataset", entities=entities, facts={})
    llm = AsyncMock(side_effect=reviewed_only)
    monkeypatch.setattr(review_entities.LLMGateway, "acreate_structured_output", llm)
    _, failed = await review_all(data)
    assert [
        len(re.findall(r"^Subject \[", call.kwargs["text_input"], re.MULTILINE))
        for call in llm.call_args_list
    ] == [25, 25, 10]
    assert failed == []
