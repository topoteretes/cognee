"""authorized_search: dataset routing narrows an unscoped search to the routed datasets."""

import importlib
import types
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.modules.retrieval.code_retriever import CodeSeedNotFoundError
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.search.dataset_routing import DatasetRoutingConfig, DatasetRoutingDecision
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _dataset(name):
    return types.SimpleNamespace(id=uuid5(NAMESPACE_OID, name), name=name, tenant_id=None)


def _payload(dataset, hits=(), **fields):
    """A dataset's result: ``hits`` is what retrieval found (its context and objects)."""
    return SearchResultPayload(
        result_object=list(hits),
        search_type=SearchType.CHUNKS,
        question="q",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        **{"context": list(hits), **fields},
    )


@pytest.fixture
def lane(monkeypatch):
    """Stub the permission lookup, the fan-out and the router; returns spies and knobs."""
    search_mod = importlib.import_module("cognee.modules.search.methods.search")
    routing_pkg = importlib.import_module("cognee.modules.search.dataset_routing")

    state = types.SimpleNamespace(
        readable=[_dataset("a"), _dataset("b")],
        fan_out_calls=[],  # dataset lists handed to search_in_datasets_context, in order
        widening_flags=[],  # the ``widening`` flag of each fan-out call, in order
        route_calls=[],
        decision=None,
        responses=[[]],  # one per fan-out call, in order; the last one repeats
        access_control=True,
        config=DatasetRoutingConfig(_env_file=None),
    )

    async def fake_readable(datasets, permission_type, user):
        assert permission_type == "read"
        if datasets is None:
            return state.readable
        return [dataset for dataset in state.readable if dataset.id in datasets]

    async def fake_fan_out(**kwargs):
        state.fan_out_calls.append([dataset.id for dataset in kwargs["search_datasets"]])
        state.widening_flags.append(kwargs.get("widening", False))
        response = state.responses[min(len(state.fan_out_calls), len(state.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    async def fake_route(query_text, dataset_ids, top_k, distance_cutoff, tie_margin):
        state.route_calls.append(
            (query_text, list(dataset_ids), top_k, distance_cutoff, tie_margin)
        )
        return state.decision

    monkeypatch.setattr(search_mod, "get_authorized_existing_datasets", fake_readable)
    monkeypatch.setattr(search_mod, "search_in_datasets_context", fake_fan_out)
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: state.access_control)
    monkeypatch.setattr(routing_pkg, "route_datasets", fake_route)
    monkeypatch.setattr(routing_pkg, "get_dataset_routing_config", lambda: state.config)
    state.search_mod = search_mod
    return state


async def _search(lane, **kwargs):
    return await lane.search_mod.authorized_search(
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=types.SimpleNamespace(id=uuid4(), tenant_id=None),
        **kwargs,
    )


def _narrowed(lane):
    a, b = (dataset.id for dataset in lane.readable)
    return DatasetRoutingDecision(candidates=(a, b), routed=(b,), unindexed=(), scores={b: 0.1})


@pytest.mark.asyncio
async def test_unscoped_search_covers_only_the_routed_dataset(lane):
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [[_payload(b, ["hit"])]]

    out = await _search(lane)

    assert lane.route_calls == [("q", [a.id, b.id], 1, 0.8, 0.001)]
    assert lane.fan_out_calls == [[b.id]]
    assert [payload.dataset_id for payload in out] == [b.id]


@pytest.mark.asyncio
async def test_top_k_cutoff_and_margin_come_from_config(lane):
    lane.decision = _narrowed(lane)
    lane.config = DatasetRoutingConfig(
        _env_file=None, top_k=2, distance_cutoff=0.6, tie_margin=0.01
    )

    await _search(lane)

    assert lane.route_calls[0][2:] == (2, 0.6, 0.01)


@pytest.mark.asyncio
async def test_a_query_beyond_the_cutoff_everywhere_searches_every_dataset_once(lane):
    """Routing scored both datasets too far away: one fan-out over all, no widening pass."""
    a, b = lane.readable
    lane.decision = DatasetRoutingDecision(
        candidates=(a.id, b.id), routed=(), unindexed=(), scores={a.id: 0.9, b.id: 0.95}
    )
    lane.responses = [[]]

    await _search(lane)

    assert lane.fan_out_calls == [[a.id, b.id]]
    assert lane.widening_flags == [False]


@pytest.mark.asyncio
async def test_an_explicit_dataset_scope_is_never_routed(lane):
    a, _ = lane.readable

    await _search(lane, dataset_ids=[a.id])

    assert lane.route_calls == []
    assert lane.fan_out_calls == [[a.id]]


@pytest.mark.asyncio
async def test_no_routing_without_backend_access_control(lane):
    lane.access_control = False

    await _search(lane)

    assert lane.route_calls == []
    assert lane.fan_out_calls == [[dataset.id for dataset in lane.readable]]


@pytest.mark.asyncio
async def test_a_single_readable_dataset_is_not_scored(lane):
    lane.readable = lane.readable[:1]

    await _search(lane)

    assert lane.route_calls == []
    assert lane.fan_out_calls == [[lane.readable[0].id]]


@pytest.mark.asyncio
async def test_an_unnarrowed_decision_keeps_every_dataset(lane):
    a, b = (dataset.id for dataset in lane.readable)
    lane.decision = DatasetRoutingDecision(candidates=(a, b), routed=(), unindexed=(a, b))

    await _search(lane)

    assert lane.fan_out_calls == [[a, b]]


@pytest.mark.asyncio
async def test_routed_datasets_come_best_first_then_unindexed(lane):
    lane.readable = [_dataset("a"), _dataset("b"), _dataset("c")]
    a, b, c = (dataset.id for dataset in lane.readable)
    lane.decision = DatasetRoutingDecision(
        candidates=(a, b, c), routed=(c,), unindexed=(a,), scores={c: 0.1, b: 0.5}
    )
    lane.responses = [[_payload(lane.readable[2], ["hit"])]]

    await _search(lane)

    assert lane.fan_out_calls == [[c, a]]


@pytest.mark.asyncio
async def test_a_routed_search_that_finds_nothing_widens_to_the_remaining_datasets(lane):
    """The routed dataset is not searched twice: the datasets routing skipped are
    searched as the rest of the same search and the results merge in readable order."""
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [[_payload(b)], [_payload(a, ["found elsewhere"])]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id]]
    assert lane.widening_flags == [False, True]
    assert [payload.dataset_id for payload in out] == [a.id, b.id]


@pytest.mark.asyncio
async def test_the_widened_results_keep_readable_order(lane):
    lane.readable = [_dataset("a"), _dataset("b"), _dataset("c")]
    a, b, c = lane.readable
    lane.decision = DatasetRoutingDecision(
        candidates=(a.id, b.id, c.id), routed=(b.id,), unindexed=(), scores={b.id: 0.1}
    )
    lane.responses = [[_payload(b)], [_payload(a), _payload(c, ["hit"])]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id, c.id]]
    assert [payload.dataset_id for payload in out] == [a.id, b.id, c.id]


def test_a_widening_pass_never_raises_for_datasets_without_memory(lane):
    """The routed datasets had memory and retrieved nothing; the remaining ones
    lacking it is reported per dataset, as a full fan-out would, not as a 404."""
    a, b = lane.readable
    outcomes = [
        lane.search_mod.DatasetNoDataError(a, NoDataError("empty"), graph_is_empty=True),
        lane.search_mod.DatasetNoDataError(b, NoDataError("empty"), graph_is_empty=True),
    ]

    with pytest.raises(NoDataError):
        lane.search_mod._collect_dataset_results(
            list(outcomes), query_type=SearchType.CHUNKS, query_text="q", only_context=False
        )
    out = lane.search_mod._collect_dataset_results(
        list(outcomes),
        query_type=SearchType.CHUNKS,
        query_text="q",
        only_context=False,
        widening=True,
    )

    assert [payload.dataset_id for payload in out] == [a.id, b.id]
    assert all(payload.error for payload in out)


@pytest.mark.asyncio
async def test_an_empty_only_context_answer_counts_as_nothing_found(lane):
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [[_payload(b, only_context=True, context="")], [_payload(a, ["x"])]]

    await _search(lane, only_context=True)

    assert lane.fan_out_calls == [[b.id], [a.id]]


@pytest.mark.asyncio
async def test_a_completion_over_an_empty_context_counts_as_nothing_found(lane):
    """The LLM answers even when retrieval found nothing ("the context does not
    mention..."); the retrieval context, not the answer, decides the widening."""
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [
        [_payload(b, completion="The context does not mention this.")],
        [_payload(a, ["hit"], completion="It is X.")],
    ]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id]]
    assert [payload.dataset_id for payload in out] == [a.id, b.id]


@pytest.mark.asyncio
async def test_a_session_turn_acknowledgement_does_not_widen(lane):
    """A turn that only set guidance ran no retrieval: no context, no error, an
    acknowledgement as the completion. Widening would replay it on every dataset."""
    lane.decision = _narrowed(lane)
    _, b = lane.readable
    lane.responses = [[_payload(b, context=None, completion=["Noted."])]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id]]
    assert [payload.dataset_id for payload in out] == [b.id]


@pytest.mark.asyncio
async def test_a_dataset_that_could_not_be_searched_counts_as_nothing_found(lane):
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [
        [_payload(b, context=None, completion={"seed_not_found": True}, error="no seed")],
        [_payload(a, ["hit"])],
    ]

    await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id]]


@pytest.mark.asyncio
async def test_routed_datasets_without_memory_widen_to_every_dataset(lane):
    """Every routed dataset failed before retrieving, so nothing is kept and the
    whole set is searched as one (the fan-out's 404 rule needs the full set)."""
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [NoDataError("nothing here"), [_payload(a, ["hit"])]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id, b.id]]
    assert [payload.dataset_id for payload in out] == [a.id]


@pytest.mark.asyncio
async def test_an_empty_row_result_without_a_context_counts_as_nothing_found(lane):
    """Cypher and natural-language retrievers report rows and no context: with no
    rows in the routed dataset the search widens to the one that has them."""
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [[_payload(b, context=None)], [_payload(a, ["row"], context=None)]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id]]
    assert [payload.dataset_id for payload in out] == [a.id, b.id]


@pytest.mark.asyncio
async def test_a_code_seed_the_routed_dataset_cannot_resolve_widens(lane):
    """The fan-out softens a seed miss per dataset only when it searches several;
    routed to one, the miss escapes and must widen rather than fail the call."""
    lane.decision = _narrowed(lane)
    a, b = lane.readable
    lane.responses = [CodeSeedNotFoundError("no such symbol"), [_payload(a, ["fact"])]]

    out = await _search(lane)

    assert lane.fan_out_calls == [[b.id], [a.id, b.id]]
    assert [payload.dataset_id for payload in out] == [a.id]


@pytest.mark.asyncio
async def test_no_data_anywhere_still_raises(lane):
    lane.decision = _narrowed(lane)
    lane.responses = [NoDataError("nothing here")]

    with pytest.raises(NoDataError):
        await _search(lane)

    assert len(lane.fan_out_calls) == 2


def test_routing_defaults():
    config = DatasetRoutingConfig(_env_file=None)
    assert not hasattr(config, "enabled"), "routing has no off switch: it is always on"
    assert config.top_k == 1
    assert config.distance_cutoff == 0.8
    assert config.tie_margin == 0.001
    with pytest.raises(ValueError):
        DatasetRoutingConfig(_env_file=None, top_k=0)
    with pytest.raises(ValueError):
        DatasetRoutingConfig(_env_file=None, distance_cutoff=2.5)
    with pytest.raises(ValueError):
        DatasetRoutingConfig(_env_file=None, tie_margin=-0.1)


def test_payload_entries_follow_the_normalization_rule():
    dataset = _dataset("a")
    assert _payload(dataset).entries() == []
    assert _payload(dataset, ["x"]).entries() == ["x"]
    assert _payload(dataset, ["x"], completion="answer").entries() == ["answer"]
    assert _payload(dataset, only_context=True, context="").entries() == []
    assert _payload(dataset, only_context=True, context=["c"]).entries() == ["c"]
    assert _payload(dataset, only_context=True, user_prompt="p", context=["c"]).entries() == ["p"]


def test_retrieved_nothing_reads_the_context_not_the_completion():
    dataset = _dataset("a")
    # Empty context: a miss, whatever the completion says.
    assert _payload(dataset).retrieved_nothing() is True
    assert _payload(dataset, context="").retrieved_nothing() is True
    assert _payload(dataset, completion="I cannot tell.").retrieved_nothing() is True
    assert _payload(dataset, completion="").retrieved_nothing() is True
    # Any context: a hit, even with no completion.
    assert _payload(dataset, ["x"]).retrieved_nothing() is False
    assert _payload(dataset, context="passage").retrieved_nothing() is False
    # No context and a completion: nothing was retrieved (an acknowledgement), not a miss.
    assert _payload(dataset, context=None, completion=["Noted."]).retrieved_nothing() is False
    # No context and no completion: a row-reporting retriever; its rows decide.
    assert _payload(dataset, context=None).retrieved_nothing() is True
    assert _payload(dataset, ["row"], context=None).retrieved_nothing() is False
    # An error always is a miss.
    assert _payload(dataset, context=None, error="no memory").retrieved_nothing() is True
