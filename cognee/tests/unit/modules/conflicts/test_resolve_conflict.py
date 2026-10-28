from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.locks.dataset_lock import dataset_lock, held_datasets
from cognee.modules.conflicts import methods
from cognee.modules.graph.utils.fact_conflicts import is_rejected_fact
from cognee.modules.retrieval.utils.resolved_context import read_user_resolutions
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.tasks.memify.review_conflicts import write_conflicts
from cognee.tasks.memify.review_conflicts.models import ConflictsToFinalize, ConflictsToWrite
from cognee.tests.unit.tasks.memify.review_conflicts.test_write_conflicts import store


@pytest.fixture
def lane(store, monkeypatch):  # noqa: F811 -- imported pytest fixture
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    dataset = SimpleNamespace(id=UUID(store.scope.dataset_id), owner_id=uuid4())
    stored = store.nodes[store.cid]
    store.scope.entities[stored["about_id"]] = {"id": stored["about_id"], "name": "Company"}
    for fact in store.scope.facts.values():
        fact["properties"]["edge_text"] = (
            f"Company has CEO {'Alice' if fact['target'] == store.alice else 'Bob'}"
        )
    permissions = AsyncMock(return_value=[dataset])
    monkeypatch.setattr(methods, "get_authorized_existing_datasets", permissions)
    monkeypatch.setattr(methods, "get_graph_engine", AsyncMock(return_value=store.graph))
    store.graph.get_filtered_graph_data = AsyncMock(
        side_effect=lambda *a, **kw: (list(store.nodes.items()), [])
    )
    monkeypatch.setattr(methods, "read_entity_facts", AsyncMock(return_value=store.scope))
    invalidation = AsyncMock(return_value={})
    monkeypatch.setattr(methods, "invalidate_sessions_for_deleted_data", invalidation)
    contexts = []

    @asynccontextmanager
    async def context(*args):
        contexts.append(args)
        yield

    @asynccontextmanager
    async def operation(*args):
        yield SimpleNamespace(set_user=lambda u: None, set_dataset=lambda d: None)

    monkeypatch.setattr(methods, "set_database_global_context_variables", context)
    monkeypatch.setattr(methods, "record_operation", operation)
    return SimpleNamespace(
        store=store,
        user=user,
        dataset=dataset,
        permissions=permissions,
        invalidation=invalidation,
        contexts=contexts,
    )


async def resolve(lane, fact=None, **kwargs):
    return await methods.resolve_conflict(
        UUID(lane.store.cid),
        dataset_id=lane.dataset.id,
        selected_fact_id=UUID(fact or next(iter(lane.store.scope.facts))),
        user=lane.user,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_choose_fact_persists_decision_marks_and_invalidates_cached_answers(lane):
    result = await resolve(lane, reason="Verified with the company")
    assert result.status == "resolved"
    assert result.resolution_source == "user"
    assert result.selected_fact == "Company has CEO Alice"
    assert result.resolved_by == lane.user.id
    assert result.resolution_reason == "Verified with the company"
    assert not lane.store.nodes[lane.store.cid]["write_pending"]
    assert [fact.status for fact in result.facts if fact.id == result.selected_fact_id] == [
        "current"
    ]
    assert all(
        fact.status == "superseded" for fact in result.facts if fact.id != result.selected_fact_id
    )
    lane.permissions.assert_awaited_once_with([lane.dataset.id], "write", lane.user)
    assert lane.contexts == [(lane.dataset.id, lane.dataset.owner_id)] * 2
    assert lane.invalidation.call_args.kwargs["deleted_edge_ids"] == set(lane.store.scope.facts)
    # A resolution affects this conflict only, preserving other conflict marks.
    for fact in lane.store.scope.facts.values():
        assert {"conflict_id": "other", "status": "conflicting"} in fact["properties"][
            "conflict_marks"
        ]


@pytest.mark.asyncio
async def test_repeat_is_idempotent_and_an_explicit_new_choice_revises_it(lane):
    first = await resolve(lane)
    calls = len(lane.store.calls)
    repeated = await resolve(lane)
    assert repeated.resolved_at == first.resolved_at
    assert len(lane.store.calls) == calls
    other = next(fid for fid in lane.store.scope.facts if fid != str(first.selected_fact_id))
    revised = await resolve(lane, other)
    assert revised.selected_fact == "Company has CEO Bob"
    assert str(revised.selected_fact_id) == other
    assert lane.store.nodes[lane.store.cid]["rejected_fact_ids"] == [str(first.selected_fact_id)]


@pytest.mark.asyncio
async def test_rejects_a_fact_from_another_conflict_before_writing(lane):
    with pytest.raises(CogneeValidationError) as error:
        await resolve(lane, str(uuid4()))
    assert error.value.status_code == 422
    assert not lane.store.calls
    lane.invalidation.assert_not_awaited()


@pytest.mark.asyncio
async def test_permission_check_precedes_graph_read(lane):
    lane.permissions.return_value = []
    with pytest.raises(PermissionDeniedError):
        await resolve(lane)
    lane.store.graph.get_filtered_graph_data.assert_not_awaited()
    assert not lane.store.calls


@pytest.mark.asyncio
async def test_cannot_resolve_conflict_from_another_dataset(lane):
    lane.store.nodes[lane.store.cid]["dataset_id"] = str(uuid4())
    with pytest.raises(CogneeValidationError) as error:
        await resolve(lane)
    assert error.value.status_code == 404
    assert not lane.store.calls


@pytest.mark.asyncio
async def test_list_includes_selectable_facts_and_checks_read_permission(lane):
    results = await methods.list_conflicts(lane.dataset.id, user=lane.user)
    assert len(results) == 1 and len(results[0].facts) == 2
    lane.permissions.assert_awaited_once_with([lane.dataset.id], "read", lane.user)
    assert await methods.list_conflicts(lane.dataset.id, status="unresolved", user=lane.user) == []


@pytest.mark.asyncio
async def test_automatic_review_cannot_reverse_or_drop_a_user_decision(lane):
    selected = await resolve(lane)
    # The fixture's review would otherwise choose Bob and overwrite the explanation.
    await write_conflicts.write_conflicts([lane.store.batch], lane.store.ctx)
    await write_conflicts.write_conflicts(
        [ConflictsToFinalize(accepted_conflicts=lane.store.batch.accepted_conflicts)],
        lane.store.ctx,
    )
    saved = lane.store.nodes[lane.store.cid]
    assert saved["selected_fact_id"] == str(selected.selected_fact_id)
    assert saved["selected_fact"] == "Company has CEO Alice"
    drop = ConflictsToWrite(lane.store.scope, dropped_conflict_ids=[lane.store.cid])
    await write_conflicts.write_conflicts([drop], lane.store.ctx)
    assert drop.dropped_conflict_ids == []
    assert drop.accepted_conflicts[0].conflict.selected_fact == "Company has CEO Alice"


@pytest.mark.asyncio
async def test_failed_write_is_pending_and_retry_completes_the_choice(lane):
    lane.store.fail.at = "links"
    with pytest.raises(RuntimeError, match="links"):
        await resolve(lane)
    assert lane.store.nodes[lane.store.cid]["write_pending"]
    lane.store.fail.at = None
    retried = await resolve(lane)
    assert not lane.store.nodes[lane.store.cid]["write_pending"]
    assert not retried.write_pending
    assert retried.selected_fact == "Company has CEO Alice"


@pytest.mark.asyncio
async def test_cache_cleanup_failure_does_not_finalize_resolution(lane):
    lane.invalidation.side_effect = RuntimeError("cache unavailable")
    with pytest.raises(RuntimeError, match="cache unavailable"):
        await resolve(lane)
    assert lane.store.nodes[lane.store.cid]["write_pending"]
    lane.invalidation.side_effect = None
    result = await resolve(lane)
    assert not result.write_pending
    assert lane.invalidation.call_args.kwargs["strict"] is True


@pytest.mark.asyncio
async def test_listing_reads_only_conflict_subjects_without_the_write_lock(lane, monkeypatch):
    @asynccontextmanager
    async def forbidden_lock(*args):
        raise AssertionError("Listing must not take the dataset write lock")
        yield

    monkeypatch.setattr(methods, "dataset_lock", forbidden_lock)
    results = await methods.list_conflicts(lane.dataset.id, user=lane.user)
    assert len(results) == 1
    assert methods.read_entity_facts.await_args.kwargs["entity_ids"] == [
        lane.store.nodes[lane.store.cid]["about_id"]
    ]
    methods.read_entity_facts.reset_mock()
    lane.store.nodes.clear()
    assert await methods.list_conflicts(lane.dataset.id, user=lane.user) == []
    methods.read_entity_facts.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("review_drops", [False, True])
async def test_review_retires_a_user_choice_when_its_selected_fact_was_forgotten(
    lane, review_drops
):
    selected = await resolve(lane)
    del lane.store.scope.facts[str(selected.selected_fact_id)]
    batch = lane.store.batch
    if review_drops:
        batch = ConflictsToWrite(lane.store.scope, dropped_conflict_ids=[lane.store.cid])
    else:
        incoming = batch.accepted_conflicts[0]
        incoming.fact_statuses = {fid: "current" for fid in lane.store.scope.facts}
        incoming.conflict.text = "Company has CEO Bob"
        # Even marks on facts left out of the review must lose the obsolete user choice.
        batch.shown_fact_ids.clear()
    await write_conflicts.write_conflicts([batch], lane.store.ctx)
    await write_conflicts.write_conflicts(
        [ConflictsToFinalize(batch.accepted_conflicts, batch.dropped_conflict_ids)], lane.store.ctx
    )
    assert await read_user_resolutions(lane.store.graph) == []
    for fact in lane.store.scope.facts.values():
        assert not is_rejected_fact(fact["properties"])
        assert all(not mark.get("selected_fact") for mark in fact["properties"]["conflict_marks"])
    if not review_drops:
        assert lane.store.nodes[lane.store.cid]["resolution_source"] == "review"
        assert lane.store.nodes[lane.store.cid]["selected_fact"] is None


@pytest.mark.asyncio
async def test_cache_cleanup_releases_dataset_lock_and_keeps_decision_pending(lane):
    async def clean(*args, **kwargs):
        assert lane.dataset.id not in held_datasets.get()
        assert lane.store.nodes[lane.store.cid]["write_pending"]
        async with dataset_lock(lane.dataset.id):
            pass

    lane.invalidation.side_effect = clean
    await resolve(lane)
    assert not lane.store.nodes[lane.store.cid]["write_pending"]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_change", ["delete", "revise"])
async def test_cleanup_finalization_does_not_resurrect_or_overwrite_a_newer_decision(
    lane, concurrent_change
):
    async def clean(*args, **kwargs):
        if concurrent_change == "delete":
            lane.store.nodes.pop(lane.store.cid)
        else:
            lane.store.nodes[lane.store.cid].update(
                selected_fact_id=str(uuid4()), selected_fact="A newer decision", resolved_at="later"
            )

    lane.invalidation.side_effect = clean
    await resolve(lane)
    if concurrent_change == "delete":
        assert lane.store.cid not in lane.store.nodes
    else:
        assert lane.store.nodes[lane.store.cid]["selected_fact"] == "A newer decision"
        assert lane.store.nodes[lane.store.cid]["write_pending"]


@pytest.mark.asyncio
async def test_automatic_review_cannot_finalize_incomplete_cache_cleanup(lane):
    async def review_while_cleaning(*args, **kwargs):
        await write_conflicts.write_conflicts([lane.store.batch], lane.store.ctx)
        await write_conflicts.write_conflicts(
            [ConflictsToFinalize(lane.store.batch.accepted_conflicts)], lane.store.ctx
        )
        assert lane.store.nodes[lane.store.cid]["write_pending"]
        raise RuntimeError("cache unavailable")

    lane.invalidation.side_effect = review_while_cleaning
    with pytest.raises(RuntimeError, match="cache unavailable"):
        await resolve(lane)
    assert lane.store.nodes[lane.store.cid]["cache_cleanup_pending"]
    assert lane.store.nodes[lane.store.cid]["write_pending"]
    lane.invalidation.side_effect = None
    await resolve(lane)
    assert not lane.store.nodes[lane.store.cid]["cache_cleanup_pending"]
    assert not lane.store.nodes[lane.store.cid]["write_pending"]
