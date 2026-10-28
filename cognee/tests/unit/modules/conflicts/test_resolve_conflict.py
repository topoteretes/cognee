from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.conflicts import methods
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
        side_effect=lambda *a: (list(store.nodes.items()), [])
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
    assert lane.contexts == [(lane.dataset.id, lane.dataset.owner_id)]
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
