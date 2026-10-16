"""Typed memory entries authorize and bind their session to the target dataset.

Runs against a real relational database and the sqlite session cache, because
the contract is about real ACLs and the session row that permission filtering
reads.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.memory import FeedbackEntry, QAEntry, SkillRunEntry
from cognee.modules.users.exceptions import PermissionDeniedError


async def _reset_engines():
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    cache_factory = importlib.import_module(
        "cognee.infrastructure.databases.cache.get_cache_engine"
    )
    for cached in (
        _create_graph_engine,
        _create_vector_engine,
        create_relational_engine,
        get_cache_config,
        cache_factory.create_cache_engine,
    ):
        cached.cache_clear()


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.modules.users.methods import create_user, get_default_user
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    monkeypatch.setenv("CACHING", "true")
    monkeypatch.setenv("CACHE_BACKEND", "sqlite")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    await _reset_engines()
    await setup()

    owner = await get_default_user()
    member = await create_user(f"member-{uuid4()}@example.com", "member-password")
    outsider = await create_user(f"outsider-{uuid4()}@example.com", "outsider-password")
    _, [shared] = await resolve_authorized_user_datasets("shared", owner)
    await give_permission_on_dataset(member, shared.id, "read")
    await give_permission_on_dataset(member, shared.id, "write")

    yield SimpleNamespace(owner=owner, member=member, outsider=outsider, shared=shared)
    await _reset_engines()


async def _cached_questions(user, session_id):
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    entries = await get_session_manager().get_session(user_id=str(user.id), session_id=session_id)
    return [entry.question for entry in entries]


async def _bound_dataset_id(user, session_id):
    from cognee.modules.session_lifecycle.metrics import get_session_row

    row = await get_session_row(session_id=session_id, user_id=user.id)
    return row.dataset_id if row else None


async def _last_activity_at(user, session_id):
    from cognee.modules.session_lifecycle.metrics import get_session_row

    row = await get_session_row(session_id=session_id, user_id=user.id)
    return row.last_activity_at


@pytest.mark.asyncio
async def test_granted_user_writes_to_shared_dataset_by_id(env):
    result = await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_id=env.shared.id,
        session_id="member-session",
        user=env.member,
    )

    assert result.status == "session_stored"
    assert result.dataset_id == str(env.shared.id)
    assert result.dataset_name == "shared"
    assert await _bound_dataset_id(env.member, "member-session") == env.shared.id
    assert await _cached_questions(env.member, "member-session") == ["q1"]


async def _owned_dataset_names(user):
    from cognee.modules.data.methods.get_datasets import get_datasets

    return sorted(dataset.name for dataset in await get_datasets(user.id))


@pytest.mark.asyncio
async def test_user_without_grant_caches_nothing(env):
    with pytest.raises(PermissionDeniedError):
        await cognee.remember(
            QAEntry(question="q1", answer="a1"),
            dataset_id=env.shared.id,
            session_id="outsider-session",
            user=env.outsider,
        )

    assert await _cached_questions(env.outsider, "outsider-session") == []


@pytest.mark.asyncio
async def test_session_bound_to_one_dataset_rejects_another_id(env):
    await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        session_id="owner-session",
        user=env.owner,
    )
    names_before = await _owned_dataset_names(env.owner)

    with pytest.raises(ValueError, match="is bound to dataset"):
        await cognee.remember(
            QAEntry(question="q2", answer="a2"),
            dataset_id=uuid4(),
            session_id="owner-session",
            user=env.owner,
        )

    assert await _bound_dataset_id(env.owner, "owner-session") == env.shared.id
    assert await _cached_questions(env.owner, "owner-session") == ["q1"]
    # The rejection happens before any dataset resolution, so nothing is created.
    assert await _owned_dataset_names(env.owner) == names_before


@pytest.mark.asyncio
async def test_bound_session_keeps_its_dataset_under_the_default_name(env):
    # "main_dataset" is remember()'s default, so it cannot be told apart from no
    # argument: a bound session targets its dataset and creates nothing.
    await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        session_id="owner-session",
        user=env.owner,
    )
    names_before = await _owned_dataset_names(env.owner)

    result = await cognee.remember(
        QAEntry(question="q2", answer="a2"),
        dataset_name="main_dataset",
        session_id="owner-session",
        user=env.owner,
    )

    assert result.dataset_id == str(env.shared.id)
    assert await _cached_questions(env.owner, "owner-session") == ["q1", "q2"]
    assert await _owned_dataset_names(env.owner) == names_before


@pytest.mark.asyncio
async def test_bound_session_rejects_another_explicit_name(env):
    # Any name but the default was typed by the caller, so a mismatch is an error
    # and, like the id mismatch, creates nothing.
    await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        session_id="owner-session",
        user=env.owner,
    )
    names_before = await _owned_dataset_names(env.owner)

    with pytest.raises(ValueError, match="is bound to dataset 'shared', not 'other'"):
        await cognee.remember(
            QAEntry(question="q2", answer="a2"),
            dataset_name="other",
            session_id="owner-session",
            user=env.owner,
        )

    assert await _cached_questions(env.owner, "owner-session") == ["q1"]
    assert await _owned_dataset_names(env.owner) == names_before


@pytest.mark.asyncio
async def test_session_bound_by_recall_accepts_feedback_without_dataset(env):
    # recall(datasets=["shared"], session_id=...) binds the session through
    # the session manager's activity record; rating the answer afterwards
    # passes no dataset and must land in the bound one.
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

    await ensure_and_touch_session(
        session_id="recall-session", user_id=env.owner.id, dataset_id=env.shared.id
    )
    names_before = await _owned_dataset_names(env.owner)

    stored = await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        session_id="recall-session",
        user=env.owner,
    )
    touched_at = await _last_activity_at(env.owner, "recall-session")
    await asyncio.sleep(0.01)
    rated = await cognee.remember(
        FeedbackEntry(qa_id=stored.entry_id, feedback_text="good", feedback_score=5),
        session_id="recall-session",
        user=env.owner,
    )

    assert stored.dataset_id == str(env.shared.id)
    assert rated.status == "session_stored"
    assert rated.dataset_id == str(env.shared.id)
    # Feedback counts as session activity, like every other typed entry.
    assert await _last_activity_at(env.owner, "recall-session") > touched_at
    assert "main_dataset" not in await _owned_dataset_names(env.owner)
    assert await _owned_dataset_names(env.owner) == names_before


@pytest.mark.asyncio
async def test_bound_session_without_grant_is_refused(env):
    # The owner revokes nothing here; the outsider simply has a session that
    # was bound to a dataset they can read but not write.
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    await give_permission_on_dataset(env.outsider, env.shared.id, "read")
    await ensure_and_touch_session(
        session_id="outsider-session", user_id=env.outsider.id, dataset_id=env.shared.id
    )

    with pytest.raises(PermissionDeniedError):
        await cognee.remember(
            QAEntry(question="q1", answer="a1"),
            session_id="outsider-session",
            user=env.outsider,
        )

    assert await _cached_questions(env.outsider, "outsider-session") == []


@pytest.mark.asyncio
async def test_skill_run_in_bound_session_targets_its_dataset(env, monkeypatch):
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

    skill_runs = importlib.import_module("cognee.modules.tools.skill_runs")
    run = SimpleNamespace(
        run_id="run-1", selected_skill_id="s", selected_skill_name="s", success_score=1.0
    )
    record_run = AsyncMock(return_value=(run, env.shared))
    monkeypatch.setattr(skill_runs, "remember_skill_run_entry", record_run)
    await ensure_and_touch_session(
        session_id="skill-session", user_id=env.owner.id, dataset_id=env.shared.id
    )

    result = await cognee.remember(
        SkillRunEntry(run_id="run-1", selected_skill_id="s", task_text="t"),
        session_id="skill-session",
        user=env.owner,
    )

    assert record_run.await_args.kwargs["dataset_id"] == env.shared.id
    assert result.dataset_id == str(env.shared.id)
    assert "main_dataset" not in await _owned_dataset_names(env.owner)


@pytest.mark.asyncio
async def test_bound_session_accepts_its_dataset_id_as_a_string(env):
    # Chaining a RememberResult back in passes its dataset_id, which is a str.
    first = await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        session_id="owner-session",
        user=env.owner,
    )

    second = await cognee.remember(
        QAEntry(question="q2", answer="a2"),
        dataset_id=first.dataset_id,
        session_id="owner-session",
        user=env.owner,
    )

    assert second.dataset_id == str(env.shared.id)
    assert await _cached_questions(env.owner, "owner-session") == ["q1", "q2"]


@pytest.mark.asyncio
async def test_dataset_id_that_is_not_a_uuid_is_rejected(env):
    with pytest.raises(ValueError, match="dataset_id must be a dataset UUID"):
        await cognee.remember(
            QAEntry(question="q1", answer="a1"),
            dataset_id="shared",
            session_id="owner-session",
            user=env.owner,
        )

    assert await _cached_questions(env.owner, "owner-session") == []


@pytest.mark.asyncio
async def test_bound_session_skips_the_binding_reread(env, monkeypatch):
    metrics = importlib.import_module("cognee.modules.session_lifecycle.metrics")
    read_row = AsyncMock(wraps=metrics.get_session_row)
    monkeypatch.setattr(metrics, "get_session_row", read_row)

    for question in ("q1", "q2"):
        await cognee.remember(
            QAEntry(question=question, answer="a"),
            dataset_name="shared",
            session_id="owner-session",
            user=env.owner,
        )
        if question == "q1":
            # Binding a fresh session re-reads the row to catch a concurrent binder.
            assert read_row.await_count == 2

    assert read_row.await_count == 3
    assert await _cached_questions(env.owner, "owner-session") == ["q1", "q2"]


@pytest.mark.asyncio
async def test_session_bound_to_a_deleted_dataset_is_refused_like_a_forbidden_one(env):
    """Deleted and revoked must look the same, or a 403 would tell a revoked
    caller by elimination that the dataset still exists."""
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

    await ensure_and_touch_session(
        session_id="stale-session", user_id=env.owner.id, dataset_id=uuid4()
    )

    with pytest.raises(PermissionDeniedError):
        await cognee.remember(
            QAEntry(question="q1", answer="a1"),
            session_id="stale-session",
            user=env.owner,
        )

    assert await _cached_questions(env.owner, "stale-session") == []


@pytest.mark.asyncio
async def test_unknown_explicit_dataset_id_is_refused_like_a_forbidden_one(env):
    # Callers must not learn whether a dataset id they cannot write exists.
    with pytest.raises(PermissionDeniedError):
        await cognee.remember(
            QAEntry(question="q1", answer="a1"),
            dataset_id=uuid4(),
            session_id="fresh-session",
            user=env.owner,
        )


@pytest.mark.asyncio
async def test_bound_session_lets_an_explicit_id_override_the_name(env):
    # dataset_id takes precedence over dataset_name, bound session or not.
    await cognee.remember(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        session_id="owner-session",
        user=env.owner,
    )
    names_before = await _owned_dataset_names(env.owner)

    result = await cognee.remember(
        QAEntry(question="q2", answer="a2"),
        dataset_name="other",
        dataset_id=env.shared.id,
        session_id="owner-session",
        user=env.owner,
    )

    assert result.dataset_id == str(env.shared.id)
    assert await _owned_dataset_names(env.owner) == names_before


@pytest.mark.asyncio
async def test_remote_client_receives_dataset_id(monkeypatch):
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    state = importlib.import_module("cognee.api.v1.serve.state")
    client = SimpleNamespace(remember_entry=AsyncMock(return_value={"status": "session_stored"}))
    monkeypatch.setattr(state, "get_remote_client", lambda: client)
    dataset_id = uuid4()

    await remember_module._remember_entry(
        QAEntry(question="q1", answer="a1"),
        dataset_name="shared",
        dataset_id=dataset_id,
        session_id="remote-session",
        user=None,
    )

    assert client.remember_entry.await_args.kwargs["dataset_id"] == dataset_id
