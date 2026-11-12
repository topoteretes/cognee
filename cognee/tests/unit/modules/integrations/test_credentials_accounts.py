"""One user, several connections to one provider: which one a lookup returns."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import update

from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


async def connect(user_id, account_id, label):
    credential = await store.upsert_credential(
        provider="notion",
        provider_account_id=account_id,
        user_id=user_id,
        token_payload={"access_token": f"token-{account_id}"},
        account_label=label,
    )
    # created_at has one-second resolution on some backends, so order the rows explicitly.
    await asyncio.sleep(0.01)
    return credential


@pytest.mark.asyncio
async def test_without_an_account_the_newest_connection_answers(credential_db):
    user_id = uuid4()
    await connect(user_id, "ws-1", "First")
    await connect(user_id, "ws-2", "Second")

    newest = await store.get_active_credential_for_user(user_id, "notion")

    assert newest.provider_account_id == "ws-2"


@pytest.mark.asyncio
async def test_a_named_account_answers_even_when_it_is_not_the_newest(credential_db):
    user_id = uuid4()
    await connect(user_id, "ws-1", "First")
    await connect(user_id, "ws-2", "Second")

    older = await store.get_active_credential_for_user(user_id, "notion", "ws-1")

    assert older.provider_account_id == "ws-1" and older.account_label == "First"


@pytest.mark.asyncio
async def test_another_users_account_is_not_found_and_does_not_fall_back(credential_db):
    owner, stranger = uuid4(), uuid4()
    await connect(owner, "ws-1", "Theirs")
    await connect(stranger, "ws-9", "Mine")

    assert await store.get_active_credential_for_user(stranger, "notion", "ws-1") is None
    assert await store.get_active_credential_for_user(stranger, "notion", "nope") is None


@pytest.mark.asyncio
async def test_a_disconnected_account_is_not_found(credential_db):
    user_id = uuid4()
    await connect(user_id, "ws-1", "First")
    await connect(user_id, "ws-2", "Second")
    await store.revoke_credential_by_account("notion", "ws-1")

    assert await store.get_active_credential_for_user(user_id, "notion", "ws-1") is None
    assert [
        credential.provider_account_id
        for credential in await store.list_active_credentials_for_provider(user_id, "notion")
    ] == ["ws-2"]


@pytest.mark.asyncio
async def test_the_connections_come_back_newest_first_for_one_provider_only(credential_db):
    user_id = uuid4()
    await connect(user_id, "ws-1", "First")
    await connect(user_id, "ws-2", "Second")
    await store.upsert_credential(
        provider="linear",
        provider_account_id="org-1",
        user_id=user_id,
        token_payload={"access_token": "x"},
    )

    listed = await store.list_active_credentials_for_provider(user_id, "notion")

    assert [credential.provider_account_id for credential in listed] == ["ws-2", "ws-1"]


@pytest.mark.asyncio
async def test_the_owner_is_the_workspace_when_set_else_the_user_in_python_and_in_sql(
    credential_db,
):
    """One rule in two forms: the owner of a row, and the SQL listing a principal's rows,
    must agree, or routes save a policy under one owner that the scheduler never runs."""
    user_id, workspace_id = uuid4(), uuid4()
    personal = await connect(user_id, "personal", "Personal")
    shared = await store.upsert_credential(
        provider="notion",
        provider_account_id="shared",
        user_id=user_id,
        workspace_id=workspace_id,
        token_payload={"access_token": "token-shared"},
    )
    await connect(user_id, "gone", "Gone")
    await store.revoke_credential_by_account("notion", "gone")

    assert [store.owner_principal_id(c) for c in (personal, shared)] == [user_id, workspace_id]
    assert await store.list_active_credential_ids_for_owner(user_id) == [personal.id]
    assert await store.list_active_credential_ids_for_owner(workspace_id) == [shared.id]


async def revoked_at(account_id, when):
    """Pin when a connection ended: ``revoked_at`` is ``now()``, too close to order on."""
    async with store.get_relational_engine().get_async_session() as db:
        await db.execute(
            update(IntegrationCredential)
            .where(IntegrationCredential.provider_account_id == account_id)
            .values(revoked_at=when)
        )
        await db.commit()


@pytest.mark.asyncio
async def test_a_connection_the_provider_ended_is_reported_to_its_owner(credential_db):
    owner = uuid4()
    await connect(owner, "ws-1", "Acme")
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)

    ended = await store.get_ended_connection(owner, "notion")

    assert ended.provider_account_id == "ws-1"
    assert ended.revoked_by == "provider" and ended.revoked_at is not None
    assert await store.get_ended_connection(uuid4(), "notion") is None


@pytest.mark.asyncio
async def test_a_later_revoke_keeps_who_ended_the_connection_first(credential_db):
    owner = uuid4()
    await connect(owner, "ws-1", "Acme")
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_OWNER)
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)

    assert (await store.get_ended_connection(owner, "notion")).revoked_by == "owner"


@pytest.mark.asyncio
async def test_the_newest_ended_connection_answers_whoever_ended_it(credential_db):
    """An older loss at the provider does not outlive the owner's newer disconnect."""
    owner = uuid4()
    await connect(owner, "ws-1", "Old")
    await connect(owner, "ws-2", "New")
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)
    await store.revoke_credential_by_account("notion", "ws-2", revoked_by=store.REVOKED_BY_OWNER)
    await revoked_at("ws-1", datetime(2026, 11, 1, tzinfo=timezone.utc))
    await revoked_at("ws-2", datetime(2026, 11, 2, tzinfo=timezone.utc))

    ended = await store.get_ended_connection(owner, "notion")

    assert (ended.provider_account_id, ended.revoked_by) == ("ws-2", "owner")
    assert (await store.get_ended_connection(owner, "notion", "ws-1")).revoked_by == "provider"


@pytest.mark.asyncio
async def test_of_two_losses_at_the_provider_the_newer_answers(credential_db):
    owner = uuid4()
    await connect(owner, "ws-1", "First")
    await connect(owner, "ws-2", "Second")
    for account in ("ws-1", "ws-2"):
        await store.revoke_credential_by_account(
            "notion", account, revoked_by=store.REVOKED_BY_PROVIDER
        )
    await revoked_at("ws-1", datetime(2026, 11, 2, tzinfo=timezone.utc))
    await revoked_at("ws-2", datetime(2026, 11, 1, tzinfo=timezone.utc))

    assert (await store.get_ended_connection(owner, "notion")).provider_account_id == "ws-1"


@pytest.mark.asyncio
async def test_reconnecting_clears_who_revoked_the_connection(credential_db):
    owner = uuid4()
    await connect(owner, "ws-1", "Acme")
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)

    again = await connect(owner, "ws-1", "Acme")

    assert again.status == "active" and again.revoked_by is None
    assert await store.get_ended_connection(owner, "notion") is None


@pytest.mark.asyncio
async def test_a_rejected_token_records_the_provider_as_the_revoker(credential_db):
    owner = uuid4()
    credential = await connect(owner, "ws-1", "Acme")

    await store.revoke_credential_if_current(credential)

    assert (await store.get_ended_connection(owner, "notion")).revoked_by == "provider"


@pytest.mark.asyncio
async def test_a_revoke_that_names_no_revoker_records_it_as_unknown(credential_db):
    """Released connector packages call it with two arguments; their revoke still lands."""
    owner = uuid4()
    await connect(owner, "ws-1", "Acme")

    assert await store.revoke_credential_by_account("notion", "ws-1") is True

    ended = await store.get_ended_connection(owner, "notion")
    assert ended.provider_account_id == "ws-1" and ended.revoked_by is None


@pytest.mark.asyncio
async def test_a_new_owner_does_not_inherit_the_previous_owners_sync_history(credential_db):
    previous, new = uuid4(), uuid4()
    original = await connect(previous, "ws-1", "Acme")
    await store.record_sync_result(original, status="completed", counts={"processed": 812})
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)

    taken_over = await connect(new, "ws-1", "Acme")

    assert taken_over.user_id == new
    assert taken_over.sync_status is None
    assert taken_over.last_synced_at is None
    assert "last_sync_counts" not in (taken_over.provider_metadata or {})


@pytest.mark.asyncio
async def test_the_same_owner_reconnecting_keeps_the_sync_history(credential_db):
    owner = uuid4()
    original = await connect(owner, "ws-1", "Acme")
    await store.record_sync_result(original, status="completed", counts={"processed": 812})
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_OWNER)

    reconnected = await connect(owner, "ws-1", "Acme")

    assert reconnected.sync_status == "completed"
    assert reconnected.last_synced_at is not None
    assert reconnected.provider_metadata["last_sync_counts"]["processed"] == 812


@pytest.mark.asyncio
async def test_overlapping_metadata_writes_are_all_kept(tmp_path, monkeypatch):
    """Every writer of ``provider_metadata`` read the row, merged in Python and wrote
    the whole column back, so of writes that overlapped only the last one stayed. A SQLite
    file gives each session its own connection, as a deployment does."""
    import base64
    from types import SimpleNamespace

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'credentials.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        store, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    try:
        await connect(uuid4(), "ws-1", "Acme")

        await asyncio.gather(
            *(store.update_provider_metadata("notion", "ws-1", {f"key_{n}": n}) for n in range(8)),
            *(
                store.change_provider_metadata(
                    "notion",
                    "ws-1",
                    lambda current, n=n: {"marks": [*current.get("marks", []), n]},
                )
                for n in range(8)
            ),
        )

        stored = (await store.get_credential_by_account("notion", "ws-1")).provider_metadata
        assert {f"key_{n}" for n in range(8)} <= stored.keys()
        assert sorted(stored["marks"]) == list(range(8))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_new_owner_starts_with_none_of_the_previous_owners_metadata(credential_db):
    """Whatever a provider kept for the previous owner (Linear: the backfill marker, the
    resume time) describes their syncs, not the new owner's."""
    previous, new = uuid4(), uuid4()
    await store.upsert_credential(
        provider="notion",
        provider_account_id="ws-1",
        user_id=previous,
        token_payload={"access_token": "token-ws-1"},
        provider_metadata={"dlt_seeded": True, "resume_needed_at": "2026-11-01T00:00:00+00:00"},
    )
    await store.revoke_credential_by_account("notion", "ws-1", revoked_by=store.REVOKED_BY_PROVIDER)

    taken_over = await store.upsert_credential(
        provider="notion",
        provider_account_id="ws-1",
        user_id=new,
        token_payload={"access_token": "token-ws-1"},
        provider_metadata={"workspace_name": "Acme"},
    )

    assert taken_over.provider_metadata == {"workspace_name": "Acme"}
