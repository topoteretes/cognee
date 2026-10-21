"""One user, several connections to one provider: which one a lookup returns."""

import asyncio
from uuid import uuid4

import pytest

from cognee.modules.integrations import credentials as store


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
