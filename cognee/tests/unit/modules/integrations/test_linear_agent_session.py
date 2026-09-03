"""Unit tests for cognee.modules.integrations.linear.agent_session.

The Linear activity client, cognee search, and the user lookup are mocked —
what's under test is the turn protocol: the acknowledgement thought is
posted BEFORE any search work (Linear's 10-second responsiveness contract),
every turn ends in a response or error activity, the question is resolved
per event shape, and refusal-only results degrade to the friendly
no-information response instead of parroting a refusal.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

session_module = importlib.import_module("cognee.modules.integrations.linear.agent_session")
adapter_module = importlib.import_module("cognee.modules.integrations.linear.adapter")

handle_agent_session = session_module.handle_agent_session

_USER_ID = uuid4()


def _credential(**overrides):
    defaults = {
        "status": "active",
        "provider_account_id": "org-1",
        "provider_metadata": {"organization_url_key": "acme-co"},
        "user_id": _USER_ID,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _created_payload(question="Where is auth handled?"):
    return {
        "type": "AgentSessionEvent",
        "action": "created",
        "organizationId": "org-1",
        "agentSession": {"id": "sess-1", "comment": {"body": question}},
    }


def _prompted_payload(question="What changed in the API?"):
    return {
        "type": "AgentSessionEvent",
        "action": "prompted",
        "organizationId": "org-1",
        "agentSession": {"id": "sess-1"},
        "agentActivity": {"body": question},
    }


@pytest.fixture
def mocks(monkeypatch):
    owner = SimpleNamespace(id=_USER_ID)
    calls = []
    mocked = SimpleNamespace(
        owner=owner,
        calls=calls,
        search_results=[{"search_result": ["The answer."]}],
    )

    async def _record_activity(access_token, agent_session_id, content):
        calls.append(("activity", content["type"], content["body"]))

    async def _record_search(**_kwargs):
        calls.append(("search",))
        return mocked.search_results

    mocked.activity = AsyncMock(side_effect=_record_activity)
    mocked.search = AsyncMock(side_effect=_record_search)
    mocked.get_user = AsyncMock(return_value=owner)

    monkeypatch.setattr(session_module, "create_agent_activity", mocked.activity)
    monkeypatch.setattr(session_module, "cognee_search", mocked.search)
    monkeypatch.setattr(session_module, "get_user", mocked.get_user)
    # Imported lazily inside handle_agent_session, so patched at its home.
    monkeypatch.setattr(adapter_module, "access_token_for", lambda _credential: "lin_tok")
    return mocked


@pytest.mark.asyncio
async def test_created_acks_with_a_thought_before_any_search_runs(mocks):
    await handle_agent_session(_credential(), _created_payload())

    # The 10-second contract: the thought MUST precede the search, and the
    # turn must end in a response.
    assert [call[:2] for call in mocks.calls] == [
        ("activity", "thought"),
        ("search",),
        ("activity", "response"),
    ]


@pytest.mark.asyncio
async def test_created_answers_with_the_search_result(mocks):
    await handle_agent_session(_credential(), _created_payload())

    kind, activity_type, body = mocks.calls[-1]
    assert (kind, activity_type) == ("activity", "response")
    assert body == "The answer."
    assert mocks.search.await_args.kwargs["query_text"] == "Where is auth handled?"
    assert mocks.search.await_args.kwargs["user"] is mocks.owner
    assert mocks.search.await_args.kwargs["datasets"] is None


@pytest.mark.asyncio
async def test_prompted_takes_the_question_from_the_agent_activity_body(mocks):
    await handle_agent_session(_credential(), _prompted_payload("What changed in the API?"))

    assert mocks.search.await_args.kwargs["query_text"] == "What changed in the API?"
    assert mocks.calls[-1][:2] == ("activity", "response")


@pytest.mark.asyncio
async def test_search_failure_ends_the_turn_in_an_error_activity_without_raising(mocks):
    mocks.search.side_effect = RuntimeError("search exploded")

    await handle_agent_session(_credential(), _created_payload())

    kind, activity_type, _body = mocks.calls[-1]
    assert (kind, activity_type) == ("activity", "error")


@pytest.mark.asyncio
async def test_refusal_only_results_produce_the_no_information_response(mocks):
    mocks.search_results = [
        {"search_result": ["I cannot answer this from the given context."]},
        {"search_result": ["The text does not contain information about auth."]},
    ]

    await handle_agent_session(_credential(), _created_payload())

    kind, activity_type, body = mocks.calls[-1]
    assert (kind, activity_type) == ("activity", "response")
    assert body == "No relevant information found in cognee memory."
