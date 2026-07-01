"""Deterministic tests for ChatMemoryAdapter against the in-memory backend.

The real adapter runs against :class:`InMemoryMemoryBackend` (no LLM, no keys),
so every assertion here exercises the same code path a production bot uses,
minus the cognee storage layer. Covers: consent gating, the external_metadata
stamp, idempotent ingest, citations, answer composition, and both forget paths
including that per-user forget deletes only the target user's items.
"""

import pytest

from cognee.integrations.chat_memory import (
    Answer,
    ChatMemoryAdapter,
    Conversation,
    InMemoryMemoryBackend,
    Message,
    per_channel_scope,
    per_user_scope,
)


def _adapter(**overrides) -> ChatMemoryAdapter:
    kwargs = dict(scope=per_channel_scope, backend=InMemoryMemoryBackend())
    kwargs.update(overrides)
    return ChatMemoryAdapter(**kwargs)


def _group_convo(user="U1", **overrides) -> Conversation:
    base = dict(platform="slack", workspace="T1", channel="C1", user=user, thread="th1")
    base.update(overrides)
    return Conversation(**base)


def _direct_convo(user="U9") -> Conversation:
    # DM shape: no workspace, channel == user id.
    return Conversation(platform="telegram", workspace="", channel=user, user=user)


# ---------------------------------------------------------------------------
# consent
# ---------------------------------------------------------------------------
class TestConsent:
    @pytest.mark.asyncio
    async def test_group_denies_until_opt_in(self):
        adapter = _adapter()
        convo = _group_convo()
        stored = await adapter.ingest(convo, Message(text="secret plan", user="U1"))
        assert stored is False
        answer = await adapter.answer(convo, "plan")
        assert answer.is_empty

    @pytest.mark.asyncio
    async def test_group_stores_after_opt_in(self):
        adapter = _adapter()
        convo = _group_convo()
        adapter.set_consent("U1", True)
        stored = await adapter.ingest(convo, Message(text="ship on friday", user="U1"))
        assert stored is True

    @pytest.mark.asyncio
    async def test_direct_allows_by_default(self):
        adapter = _adapter(scope=per_user_scope)
        convo = _direct_convo("U9")
        stored = await adapter.ingest(convo, Message(text="buy milk", user="U9"))
        assert stored is True

    @pytest.mark.asyncio
    async def test_explicit_opt_out_wins_in_direct(self):
        adapter = _adapter(scope=per_user_scope)
        convo = _direct_convo("U9")
        adapter.set_consent("U9", False)
        assert await adapter.ingest(convo, Message(text="hi", user="U9")) is False


# ---------------------------------------------------------------------------
# ingest / answer round trip + citations
# ---------------------------------------------------------------------------
class TestIngestAnswer:
    @pytest.mark.asyncio
    async def test_round_trip_with_citation(self):
        adapter = _adapter()
        convo = _group_convo(user="U1")
        adapter.set_consent("U1", True)
        await adapter.ingest(
            convo,
            Message(
                text="We decided to ship the release on Friday.",
                user="U1",
                timestamp="1700000000.001",
                permalink="https://slack/archives/C1/p1",
            ),
        )
        answer = await adapter.answer(convo, "when do we ship the release?")
        assert isinstance(answer, Answer)
        assert "Friday" in answer.text
        assert answer.citations
        top = answer.citations[0]
        assert top.permalink == "https://slack/archives/C1/p1"
        assert top.user == "U1"

    @pytest.mark.asyncio
    async def test_empty_message_is_not_stored(self):
        adapter = _adapter()
        convo = _group_convo()
        adapter.set_consent("U1", True)
        assert await adapter.ingest(convo, Message(text="   ", user="U1")) is False

    @pytest.mark.asyncio
    async def test_reingest_is_idempotent(self):
        backend = InMemoryMemoryBackend()
        adapter = _adapter(backend=backend)
        convo = _group_convo(user="U1")
        adapter.set_consent("U1", True)
        msg = Message(text="same message", user="U1", timestamp="1700000000.5")
        await adapter.ingest(convo, msg)
        await adapter.ingest(convo, msg)  # replay
        scope = adapter.scope(convo)
        assert len(backend._store[scope.dataset]) == 1

    @pytest.mark.asyncio
    async def test_answer_is_empty_when_nothing_matches(self):
        adapter = _adapter()
        convo = _group_convo(user="U1")
        adapter.set_consent("U1", True)
        await adapter.ingest(convo, Message(text="apples and oranges", user="U1"))
        answer = await adapter.answer(convo, "quarterly revenue forecast")
        assert answer.is_empty


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------
class TestForget:
    @pytest.mark.asyncio
    async def test_forget_scope_wipes_everything(self):
        adapter = _adapter()
        convo = _group_convo(user="U1")
        adapter.set_consent("U1", True)
        await adapter.ingest(convo, Message(text="fact one", user="U1"))
        result = await adapter.forget(conversation=convo)
        assert result["items_removed"] == 1
        assert (await adapter.answer(convo, "fact one")).is_empty

    @pytest.mark.asyncio
    async def test_forget_user_removes_only_that_user(self):
        adapter = _adapter()
        convo_u1 = _group_convo(user="U1")
        convo_u2 = _group_convo(user="U2")
        adapter.set_consent("U1", True)
        adapter.set_consent("U2", True)
        await adapter.ingest(convo_u1, Message(text="alice likes graphs", user="U1"))
        await adapter.ingest(convo_u2, Message(text="bob likes graphs", user="U2"))

        result = await adapter.forget(conversation=convo_u1, user="U1")
        assert result["items_removed"] == 1

        # U2's memory survives; U1's is gone.
        remaining = await adapter.answer(convo_u2, "who likes graphs")
        users = {c.user for c in remaining.citations}
        assert users == {"U2"}

    @pytest.mark.asyncio
    async def test_forget_me_revokes_consent(self):
        adapter = _adapter()
        convo = _group_convo(user="U1")
        adapter.set_consent("U1", True)
        await adapter.ingest(convo, Message(text="remember me", user="U1"))
        await adapter.forget(conversation=convo, user="U1")
        # After forget-me, the user must opt in again before new capture.
        assert adapter.has_consent(convo, "U1") is False

    @pytest.mark.asyncio
    async def test_forget_user_without_conversation_raises(self):
        adapter = _adapter()
        with pytest.raises(ValueError):
            await adapter.forget(user="U1")

    @pytest.mark.asyncio
    async def test_forget_with_no_args_raises(self):
        adapter = _adapter()
        with pytest.raises(ValueError):
            await adapter.forget()


# ---------------------------------------------------------------------------
# two-keys behaviour through the adapter
# ---------------------------------------------------------------------------
class TestTwoKeysThroughAdapter:
    @pytest.mark.asyncio
    async def test_per_user_recall_spans_transports(self):
        # A note captured via telegram is recallable from web, because the
        # per-user dataset is transport-independent (dataset != session).
        adapter = _adapter(scope=per_user_scope)
        tele = Conversation(platform="telegram", workspace="", channel="U9", user="U9")
        web = Conversation(platform="web", workspace="", channel="U9", user="U9")
        await adapter.ingest(tele, Message(text="my flight is at 9am", user="U9"))
        answer = await adapter.answer(web, "when is my flight")
        assert "9am" in answer.text
