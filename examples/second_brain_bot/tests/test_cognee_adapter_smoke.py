"""Smoke test for the real cognee adapter: it constructs and resolves scope
without importing cognee or touching the network (cognee is imported lazily
inside the ingest/answer/forget methods, not at construction)."""

from second_brain_bot.adapter.cognee_adapter import CogneeChatMemoryAdapter
from second_brain_bot.adapter.interface import Conversation


def test_cognee_adapter_scope_is_per_user_brain():
    adapter = CogneeChatMemoryAdapter()
    convo = Conversation(
        transport="telegram",
        source="chat_42",
        canonical_user="abc-canonical",
        external_user="tg_user_1",
    )
    scope = adapter.scope(convo)
    assert scope.dataset == "brain:abc-canonical"
    assert scope.session == "telegram:chat_42"
