"""Smoke test for the real cognee adapter: it constructs, and dataset_for resolves
the per-user brain, without importing cognee or touching the network (cognee is
imported lazily inside ingest/answer/forget, not at construction)."""

from second_brain_bot.adapter.cognee_adapter import CogneeChatMemoryAdapter
from second_brain_bot.adapter.interface import Conversation, dataset_for


def test_cognee_adapter_constructs_without_cognee():
    CogneeChatMemoryAdapter()  # no cognee import, no network


def test_dataset_is_per_user_brain():
    convo = Conversation(
        transport="telegram",
        source="chat_42",
        canonical_user="abc-canonical",
        external_user="tg_user_1",
    )
    assert dataset_for(convo) == "brain:abc-canonical"
