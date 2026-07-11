"""Unit tests for the thin chat-memory adapter interface (issue #3609, commit 1).

Covers only the framework-agnostic contract: the session/dataset mapping helper,
the value types, and abstractness of the interface. No cognee, no Slack, no keys.
"""

import asyncio
import uuid

import pytest

from src.memory_adapter import (
    SLACK_UUID_NAMESPACE,
    Answer,
    ChatMemory,
    Citation,
    ConversationRef,
    message_data_id,
)


# --------------------------------------------------------------------------- #
# Session / dataset mapping helper                                            #
# --------------------------------------------------------------------------- #


def test_dataset_name_is_channel_scoped():
    ref = ConversationRef(team_id="T1", channel_id="C42", thread_ts="1700000000.1")
    assert ref.dataset_name == "slack_C42"


def test_node_set_tags_the_channel():
    ref = ConversationRef(team_id="T1", channel_id="C42")
    assert ref.node_set == ["C42"]


def test_mapping_is_deterministic():
    a = ConversationRef(team_id="T1", channel_id="C42", thread_ts="99.1")
    b = ConversationRef(team_id="T1", channel_id="C42", thread_ts="99.1")
    assert a == b
    assert a.dataset_name == b.dataset_name
    # Frozen dataclass is hashable (usable as a dict/queue key).
    assert hash(a) == hash(b)


def test_message_data_id_is_deterministic_and_channel_ts_scoped():
    first = message_data_id("C42", "1700000000.000100")
    second = message_data_id("C42", "1700000000.000100")
    assert isinstance(first, uuid.UUID)
    assert first == second
    # Matches the documented uuid5(namespace, "channel:ts") derivation exactly.
    assert first == uuid.uuid5(SLACK_UUID_NAMESPACE, "C42:1700000000.000100")


def test_message_data_id_differs_per_message():
    a = message_data_id("C42", "1700000000.000100")
    b = message_data_id("C42", "1700000000.000200")
    c = message_data_id("C99", "1700000000.000100")
    assert a != b
    assert a != c


# --------------------------------------------------------------------------- #
# Value types                                                                 #
# --------------------------------------------------------------------------- #


def test_citation_defaults_ok_true_and_answer_defaults_to_no_citations():
    # The two defaults the code relies on: a Citation is a good link unless marked
    # otherwise, and an Answer with no sources has an empty (not None) citation list.
    assert Citation(channel_id="C", ts="1", permalink="p", author="a", snippet="s").ok is True
    assert Answer(text="I don't know yet.").citations == []


# --------------------------------------------------------------------------- #
# Interface contract                                                          #
# --------------------------------------------------------------------------- #


def test_chatmemory_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ChatMemory()  # type: ignore[abstract]


def test_complete_subclass_satisfies_the_contract():
    class Fake(ChatMemory):
        def __init__(self):
            self.calls = []

        async def ingest(self, ref, *, ts, text, permalink, author):
            self.calls.append(("ingest", ref, ts, text, permalink, author))

        async def flush(self, ref):
            self.calls.append(("flush", ref))

        async def answer(self, ref, *, query):
            self.calls.append(("answer", ref, query))
            return Answer(text=f"echo:{query}")

        async def forget(self, ref):
            self.calls.append(("forget", ref))

    fake = Fake()
    ref = ConversationRef(team_id="T1", channel_id="C42")

    async def _exercise():
        await fake.ingest(
            ref, ts="1.0", text="hi", permalink="https://slack.example/x", author="alice"
        )
        await fake.flush(ref)
        result = await fake.answer(ref, query="what did we decide?")
        await fake.forget(ref)
        return result

    answer = asyncio.run(_exercise())

    assert isinstance(answer, Answer)
    assert answer.text == "echo:what did we decide?"
    assert [c[0] for c in fake.calls] == ["ingest", "flush", "answer", "forget"]
