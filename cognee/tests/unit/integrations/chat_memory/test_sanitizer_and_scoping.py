"""Deterministic tests for the chat-memory sanitizer and scope strategies.

Pure functions, no I/O, no keys. These lock down the two contracts every bot
depends on: canonical key derivation, and the two-independent-keys scope shape.
"""

import pytest

from cognee.integrations.chat_memory import (
    Conversation,
    per_channel_scope,
    per_user_scope,
    sanitize_key,
)
from cognee.integrations.chat_memory.sanitizer import sanitize_token


# ---------------------------------------------------------------------------
# sanitizer
# ---------------------------------------------------------------------------
class TestSanitizer:
    def test_lowercases_and_replaces_unsafe(self):
        assert sanitize_token("C123 General!") == "c123-general"

    def test_colon_is_not_a_within_token_char(self):
        # Colons are the segment separator, so a raw token drops them.
        assert ":" not in sanitize_token("a:b")

    def test_empty_token_is_empty(self):
        assert sanitize_token("") == ""
        assert sanitize_token("   ") == ""

    def test_key_joins_with_colon(self):
        assert sanitize_key("chat", "slack", "T1", "C1") == "chat:slack:t1:c1"

    def test_empty_segments_are_dropped(self):
        # A platform with no workspace collapses cleanly instead of "chat:slack::c1".
        assert sanitize_key("chat", "slack", "", "C1") == "chat:slack:c1"

    def test_all_empty_raises(self):
        with pytest.raises(ValueError):
            sanitize_key("", "  ", "!")

    def test_is_deterministic(self):
        assert sanitize_key("chat", "slack", "T1", "C1") == sanitize_key(
            "chat", "slack", "T1", "C1"
        )

    def test_long_key_is_truncated_but_unique(self):
        a = sanitize_key("x" * 400)
        b = sanitize_key("x" * 401)
        assert len(a) <= 255 and len(b) <= 255
        assert a != b  # content hash keeps distinct long keys apart


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------
def _convo(**overrides) -> Conversation:
    base = dict(platform="slack", workspace="T1", channel="C1", user="U1", thread="th1")
    base.update(overrides)
    return Conversation(**base)


class TestScoping:
    def test_scope_has_two_independent_keys(self):
        scope = per_channel_scope(_convo())
        assert scope.dataset and scope.session
        assert hasattr(scope, "dataset") and hasattr(scope, "session")

    def test_per_channel_dataset_is_the_channel(self):
        scope = per_channel_scope(_convo())
        assert scope.dataset == "chat:slack:t1:c1"
        assert scope.session == "slack:t1:c1:th1"

    def test_per_channel_dataset_ignores_thread(self):
        # Two threads in one channel share the channel dataset (shared memory)
        # but have distinct sessions (distinct live context).
        a = per_channel_scope(_convo(thread="th1"))
        b = per_channel_scope(_convo(thread="th2"))
        assert a.dataset == b.dataset
        assert a.session != b.session

    def test_per_user_decouples_dataset_from_conversation(self):
        # A per-user brain's dataset is keyed by the user, not the channel, so
        # it stays stable across transports.
        web = per_user_scope(_convo(platform="web", channel="web-1", thread=None))
        tele = per_user_scope(_convo(platform="telegram", channel="tg-9", thread=None))
        assert web.dataset == tele.dataset == "brain:u1"
        # The live session still follows the transport.
        assert web.session != tele.session

    def test_missing_identity_segment_raises(self):
        # A channel that sanitizes to empty must not silently alias into one
        # dataset — the privacy/forget boundary has to stay distinct.
        with pytest.raises(ValueError):
            per_channel_scope(_convo(channel="!!!"))
        with pytest.raises(ValueError):
            per_user_scope(_convo(user=""))

    def test_collapsed_case_is_representable(self):
        # A per-channel bot that wants dataset == session can build one; the
        # point is the shape *allows* both collapsed and decoupled.
        scope = per_channel_scope(_convo(thread=None))
        # dataset and session differ by prefix here, but both are valid keys.
        assert isinstance(scope.dataset, str) and isinstance(scope.session, str)
