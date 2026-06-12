from unittest.mock import patch

import pytest

from cognee.modules.retrieval.utils import lexical_corpus_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    lexical_corpus_cache.invalidate()
    yield
    lexical_corpus_cache.invalidate()


def test_put_get_roundtrip_and_unknown_key():
    state = {"chunks": {"a": ["token"]}}

    lexical_corpus_cache.put("key", state)

    assert lexical_corpus_cache.get("key") is state
    assert lexical_corpus_cache.get("other") is None


def test_entry_expires_after_ttl():
    with patch("cognee.modules.retrieval.utils.lexical_corpus_cache.time") as mock_time:
        mock_time.monotonic.side_effect = [0.0, 30.0, 1000.0]
        lexical_corpus_cache.put("key", {"chunks": {}})

        assert lexical_corpus_cache.get("key") is not None
        assert lexical_corpus_cache.get("key") is None


def test_invalidate_clears_all_entries():
    lexical_corpus_cache.put("key-1", {"chunks": {}})
    lexical_corpus_cache.put("key-2", {"chunks": {}})

    lexical_corpus_cache.invalidate()

    assert lexical_corpus_cache.get("key-1") is None
    assert lexical_corpus_cache.get("key-2") is None


def test_oldest_entry_is_evicted_beyond_max_contexts():
    for index in range(lexical_corpus_cache.MAX_CACHED_CONTEXTS + 1):
        lexical_corpus_cache.put(f"key-{index}", {"chunks": {}})

    assert lexical_corpus_cache.get("key-0") is None
    assert lexical_corpus_cache.get("key-1") is not None


@pytest.mark.asyncio
async def test_lock_is_stable_per_key_within_a_loop():
    assert lexical_corpus_cache.lock("key") is lexical_corpus_cache.lock("key")
    assert lexical_corpus_cache.lock("key") is not lexical_corpus_cache.lock("other")
