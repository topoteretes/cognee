import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def tapes_adapter():
    """TapesCacheAdapter rooted in a temp cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.tapes.TapesCacheAdapter import (
                TapesCacheAdapter,
            )

            inst = TapesCacheAdapter(
                tapes_ingest_url="http://tapes.test:8081",
                tapes_agent_name="cognee-unit",
                tapes_model="unit-model",
            )
            yield inst
            inst.cache.close()


def _patched_post_returning(status_code: int = 202):
    """Build a mock httpx.AsyncClient.post returning a response with given status."""
    response = MagicMock()
    response.status_code = status_code
    response.text = ""
    return AsyncMock(return_value=response)


@pytest.mark.asyncio
async def test_create_qa_entry_writes_fs_and_mirrors_to_tapes(tapes_adapter):
    mock_post = _patched_post_returning(202)
    with patch("httpx.AsyncClient.post", mock_post):
        await tapes_adapter.create_qa_entry(
            user_id="u1",
            session_id="s1",
            question="What is cognee?",
            context="Background context here.",
            answer="An AI memory platform.",
            qa_id="qa-1",
        )

    entries = await tapes_adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0]["qa_id"] == "qa-1"
    assert entries[0]["answer"] == "An AI memory platform."

    assert mock_post.await_count == 1
    call = mock_post.await_args
    assert call.args[0] == "http://tapes.test:8081/v1/ingest"
    payload = call.kwargs["json"]
    assert payload["provider"] == "openai"
    assert payload["agent_name"] == "cognee-unit"
    assert payload["request"]["model"] == "unit-model"
    assert payload["request"]["messages"] == [
        {"role": "system", "content": "Background context here."},
        {"role": "user", "content": "What is cognee?"},
    ]
    assert payload["response"]["model"] == "unit-model"
    assert payload["response"]["choices"][0]["message"]["content"] == "An AI memory platform."


@pytest.mark.asyncio
async def test_ingest_failure_does_not_break_fs_write(tapes_adapter):
    mock_post = AsyncMock(side_effect=RuntimeError("tapes unreachable"))
    with patch("httpx.AsyncClient.post", mock_post):
        await tapes_adapter.create_qa_entry(
            user_id="u1",
            session_id="s1",
            question="Q?",
            context="",
            answer="A.",
            qa_id="qa-2",
        )

    entries = await tapes_adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0]["qa_id"] == "qa-2"


@pytest.mark.asyncio
async def test_anthropic_provider_shape(tapes_adapter):
    tapes_adapter.tapes_provider = "anthropic"
    mock_post = _patched_post_returning(202)
    with patch("httpx.AsyncClient.post", mock_post):
        await tapes_adapter.create_qa_entry(
            user_id="u1",
            session_id="s1",
            question="Hi?",
            context="sys",
            answer="Hello.",
            qa_id="qa-3",
        )

    payload = mock_post.await_args.kwargs["json"]
    assert payload["provider"] == "anthropic"
    assert payload["request"]["system"] == "sys"
    assert payload["request"]["messages"] == [{"role": "user", "content": "Hi?"}]
    assert payload["response"]["content"] == [{"type": "text", "text": "Hello."}]


@pytest.mark.asyncio
async def test_update_and_delete_are_not_mirrored(tapes_adapter):
    """Updates and deletes stay local: tapes is append-only."""
    mock_post = _patched_post_returning(202)
    with patch("httpx.AsyncClient.post", mock_post):
        await tapes_adapter.create_qa_entry(
            user_id="u1",
            session_id="s1",
            question="Q?",
            context="",
            answer="A.",
            qa_id="qa-4",
        )
        assert mock_post.await_count == 1

        updated = await tapes_adapter.update_qa_entry(
            user_id="u1",
            session_id="s1",
            qa_id="qa-4",
            feedback_score=5,
        )
        assert updated is True

        deleted = await tapes_adapter.delete_qa_entry("u1", "s1", "qa-4")
        assert deleted is True

    # Only the initial create_qa_entry should have reached tapes.
    assert mock_post.await_count == 1
