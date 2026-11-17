"""EmbeddingGateway returns an engine's vectors and records its token usage,
which never fails the embedding."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingResult,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import (
    EmbeddingGateway,
)

RECORD = "cognee.modules.session_lifecycle.usage_tracking.record_embedding_call"


class _Engine:
    model = "fake-embedding-model"

    def get_vector_size(self):
        return 2

    async def embed_text(self, text):
        return EmbeddingResult([[1.0, 0.0] for _ in text], tokens=7)


class _PlainListEngine(_Engine):
    """A custom engine written before token counting: returns a plain list."""

    async def embed_text(self, text):
        return [[1.0, 0.0] for _ in text]


class _SizedEngine(_Engine):
    """An engine that sends two texts per request and records every request."""

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.requests = []

    def get_batch_size(self):
        return self.batch_size

    async def embed_text(self, text):
        self.requests.append(list(text))
        return EmbeddingResult([[float(len(value)), 0.0] for value in text], tokens=len(text))


@pytest.mark.asyncio
async def test_embed_text_returns_vectors_and_records_the_tokens():
    with patch(RECORD, new=AsyncMock()) as record:
        vectors = await EmbeddingGateway(_Engine()).embed_text(["a", "b"])

    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    record.assert_awaited_once_with(model="fake-embedding-model", tokens=7)


@pytest.mark.asyncio
async def test_an_engine_returning_a_plain_list_fails():
    with pytest.raises(AttributeError):
        await EmbeddingGateway(_PlainListEngine()).embed_text(["a"])


@pytest.mark.asyncio
async def test_a_recording_failure_does_not_fail_the_embedding():
    with patch(RECORD, new=AsyncMock(side_effect=RuntimeError("usage store down"))):
        vectors = await EmbeddingGateway(_Engine()).embed_text(["a"])

    assert vectors == [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_more_texts_than_one_request_carries_go_in_consecutive_requests():
    """A write batch larger than the provider's request size is embedded request by
    request, in order, with the token counts summed and recorded once."""
    engine = _SizedEngine(batch_size=2)
    with patch(RECORD, new=AsyncMock()) as record:
        vectors = await EmbeddingGateway(engine).embed_text(["a", "bb", "ccc", "dddd", "e"])

    assert engine.requests == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [1.0, 0.0]]
    assert vectors.tokens == 5
    record.assert_awaited_once_with(model="fake-embedding-model", tokens=5)


@pytest.mark.asyncio
async def test_a_request_sized_call_is_one_request():
    engine = _SizedEngine(batch_size=3)
    with patch(RECORD, new=AsyncMock()):
        vectors = await EmbeddingGateway(engine).embed_text(["a", "b", "c"])

    assert engine.requests == [["a", "b", "c"]]
    assert vectors.tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 0, -4, True])
async def test_an_engine_without_a_usable_request_size_gets_one_request(batch_size):
    engine = _SizedEngine(batch_size=batch_size)
    with patch(RECORD, new=AsyncMock()):
        vectors = await EmbeddingGateway(engine).embed_text(["a", "b", "c"])

    assert engine.requests == [["a", "b", "c"]]
    assert len(vectors) == 3


def test_other_attributes_pass_through_to_the_engine():
    gateway = EmbeddingGateway(_Engine())

    assert gateway.get_vector_size() == 2
    assert gateway.model == "fake-embedding-model"


def test_embedding_vectors_behave_as_the_plain_list_they_replace():
    vectors = EmbeddingResult([[1.0, 0.0], [0.0, 1.0]], tokens=5)

    assert isinstance(vectors, list)
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert repr(vectors) == "[[1.0, 0.0], [0.0, 1.0]]"
    assert json.loads(json.dumps(vectors)) == [[1.0, 0.0], [0.0, 1.0]]
    assert vectors.tokens == 5


def test_reading_a_count_that_was_not_given_raises():
    with pytest.raises(ValueError):
        _ = EmbeddingResult([[1.0, 0.0]]).tokens


def test_a_new_list_built_from_the_result_has_no_count():
    with pytest.raises(AttributeError):
        _ = EmbeddingResult([[1.0, 0.0], [0.0, 1.0]], tokens=5)[:1].tokens
