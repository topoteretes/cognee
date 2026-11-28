"""An over-length text is embedded as consecutive token windows by every engine.

The cutter (``windows_by_count``) against a word tokenizer and the real TikToken one,
the token-weighted pooling, and each engine that cuts with it (LiteLLM, the
OpenAI-compatible engine, Ollama, the embedding worker): one request carries every
window of the batch, each text gets one vector, and a text that fits is sent as it is.
fastembed cuts from its own tokenizer's encoding; see
``test_embedding_input_limit_engines.py``.
"""

from __future__ import annotations

import base64
import random
from types import SimpleNamespace

import numpy as np
import pytest

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.token_windows import (
    pool_windows,
    windows_by_count,
)

LIMIT = 5


class WordTokenizer:
    """One token per whitespace-separated word, and one per started three
    characters of a word longer than that (so a long word can exceed the limit)."""

    def count_tokens(self, text: str) -> int:
        return sum(max(1, -(-len(word) // 3)) for word in text.split())


def count(text: str) -> int:
    return WordTokenizer().count_tokens(text)


def vector_for(text: str) -> list[float]:
    """A vector that records the text it was made from: its length, and a constant."""
    return [float(len(text)), 1.0]


def words(n: int) -> str:
    return " ".join(f"w{i % 10}" for i in range(n))


def use_limit(engine, limit: int = LIMIT) -> None:
    """Give an engine a resolved input limit and the word tokenizer."""
    engine.tokenizer = WordTokenizer()
    engine.input_limit_resolved = True
    engine.max_completion_tokens = limit


# --------------------------------------------------------------------------- #
# The cutter and the pooling
# --------------------------------------------------------------------------- #


def test_a_text_that_fits_is_one_window():
    assert windows_by_count("w1 w2 w3", count, LIMIT) == [("w1 w2 w3", 3)]


def test_an_over_length_text_is_cut_into_consecutive_windows_within_the_limit():
    text = words(23)

    windows = windows_by_count(text, count, LIMIT)

    assert "".join(window for window, _ in windows) == text, "nothing lost, no overlap"
    assert [tokens for _, tokens in windows] == [5, 5, 5, 5, 3]
    assert all(count(window) == tokens for window, tokens in windows)


def test_a_word_longer_than_the_limit_is_cut_inside():
    word = "x" * 40  # 14 tokens by the word tokenizer
    text = f"a {word} b"

    windows = windows_by_count(text, count, LIMIT)

    assert "".join(window for window, _ in windows) == text
    assert all(tokens <= LIMIT for _, tokens in windows)
    assert len(windows) == 4


def test_windows_are_given_back_words_when_the_whole_counts_more_than_its_words():
    # A tokenizer whose count of a joined text exceeds the sum of its words' counts.
    def merging_count(text: str) -> int:
        return len(text.split()) + text.count(" w")

    text = words(12)
    windows = windows_by_count(text, merging_count, LIMIT)

    assert "".join(window for window, _ in windows) == text
    assert all(merging_count(window) <= LIMIT for window, _ in windows)


@pytest.mark.parametrize(
    "text",
    [
        " ".join(
            random.Random(7).choice(["memory", "graph", "Ladybug", "1,024", "naïve", "图谱"])
            for _ in range(3000)
        ),
        base64.b64encode(random.Random(7).randbytes(6000)).decode(),
        "知识图谱是一种用图模型来描述知识的技术方法。" * 200,
    ],
    ids=["prose", "base64", "cjk"],
)
def test_tiktoken_windows_stay_within_the_limit_and_join_back(text):
    from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer

    tokenizer = TikTokenTokenizer()
    limit = 512

    windows = windows_by_count(text, tokenizer.count_tokens, limit)

    assert "".join(window for window, _ in windows) == text
    assert all(tokenizer.count_tokens(window) <= limit for window, _ in windows)
    total = tokenizer.count_tokens(text)
    assert len(windows) <= -(-total // limit) + 2, "windows are packed, not cut short"


def test_pooling_weighs_each_window_by_its_tokens():
    windows = [[("a", 1)], [("b", 3), ("c", 1)]]
    vectors = [[9.0, 9.0], [4.0, 0.0], [0.0, 4.0]]

    pooled = pool_windows(windows, vectors)

    assert pooled == [[9.0, 9.0], [3.0, 1.0]]
    unit = pool_windows(windows, vectors, unit_length=True)
    assert unit[0] == [9.0, 9.0], "a single window is its own vector"
    assert np.linalg.norm(unit[1]) == pytest.approx(1.0)


def expected_vector(text: str) -> list[float]:
    windows = windows_by_count(text, count, LIMIT)
    return pool_windows([windows], [vector_for(window) for window, _ in windows])[0]


# --------------------------------------------------------------------------- #
# The engines
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_litellm_sends_every_window_of_the_batch_in_one_request(monkeypatch):
    import litellm

    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    engine = LiteLLMEmbeddingEngine(dimensions=2, encoding_format="float")
    use_limit(engine)
    requests: list[list[str]] = []

    async def fake_aembedding(**kwargs):
        requests.append(kwargs["input"])
        return SimpleNamespace(
            data=[{"embedding": vector_for(text)} for text in kwargs["input"]],
            usage=SimpleNamespace(prompt_tokens=sum(count(text) for text in kwargs["input"])),
        )

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    long_text = words(12)

    result = await engine.embed_text(["short", long_text, "also short"])

    assert len(requests) == 1
    assert requests[0] == ["short", *[w for w, _ in windows_by_count(long_text, count, 5)]] + [
        "also short"
    ]
    assert list(result) == [
        vector_for("short"),
        expected_vector(long_text),
        vector_for("also short"),
    ]
    assert result.tokens == count("short") + 12 + count("also short"), "counted once"

    requests.clear()
    await engine.embed_text(["short", "also short"])
    assert requests == [["short", "also short"]], "texts that fit go as they are"


@pytest.mark.asyncio
async def test_openai_compatible_sends_every_window_in_one_request(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
        OpenAICompatibleEmbeddingEngine,
    )

    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model", dimensions=2, endpoint="http://localhost:8099", api_key="k"
    )
    use_limit(engine)
    requests: list[list[str]] = []

    async def fake_create(**kwargs):
        requests.append(kwargs["input"])
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector_for(text)) for text in kwargs["input"]],
            usage=SimpleNamespace(prompt_tokens=sum(count(text) for text in kwargs["input"])),
        )

    monkeypatch.setattr(engine._client.embeddings, "create", fake_create)
    long_text = words(11)

    result = await engine.embed_text([long_text, "short"])

    assert requests == [[w for w, _ in windows_by_count(long_text, count, 5)] + ["short"]]
    assert list(result) == [expected_vector(long_text), vector_for("short")]
    assert result.tokens == 11 + count("short")


@pytest.mark.asyncio
async def test_ollama_embeds_each_window_and_pools_them(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
        OllamaEmbeddingEngine,
    )

    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    engine = OllamaEmbeddingEngine(model="test-model", dimensions=2)
    use_limit(engine)
    prompts: list[str] = []

    async def fake_get_embedding(prompt):
        prompts.append(prompt)
        return EmbeddingResult(vectors=[vector_for(prompt)], tokens=count(prompt))

    monkeypatch.setattr(engine, "_get_embedding", fake_get_embedding)
    long_text = words(12)

    result = await engine.embed_text(["short", long_text])

    assert prompts == ["short", *[w for w, _ in windows_by_count(long_text, count, 5)]]
    assert list(result) == [vector_for("short"), expected_vector(long_text)]
    assert result.tokens == count("short") + 12
