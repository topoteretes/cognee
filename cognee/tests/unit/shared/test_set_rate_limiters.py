"""set_rate_limiters: a host-supplied limiter is the one dispatch enters."""

from types import SimpleNamespace

import pytest

import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module
import cognee.infrastructure.llm.config as llm_config_module
from cognee.shared import rate_limiting
from cognee.shared.rate_limiting import (
    embedding_rate_limiter_context_manager,
    llm_rate_limiter_context_manager,
    set_rate_limiters,
)


class CountingLimiter:
    def __init__(self):
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture(autouse=True)
def paced(monkeypatch):
    """Pacing on, and the module globals restored after each test."""
    monkeypatch.setattr(rate_limiting, "_llm_rate_limiter", None)
    monkeypatch.setattr(rate_limiting, "_embedding_rate_limiter", None)
    monkeypatch.setattr(
        llm_config_module,
        "get_llm_config",
        lambda: SimpleNamespace(
            llm_rate_limit_enabled=True,
            auto_rate_limit=False,
            llm_rate_limit_requests=60,
            llm_rate_limit_interval=60,
        ),
    )
    monkeypatch.setattr(
        embedding_config_module,
        "get_embedding_config",
        lambda: SimpleNamespace(
            embedding_rate_limit_enabled=True,
            embedding_rate_limit_requests=60,
            embedding_rate_limit_interval=60,
        ),
    )


@pytest.mark.asyncio
async def test_supplied_limiters_pace_dispatch():
    llm, embedding = CountingLimiter(), CountingLimiter()
    set_rate_limiters(llm=llm, embedding=embedding)

    async with llm_rate_limiter_context_manager():
        pass
    async with embedding_rate_limiter_context_manager():
        pass

    assert (llm.entered, embedding.entered) == (1, 1)


@pytest.mark.asyncio
async def test_none_restores_the_lazy_default():
    set_rate_limiters(llm=CountingLimiter(), embedding=CountingLimiter())
    set_rate_limiters()

    async with llm_rate_limiter_context_manager():
        pass

    assert isinstance(rate_limiting._llm_rate_limiter, rate_limiting.AsyncLimiter)
    assert isinstance(embedding_rate_limiter_context_manager(), rate_limiting.AsyncLimiter)
