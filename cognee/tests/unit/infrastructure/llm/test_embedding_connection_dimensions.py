from types import SimpleNamespace

import pytest

from cognee.infrastructure.llm import utils as llm_utils


class _FakeEmbeddingEngine:
    def __init__(self, vector, dimensions):
        self._vector = vector
        self.dimensions = dimensions

    async def embed_text(self, _texts):
        return [self._vector]


@pytest.mark.asyncio
async def test_embedding_connection_returns_detected_dimensions(monkeypatch):
    fake_engine = _FakeEmbeddingEngine(vector=[0.1, 0.2, 0.3], dimensions=3072)
    fake_vector_engine = SimpleNamespace(embedding_engine=fake_engine)

    import cognee.infrastructure.databases.vector as vector_module

    monkeypatch.setattr(vector_module, "get_vector_engine", lambda: fake_vector_engine)
    assert await llm_utils.test_embedding_connection() == 3


def test_determine_embedding_dimensions_uses_env_dimensions_when_provided(monkeypatch):
    fake_engine = _FakeEmbeddingEngine(vector=[0.1, 0.2, 0.3], dimensions=3072)
    fake_vector_engine = SimpleNamespace(embedding_engine=fake_engine)
    fake_embedding_config = SimpleNamespace(embedding_dimensions=3072)

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    import cognee.infrastructure.databases.vector as vector_module
    import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module

    monkeypatch.setattr(vector_module, "get_vector_engine", lambda: fake_vector_engine)
    monkeypatch.setattr(
        embedding_config_module, "get_embedding_config", lambda: fake_embedding_config
    )

    llm_utils.determine_embedding_dimensions(3)

    assert fake_embedding_config.embedding_dimensions == 3072
    assert fake_engine.dimensions == 3072


def test_determine_embedding_dimensions_uses_detected_dimensions_when_env_missing(monkeypatch):
    fake_engine = _FakeEmbeddingEngine(vector=[0.1, 0.2, 0.3], dimensions=3072)
    fake_vector_engine = SimpleNamespace(embedding_engine=fake_engine)
    fake_embedding_config = SimpleNamespace(embedding_dimensions=3072)

    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    import cognee.infrastructure.databases.vector as vector_module
    import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module

    monkeypatch.setattr(vector_module, "get_vector_engine", lambda: fake_vector_engine)
    monkeypatch.setattr(
        embedding_config_module, "get_embedding_config", lambda: fake_embedding_config
    )

    llm_utils.determine_embedding_dimensions(3)

    assert fake_embedding_config.embedding_dimensions == 3
    assert fake_engine.dimensions == 3
