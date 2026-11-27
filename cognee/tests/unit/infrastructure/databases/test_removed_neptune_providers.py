"""The Neptune adapters moved to cognee-community; core no longer accepts their provider names."""

import pytest

from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.infrastructure.databases.vector import create_vector_engine as vector_factory

COMMUNITY_REPOSITORY = "https://github.com/topoteretes/cognee-community"


@pytest.mark.parametrize("provider", ["neptune", "neptune_analytics"])
def test_neptune_graph_provider_is_unsupported(provider):
    with pytest.raises(OSError, match=f"Unsupported graph database provider: {provider}") as error:
        create_graph_engine(
            graph_database_provider=provider,
            graph_file_path="",
            graph_database_url="neptune-graph://g-abc123",
        )

    assert COMMUNITY_REPOSITORY in str(error.value)


def test_neptune_analytics_vector_provider_is_unsupported(monkeypatch):
    # The factory builds the embedding engine before it looks at the provider;
    # stub it so the test needs no embedding model or key.
    monkeypatch.setattr(vector_factory, "get_embedding_engine", lambda: None)
    monkeypatch.setattr(vector_factory, "EmbeddingGateway", lambda engine: engine)

    with pytest.raises(
        OSError, match="Unsupported vector database provider: neptune_analytics"
    ) as error:
        vector_factory.create_vector_engine(
            vector_db_provider="neptune_analytics",
            vector_db_url="neptune-graph://g-abc123",
            vector_db_name="",
        )

    message = str(error.value)
    assert "neptune_analytics" not in message.split("Supported providers are:")[1]
    assert COMMUNITY_REPOSITORY in message
