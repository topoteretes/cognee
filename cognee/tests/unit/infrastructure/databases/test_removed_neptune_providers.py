"""The Neptune adapters moved to cognee-community; core no longer accepts their provider names."""

import pytest

from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine


@pytest.mark.parametrize("provider", ["neptune", "neptune_analytics"])
def test_neptune_graph_provider_is_unsupported(provider):
    with pytest.raises(OSError, match=f"Unsupported graph database provider: {provider}"):
        create_graph_engine(
            graph_database_provider=provider,
            graph_file_path="",
            graph_database_url="neptune-graph://g-abc123",
        )


def test_neptune_analytics_vector_provider_is_unsupported():
    with pytest.raises(
        OSError, match="Unsupported vector database provider: neptune_analytics"
    ) as error:
        create_vector_engine(
            vector_db_provider="neptune_analytics",
            vector_db_url="neptune-graph://g-abc123",
            vector_db_name="",
        )

    assert "neptune_analytics" not in str(error.value).split("Supported providers are:")[1]
