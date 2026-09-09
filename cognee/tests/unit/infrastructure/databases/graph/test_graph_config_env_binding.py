"""Regression tests for GraphConfig environment-variable binding.

GraphConfig previously declared fields with the deprecated pydantic
``Field(..., env="...")`` keyword. pydantic-settings ignores ``env=`` (it binds
each field from its upper-cased name), so the keyword only emitted
``PydanticDeprecatedSince20`` warnings. These tests pin that:

1. fields still bind from their environment variables by name, and
2. importing the config module emits no ``env=`` deprecation warning.
"""

import importlib
import warnings

import pytest


ENV_FIELDS = [
    ("GRAPH_DATABASE_PROVIDER", "graph_database_provider", "neo4j", "neo4j"),
    ("KUZU_NUM_THREADS", "kuzu_num_threads", "7", 7),
    ("KUZU_BUFFER_POOL_SIZE", "kuzu_buffer_pool_size", "12345", 12345),
    ("KUZU_MAX_DB_SIZE", "kuzu_max_db_size", "67890", 67890),
]


@pytest.mark.parametrize("env_name,field_name,raw_value,expected", ENV_FIELDS)
def test_fields_bind_from_environment(monkeypatch, env_name, field_name, raw_value, expected):
    """Each field still reads its value from the upper-cased env var by name."""
    monkeypatch.setenv(env_name, raw_value)

    from cognee.infrastructure.databases.graph.config import GraphConfig

    config = GraphConfig()

    # graph_database_provider is lower-cased by the model validator.
    actual = getattr(config, field_name)
    if isinstance(expected, str):
        assert actual == expected.lower()
    else:
        assert actual == expected


def test_no_field_env_deprecation_warning_on_import():
    """Reloading the config module must not emit a Field(env=) deprecation warning."""
    import cognee.infrastructure.databases.graph.config as config_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(config_module)

    offending = [w for w in caught if "extra keys: 'env'" in str(w.message).lower()]
    assert not offending, (
        f"unexpected env= deprecation warning(s): {[str(w.message) for w in offending]}"
    )
