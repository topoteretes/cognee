"""GraphConfig must not emit Pydantic deprecation warnings, and must keep binding
its values from the environment.

``Field(..., env=...)`` is a Pydantic V1 spelling. Under V2 the extra keyword is
ignored and pydantic-settings binds from ``<field_name>.upper()`` instead, so the
call only produced a ``PydanticDeprecatedSince20`` warning at import time without
changing how any value was read. See #3702.
"""

import importlib
import os
import warnings
from unittest.mock import patch

import pytest

from cognee.infrastructure.databases.graph.config import GraphConfig

# field name -> environment variable it must still bind from.
# Expected values are the post-coercion types: pydantic parses env strings into
# the field's annotated type, so the int fields never surface as str.
ENV_BOUND_FIELDS = [
    ("graph_database_provider", "GRAPH_DATABASE_PROVIDER", "neo4j"),
    ("kuzu_num_threads", "KUZU_NUM_THREADS", 7),
    ("kuzu_buffer_pool_size", "KUZU_BUFFER_POOL_SIZE", 123456),
    ("kuzu_max_db_size", "KUZU_MAX_DB_SIZE", 987654),
]


@pytest.mark.parametrize(("field", "env_var", "value"), ENV_BOUND_FIELDS)
def test_graph_config_field_values_still_bind_from_environment(field, env_var, value):
    """Removing the ignored ``env=`` kwarg must not change what the env var does."""
    with patch.dict(os.environ, {env_var: str(value)}):
        config = GraphConfig()

    actual = getattr(config, field)

    assert actual == value, (
        f"{field} did not pick up {env_var}; pydantic-settings binds by field name"
    )
    assert isinstance(actual, type(value)), (
        f"{field} should be coerced to {type(value).__name__}, got {type(actual).__name__}"
    )


def test_graph_config_defaults_apply_without_environment():
    from cognee.infrastructure.databases.graph.kuzu.adapter import (
        DEFAULT_KUZU_BUFFER_POOL_SIZE,
        DEFAULT_KUZU_MAX_DB_SIZE,
    )

    with patch.dict(os.environ, {}, clear=True):
        config = GraphConfig()

    assert config.graph_database_provider == "ladybug"
    assert config.kuzu_num_threads == 0
    assert config.kuzu_buffer_pool_size == DEFAULT_KUZU_BUFFER_POOL_SIZE
    assert config.kuzu_max_db_size == DEFAULT_KUZU_MAX_DB_SIZE


def test_graph_config_emits_no_pydantic_field_deprecation_warning():
    """Importing the module must not raise PydanticDeprecatedSince20 about env=.

    The warning fires while the class body executes, so the module has to be
    re-imported with the warnings filter inside the recording block.
    """
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module("cognee.infrastructure.databases.graph.config"))

    field_env_warnings = [
        w
        for w in recorded
        if issubclass(w.category, DeprecationWarning)
        and "env" in str(w.message)
        and "Field" in str(w.message)
    ]

    assert not field_env_warnings, "unexpected deprecation warning(s): " + "; ".join(
        f"{w.filename}:{w.lineno} {w.message}" for w in field_env_warnings
    )


def test_no_graph_config_field_declares_a_deprecated_env_kwarg():
    """Guard the source itself, so a re-added ``env=`` fails here even if a future
    Pydantic stops warning about it."""
    import inspect

    from cognee.infrastructure.databases.graph import config as config_module

    source = inspect.getsource(config_module)
    offending = [
        line.strip()
        for line in source.splitlines()
        if "Field(" in line and "env=" in line and not line.strip().startswith("#")
    ]

    assert not offending, "Field(env=...) reintroduced in: " + "; ".join(offending)
