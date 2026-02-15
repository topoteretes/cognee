"""Test for vector database configuration validation."""

import pytest

from cognee.api.v1.config.config import config
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError


def test_set_vector_db_config_invalid_attribute_raises():
    """Ensure invalid vector DB config attributes raise an error."""
    with pytest.raises(InvalidConfigAttributeError):
        config.set_vector_db_config({"invalid_attribute": "should_fail"})
