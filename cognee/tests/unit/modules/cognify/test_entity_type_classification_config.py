import os
from unittest.mock import patch

from cognee.modules.cognify.config import CognifyConfig


def test_classification_is_off_by_default():
    """It adds an LLM call per batch, so the standard cognify pipeline stays unchanged."""
    with patch.dict(os.environ, {}, clear=True):
        config = CognifyConfig()

    assert config.entity_type_classification is False
    assert config.to_dict()["entity_type_classification"] is False


def test_the_environment_variable_turns_it_on():
    with patch.dict(os.environ, {"ENTITY_TYPE_CLASSIFICATION": "true"}, clear=True):
        assert CognifyConfig().entity_type_classification is True
