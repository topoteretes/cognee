from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.ontology.get_default_ontology_resolver import get_configured_ontology_resolver


def _mock_resolver():
    return MagicMock()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"ontology_config": None},
        {"ontology_config": {}},
        {"ontology_config": {"ontology_resolver": None}},
    ],
)
def test_explicit_empty_config_shapes_return_none(config):
    assert get_configured_ontology_resolver(config) is None


def test_explicit_resolver_is_returned_as_is():
    resolver = _mock_resolver()
    config = {"ontology_config": {"ontology_resolver": resolver}}
    assert get_configured_ontology_resolver(config) is resolver


@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_resolver_from_env")
def test_none_config_uses_env_when_configured(mock_from_env, mock_env_config):
    mock_env_config.return_value.ontology_file_path = "ontology.owl"
    mock_env_config.return_value.ontology_resolver = "rdflib"
    mock_env_config.return_value.matching_strategy = "fuzzy"
    mock_env_config.return_value.to_dict.return_value = {
        "ontology_resolver": "rdflib",
        "matching_strategy": "fuzzy",
        "ontology_file_path": "ontology.owl",
    }
    resolver = _mock_resolver()
    mock_from_env.return_value = resolver

    assert get_configured_ontology_resolver(None) is resolver
    mock_from_env.assert_called_once()


@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_none_config_without_env_returns_none(mock_env_config):
    mock_env_config.return_value.ontology_file_path = ""
    mock_env_config.return_value.ontology_resolver = "rdflib"
    mock_env_config.return_value.matching_strategy = "fuzzy"

    assert get_configured_ontology_resolver(None) is None
