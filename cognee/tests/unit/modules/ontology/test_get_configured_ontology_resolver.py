from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.ontology.exceptions import EmptyOntologyInStrictModeError
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_configured_ontology_mode,
    get_configured_ontology_resolver,
)
from cognee.modules.ontology.ontology_env_config import OntologyEnvConfig


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


@patch("cognee.modules.ontology.get_default_ontology_resolver.RDFLibOntologyResolver")
@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_none_config_splats_real_to_dict_into_real_factory(mock_env_config, mock_resolver_class):
    """The real env config's to_dict() must match the real factory's signature.

    Regression test: a key added to to_dict() that the factory does not accept
    breaks every env-configured cognify run with a TypeError. Only the resolver
    class itself is stubbed here — config and factory bodies run for real.
    """
    mock_env_config.return_value = OntologyEnvConfig(
        _env_file=None,
        ontology_resolver="rdflib",
        matching_strategy="fuzzy",
        ontology_file_path="ontology.owl",
    )
    resolver = _mock_resolver()
    resolver.lookup = {"classes": {"car": object()}, "individuals": {}}
    mock_resolver_class.return_value = resolver

    assert get_configured_ontology_resolver(None) is resolver


@patch("cognee.modules.ontology.get_default_ontology_resolver.RDFLibOntologyResolver")
@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_env_strict_mode_with_empty_ontology_fails_fast(mock_env_config, mock_resolver_class):
    mock_env_config.return_value = OntologyEnvConfig(
        _env_file=None,
        ontology_file_path="/typo/does_not_exist.owl",
        ontology_mode="strict",
    )
    empty_resolver = _mock_resolver()
    empty_resolver.lookup = {"classes": {}, "individuals": {}}
    mock_resolver_class.return_value = empty_resolver

    with pytest.raises(EmptyOntologyInStrictModeError):
        get_configured_ontology_resolver(None)


@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_ontology_mode_from_config_overrides_env(mock_env_config):
    mock_env_config.return_value.ontology_mode = "annotate"

    config = {"ontology_config": {"ontology_resolver": _mock_resolver(), "ontology_mode": "strict"}}
    assert get_configured_ontology_mode(config) == "strict"


@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_ontology_mode_falls_back_to_env(mock_env_config):
    mock_env_config.return_value.ontology_mode = "strict"

    assert get_configured_ontology_mode(None) == "strict"
    assert (
        get_configured_ontology_mode({"ontology_config": {"ontology_resolver": None}}) == "strict"
    )


@patch("cognee.modules.ontology.get_default_ontology_resolver.get_ontology_env_config")
def test_invalid_per_call_ontology_mode_falls_back_to_default(mock_env_config):
    mock_env_config.return_value.ontology_mode = "annotate"

    config = {"ontology_config": {"ontology_resolver": None, "ontology_mode": "bogus"}}
    assert get_configured_ontology_mode(config) == "annotate"
