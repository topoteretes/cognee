"""codegraph_config validation (SDK-793)."""

import pytest

from cognee.tasks.code_graph.config import CODEGRAPH_CONFIG_KEYS, validate_codegraph_config


def test_keys():
    assert CODEGRAPH_CONFIG_KEYS == {
        "index_vectors",
        "include_documents",
        "repo_credentials",
        "treat_as_repository",
    }


def test_none_and_none_values_count_as_absent():
    assert validate_codegraph_config(None) == {}
    assert validate_codegraph_config({"index_vectors": None, "include_documents": False}) == {
        "include_documents": False
    }


def test_a_valid_config_is_returned_as_a_new_dict():
    config = {"treat_as_repository": True, "repo_credentials": "token"}
    validated = validate_codegraph_config(config)
    assert validated == config
    assert validated is not config


@pytest.mark.parametrize(
    "config, error, match",
    [
        ("index_vectors", TypeError, "must be a dict"),
        ({"index_vector": True}, ValueError, "Unknown codegraph_config keys: index_vector"),
        # "false" is truthy: accepting it would invert the flag.
        ({"include_documents": "false"}, TypeError, "must be a bool"),
        ({"treat_as_repository": 1}, TypeError, "must be a bool"),
        ({"repo_credentials": ""}, TypeError, "non-empty str"),
        ({"repo_credentials": 42}, TypeError, "non-empty str"),
    ],
)
def test_invalid_configs_are_refused(config, error, match):
    with pytest.raises(error, match=match):
        validate_codegraph_config(config)
