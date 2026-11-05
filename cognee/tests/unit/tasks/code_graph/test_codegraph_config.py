"""Validation of the per-call ``codegraph_config`` options."""

import pytest

from cognee.tasks.code_graph.config import requests_codegraph_options, validate_codegraph_config


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"index_vectors": False}, True),
        ({"include_documents": True}, True),
        ({"treat_as_repository": True}, True),
        ({"include_documents": False}, True),
        ({"index_vectors": True}, True),
        ({"repo_credentials": "tok"}, True),
    ],
)
def test_any_option_counts_as_requested(config, expected):
    assert requests_codegraph_options(config) is expected


@pytest.mark.parametrize(
    ("key", "value"),
    [("index_vectors", "false"), ("include_documents", 0), ("treat_as_repository", "true")],
)
def test_flags_must_be_real_bools(key, value):
    # "false" is truthy: accepting it would invert the caller's intent.
    with pytest.raises(TypeError, match=f"codegraph_config\\['{key}'\\] must be a bool"):
        validate_codegraph_config({key: value})


@pytest.mark.parametrize("value", ["", 123, b"tok"])
def test_repo_credentials_must_be_a_non_empty_str(value):
    with pytest.raises(TypeError, match="repo_credentials'\\] must be a non-empty str"):
        validate_codegraph_config({"repo_credentials": value})


def test_unknown_keys_are_rejected_and_named():
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: batch, index_vector"):
        validate_codegraph_config({"index_vector": True, "batch": 1})


def test_a_non_dict_is_rejected():
    with pytest.raises(TypeError, match="must be a dict"):
        validate_codegraph_config(["index_vectors"])  # type: ignore[arg-type]


def test_none_values_count_as_absent():
    assert validate_codegraph_config({"index_vectors": None, "include_documents": False}) == {
        "include_documents": False
    }
