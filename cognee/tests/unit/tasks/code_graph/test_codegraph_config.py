"""The one translation of the removed ``content_type="code"`` every surface shares."""

import pytest

from cognee.tasks.code_graph.config import (
    LEGACY_CODE_CONTENT_TYPE_CONFIG,
    is_legacy_code_content_type,
    validate_codegraph_config,
    with_legacy_code_content_type,
)


def test_legacy_value_means_code_graph_only_declared_repositories():
    assert with_legacy_code_content_type(None) == LEGACY_CODE_CONTENT_TYPE_CONFIG
    assert with_legacy_code_content_type({}) == {
        "include_documents": False,
        "treat_as_repository": True,
    }


def test_explicit_keys_win_over_the_legacy_defaults():
    config = with_legacy_code_content_type({"include_documents": True, "index_vectors": True})

    assert config == {
        "include_documents": True,
        "treat_as_repository": True,
        "index_vectors": True,
    }
    assert validate_codegraph_config(config) == config


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"treat_as_repository": True}, True),
        ({"treat_as_repository": True, "include_documents": False}, True),
        ({"treat_as_repository": True, "include_documents": True}, False),
        ({"include_documents": False}, False),
        (LEGACY_CODE_CONTENT_TYPE_CONFIG, True),
    ],
)
def test_wire_form_matches_exactly_what_the_server_reads_back(config, expected):
    assert is_legacy_code_content_type(config) is expected
    if expected:
        # Round trip: what travels as content_type='code' is read back as itself.
        assert with_legacy_code_content_type(config) == {
            **LEGACY_CODE_CONTENT_TYPE_CONFIG,
            **config,
        }
