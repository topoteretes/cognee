"""Extractor resolution split into a pure half (SDK-775) and the keyless
errors' HTTP status."""

import pytest
from fastapi import status

from cognee.infrastructure.databases.vector.embeddings.config import (
    KeylessEmbedderNotInstalledError,
)
from cognee.modules.cognify import config as cognify_config


def _config(value="auto"):
    return cognify_config.CognifyConfig(graph_extractor=value)


def test_keyless_extractor_error_is_a_422_with_the_install_hint_as_remediation():
    error = cognify_config.KeylessExtractorNotInstalledError()

    assert error.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert 'pip install "cognee[gliner]"' in error.remediation
    assert 'pip install "cognee[gliner]"' in str(error)


def test_keyless_embedder_error_is_a_422_with_the_install_hint_as_remediation():
    error = KeylessEmbedderNotInstalledError()

    assert error.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "pip install fastembed" in error.remediation
    assert "pip install fastembed" in str(error)


def test_resolve_extractor_name_decides_auto_without_checking_the_install():
    """Both halves only decide; the runtime check is ``ensure_extractor_runtime``'s
    (covered in test_gliner_install), so neither can raise the keyless error."""
    assert cognify_config.resolve_extractor_name(None, _config(), llm_configured=False) == (
        cognify_config.GLINER_DEMO_EXTRACTOR
    )
    assert cognify_config.resolve_extractor_name(None, _config(), llm_configured=True) == (
        cognify_config.LLM_EXTRACTOR
    )
    assert cognify_config.resolve_extractor(None, _config(), llm_configured=False) == (
        cognify_config.GLINER_DEMO_EXTRACTOR
    )


def test_unknown_values_pass_through_the_pure_half_and_fail_the_checked_one():
    assert cognify_config.resolve_extractor_name("bogus", _config()) == "bogus"
    with pytest.raises(ValueError):
        cognify_config.resolve_extractor("bogus", _config())
