"""Extractor resolution split into a pure half (SDK-775) and the keyless
errors' HTTP status."""

import importlib.util

import pytest
from fastapi import status

from cognee.infrastructure.databases.vector.embeddings.config import (
    KeylessEmbedderNotInstalledError,
)
from cognee.modules.cognify import config as cognify_config

_real_find_spec = importlib.util.find_spec


def _without_gliner2(name, *args, **kwargs):
    return None if name == "gliner2" else _real_find_spec(name, *args, **kwargs)


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


def test_resolve_extractor_name_decides_auto_without_checking_the_install(monkeypatch):
    monkeypatch.setattr(cognify_config.importlib.util, "find_spec", _without_gliner2)

    assert cognify_config.resolve_extractor_name(None, _config(), llm_configured=False) == (
        cognify_config.GLINER_DEMO_EXTRACTOR
    )
    assert cognify_config.resolve_extractor_name(None, _config(), llm_configured=True) == (
        cognify_config.LLM_EXTRACTOR
    )
    with pytest.raises(cognify_config.KeylessExtractorNotInstalledError):
        cognify_config.resolve_extractor(None, _config(), llm_configured=False)


def test_explicit_gliner_without_the_package_is_not_the_keyless_error(monkeypatch):
    """Only the keyless default raises KeylessExtractorNotInstalledError; a pinned
    ``gliner`` fails later with GlinerNotInstalledError, as before the split."""
    monkeypatch.setattr(cognify_config.importlib.util, "find_spec", _without_gliner2)

    assert cognify_config.resolve_extractor("gliner", _config()) == (
        cognify_config.GLINER_DEMO_EXTRACTOR
    )


def test_unknown_values_pass_through_the_pure_half_and_fail_the_checked_one():
    assert cognify_config.resolve_extractor_name("bogus", _config()) == "bogus"
    with pytest.raises(ValueError):
        cognify_config.resolve_extractor("bogus", _config())
