"""Extractor selection is resolved once, without installing a runtime."""

import pytest

from cognee.modules.cognify import config as cognify_config


def _config(value="auto"):
    return cognify_config.CognifyConfig(graph_extractor=value)


def test_resolve_extractor_decides_auto_without_checking_the_install():
    """Runtime installation is handled separately by ensure_extractor_runtime."""
    assert cognify_config.resolve_extractor(None, _config(), llm_configured=False) == (
        cognify_config.GLINER_DEMO_EXTRACTOR
    )
    assert cognify_config.resolve_extractor(None, _config(), llm_configured=True) == (
        cognify_config.LLM_EXTRACTOR
    )


def test_unknown_values_fail_resolution():
    with pytest.raises(ValueError):
        cognify_config.resolve_extractor("bogus", _config())
