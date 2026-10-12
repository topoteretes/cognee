"""Removed ingestion options fail early or leave the pipeline unchanged."""

import importlib
from unittest.mock import patch

import pytest

from cognee.modules.cognify.config import CognifyConfig, get_cognify_config

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
config_module = importlib.import_module("cognee.modules.cognify.config")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")

_BASE_SEQUENCE = [
    "classify_documents",
    "extract_chunks_from_documents",
    "extract_graph_and_summarize",
    "add_data_points",
]


async def _task_name_sequence(config):
    with patch.object(cognify_module, "get_cognify_config", return_value=config):
        tasks = await cognify_module.get_default_tasks(
            config={"ontology_config": {"ontology_resolver": None}},
            chunk_size=1024,
        )
    return [task.executable.__name__ for task in tasks]


@pytest.mark.asyncio
async def test_removed_keyword_raises_before_setup():
    with (
        patch.object(
            cognify_module, "get_cognify_config", side_effect=AssertionError("config read")
        ),
        patch.object(
            serve_state_module, "get_remote_client", side_effect=AssertionError("remote call")
        ),
        patch.object(
            cognify_module, "ensure_extractor_runtime", side_effect=AssertionError("install")
        ),
        pytest.raises(TypeError, match="no longer accepts functional_relationships"),
    ):
        await cognify_module.cognify(functional_relationships={"has_ceo"})


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["true", "false", ""])
async def test_removed_setting_warns_once_per_cache_fill_and_leaves_tasks_unchanged(
    monkeypatch, value
):
    monkeypatch.delenv("CONTRADICTION_DETECTION", raising=False)
    monkeypatch.setenv("PROVENANCE_TRACKING", "false")
    unset_tasks = await _task_name_sequence(CognifyConfig())
    monkeypatch.setenv("CONTRADICTION_DETECTION", value)
    get_cognify_config.cache_clear()
    try:
        with patch.object(config_module.logger, "warning") as warning:
            config = get_cognify_config()
            assert get_cognify_config() is config
            warning.assert_called_once()
            assert "improve(review_conflicts=True)" in warning.call_args.args[0]
            assert "IMPROVE_REVIEW_CONFLICTS" in warning.call_args.args[0]
            assert await _task_name_sequence(config) == unset_tasks == _BASE_SEQUENCE
            get_cognify_config.cache_clear()
            get_cognify_config()
            assert warning.call_count == 2
    finally:
        get_cognify_config.cache_clear()
