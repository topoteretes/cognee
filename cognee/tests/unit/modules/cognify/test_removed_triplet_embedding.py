"""TRIPLET_EMBEDDING was removed (SDK-1115): a stale setting warns once and writes nothing."""

import importlib
import inspect
from unittest.mock import patch

import pytest

from cognee.modules.cognify.config import CognifyConfig, get_cognify_config
from cognee.tasks.storage.add_data_points import add_data_points

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
config_module = importlib.import_module("cognee.modules.cognify.config")


@pytest.mark.asyncio
async def test_set_triplet_embedding_warns_once_and_leaves_the_writer_unchanged(monkeypatch):
    monkeypatch.delenv("CONTRADICTION_DETECTION", raising=False)
    monkeypatch.setenv("TRIPLET_EMBEDDING", "true")
    get_cognify_config.cache_clear()
    try:
        with patch.object(config_module.logger, "warning") as warning:
            config = get_cognify_config()
            assert get_cognify_config() is config
            warning.assert_called_once()
            assert "TRIPLET_EMBEDDING is ignored" in warning.call_args.args[0]

        assert not hasattr(config, "triplet_embedding")
        assert "triplet_embedding" not in config.to_dict()
        with patch.object(cognify_module, "get_cognify_config", return_value=config):
            tasks = await cognify_module.get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}}, chunk_size=1024
            )
        [store] = [task for task in tasks if task.executable is add_data_points]
        assert "embed_triplets" not in store.default_params["kwargs"]
    finally:
        get_cognify_config.cache_clear()


def test_unset_triplet_embedding_does_not_warn(monkeypatch):
    monkeypatch.delenv("TRIPLET_EMBEDDING", raising=False)
    monkeypatch.delenv("CONTRADICTION_DETECTION", raising=False)
    get_cognify_config.cache_clear()
    try:
        with patch.object(config_module.logger, "warning") as warning:
            get_cognify_config()
        warning.assert_not_called()
    finally:
        get_cognify_config.cache_clear()


def test_add_data_points_has_no_triplet_writer():
    assert "embed_triplets" not in inspect.signature(add_data_points).parameters
    assert "triplet_embedding" not in CognifyConfig.model_fields
