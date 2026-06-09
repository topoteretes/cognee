"""Tests for the refactored get_default_tasks helpers in cognify.py."""

from unittest.mock import patch, MagicMock

import pytest

from cognee.api.v1.cognify.cognify import (
    _resolve_ontology_config,
    _resolve_chunks_per_batch,
    _build_extract_tasks,
    get_default_tasks,
    get_temporal_tasks,
)


# ---------------------------------------------------------------------------
# _resolve_ontology_config
# ---------------------------------------------------------------------------


class TestResolveOntologyConfig:
    @patch("cognee.api.v1.cognify.cognify.get_ontology_env_config")
    @patch("cognee.api.v1.cognify.cognify.get_ontology_resolver_from_env")
    def test_uses_env_resolver_when_all_fields_present(
        self, mock_resolver_from_env, mock_env_config
    ):
        cfg = MagicMock()
        cfg.ontology_file_path = "/some/path.owl"
        cfg.ontology_resolver = "rdflib"
        cfg.matching_strategy = "fuzzy"
        cfg.to_dict.return_value = {"ontology_file_path": "/some/path.owl"}
        mock_env_config.return_value = cfg
        mock_resolver_from_env.return_value = "env_resolver"

        result = _resolve_ontology_config()

        assert result == {"ontology_config": {"ontology_resolver": "env_resolver"}}
        mock_resolver_from_env.assert_called_once()

    @patch("cognee.api.v1.cognify.cognify.get_ontology_env_config")
    @patch("cognee.api.v1.cognify.cognify.get_default_ontology_resolver")
    def test_uses_default_resolver_when_env_fields_missing(
        self, mock_default_resolver, mock_env_config
    ):
        cfg = MagicMock()
        cfg.ontology_file_path = None
        cfg.ontology_resolver = None
        cfg.matching_strategy = None
        mock_env_config.return_value = cfg
        mock_default_resolver.return_value = "default_resolver"

        result = _resolve_ontology_config()

        assert result == {"ontology_config": {"ontology_resolver": "default_resolver"}}


# ---------------------------------------------------------------------------
# _resolve_chunks_per_batch
# ---------------------------------------------------------------------------


class TestResolveChunksPerBatch:
    def test_explicit_value_takes_precedence(self):
        assert _resolve_chunks_per_batch(42) == 42

    @patch("cognee.api.v1.cognify.cognify.get_cognify_config")
    def test_falls_back_to_cognify_config(self, mock_config):
        mock_config.return_value.chunks_per_batch = 77
        assert _resolve_chunks_per_batch(None) == 77

    @patch("cognee.api.v1.cognify.cognify.get_cognify_config")
    def test_falls_back_to_default_when_config_is_none(self, mock_config):
        mock_config.return_value.chunks_per_batch = None
        assert _resolve_chunks_per_batch(None, default=50) == 50


# ---------------------------------------------------------------------------
# _build_extract_tasks
# ---------------------------------------------------------------------------


class TestBuildExtractTasks:
    def test_returns_two_tasks(self):
        tasks = _build_extract_tasks()
        assert len(tasks) == 2

    def test_first_task_is_classify(self):
        from cognee.tasks.documents import classify_documents

        tasks = _build_extract_tasks()
        assert tasks[0].executable is classify_documents

    def test_second_task_is_chunk_extraction(self):
        from cognee.tasks.documents import extract_chunks_from_documents

        tasks = _build_extract_tasks()
        assert tasks[1].executable is extract_chunks_from_documents


# ---------------------------------------------------------------------------
# get_default_tasks
# ---------------------------------------------------------------------------


class TestGetDefaultTasks:
    @pytest.mark.asyncio
    @patch("cognee.api.v1.cognify.cognify._resolve_ontology_config")
    @patch("cognee.api.v1.cognify.cognify.get_cognify_config")
    async def test_returns_five_tasks(self, mock_config, mock_ontology):
        mock_config.return_value.triplet_embedding = False
        mock_config.return_value.chunks_per_batch = 10
        mock_ontology.return_value = {"ontology_config": {"ontology_resolver": "mock"}}

        tasks = await get_default_tasks()

        assert len(tasks) == 5

    @pytest.mark.asyncio
    @patch("cognee.api.v1.cognify.cognify._resolve_ontology_config")
    @patch("cognee.api.v1.cognify.cognify.get_cognify_config")
    async def test_does_not_call_resolve_when_config_provided(self, mock_config, mock_ontology):
        mock_config.return_value.triplet_embedding = False
        mock_config.return_value.chunks_per_batch = 10
        custom_config = {"ontology_config": {"ontology_resolver": "custom"}}

        await get_default_tasks(config=custom_config)

        mock_ontology.assert_not_called()


# ---------------------------------------------------------------------------
# get_temporal_tasks
# ---------------------------------------------------------------------------


class TestGetTemporalTasks:
    @pytest.mark.asyncio
    @patch("cognee.api.v1.cognify.cognify.get_cognify_config")
    async def test_returns_five_tasks(self, mock_config):
        mock_config.return_value.chunks_per_batch = 10

        tasks = await get_temporal_tasks()

        assert len(tasks) == 5
