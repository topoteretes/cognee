"""Wiring tests for opt-in contradiction detection (issue #3699).

Contradiction detection is toggled by the ``contradiction_detection`` CognifyConfig
flag rather than a cognify() argument, so what needs covering is the config surface,
the splice in get_default_tasks, and the fact that remember() — which builds its
graph through cognify() — inherits the flag. The flag-OFF case is the critical
invariant: the task list must be element-for-element identical to the pre-detection
pipeline. No real API keys are required (get_cognify_config is patched; config +
chunk_size are passed so no ontology/LLM setup runs).
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.api.v1.remember.remember import remember
from cognee.modules.cognify.config import CognifyConfig
from cognee.tasks.graph.models import Contradiction, ContradictionList

# `from cognee.api.v1.cognify import cognify` would resolve to the re-exported
# cognify FUNCTION; grab the actual module objects for patching.
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]
remember_module = sys.modules["cognee.api.v1.remember.remember"]

# The canonical pre-detection task order.
_BASE_SEQUENCE = [
    "classify_documents",
    "extract_chunks_from_documents",
    "extract_graph_and_summarize",
    "add_data_points",
    "extract_dlt_fk_edges",
]


async def _task_name_sequence(flag_value):
    """Build the default task list with the flag set, returning executable names."""
    config = CognifyConfig(contradiction_detection=flag_value)
    with patch.object(cognify_module, "get_cognify_config", return_value=config):
        tasks = await get_default_tasks(
            # Pass a non-None config so the ontology-env branch is skipped, and an
            # explicit chunk_size so get_max_chunk_tokens() is never called.
            config={"ontology_config": {"ontology_resolver": None}},
            chunk_size=1024,
        )
    return [task.executable.__name__ for task in tasks]


class TestContradictionDetectionConfig:
    """CognifyConfig gains the #3699 flag + 2 tunables, default OFF."""

    def test_contradiction_defaults(self):
        """Flag defaults False and each tunable has the planned default."""
        with patch.dict(os.environ, {}, clear=True):
            config = CognifyConfig()
            assert config.contradiction_detection is False
            assert config.contradiction_confidence_threshold == 0.5
            assert config.contradiction_max_facts == 500

    def test_to_dict_includes_contradiction_keys(self):
        """to_dict() surfaces all three new keys with their values."""
        with patch.dict(os.environ, {}, clear=True):
            config_dict = CognifyConfig().to_dict()
            assert config_dict["contradiction_detection"] is False
            assert config_dict["contradiction_confidence_threshold"] == 0.5
            assert config_dict["contradiction_max_facts"] == 500

    def test_flag_is_env_overridable(self):
        """CONTRADICTION_DETECTION env var flips the flag on (opt-in path)."""
        with patch.dict(os.environ, {"CONTRADICTION_DETECTION": "true"}, clear=True):
            assert CognifyConfig().contradiction_detection is True


class TestGetDefaultTasksSplice:
    """The task is spliced into the cognify pipeline by the config flag alone."""

    @pytest.mark.asyncio
    async def test_flag_off_task_list_is_unchanged(self):
        """(f-OFF) With the flag off, the task list matches today's pipeline exactly."""
        sequence = await _task_name_sequence(False)
        assert sequence == _BASE_SEQUENCE
        assert "detect_contradictions" not in sequence

    @pytest.mark.asyncio
    async def test_flag_on_appends_detection_after_storage(self):
        """(f-ON) With the flag on, detect_contradictions runs last, after storage."""
        sequence = await _task_name_sequence(True)
        assert sequence == _BASE_SEQUENCE + ["detect_contradictions"]
        # Comparing new facts against stored ones only works once they are persisted.
        assert sequence.index("detect_contradictions") > sequence.index("add_data_points")

    @pytest.mark.asyncio
    async def test_cognify_has_no_detect_contradictions_argument(self):
        """The flag lives in config only — no cognify()/get_default_tasks() kwarg."""
        from inspect import signature

        from cognee.api.v1.cognify.cognify import cognify

        assert "detect_contradictions" not in signature(cognify).parameters
        assert "detect_contradictions" not in signature(get_default_tasks).parameters


class TestRememberInheritsTheFlag:
    """remember() builds its graph through cognify(), so the flag reaches it too.

    This is the behaviour the config flag buys: remember() routes only a fixed
    allow-list of kwargs to cognify() (see _COGNIFY_ONLY), so a cognify() argument
    would have been rejected with "Unexpected keyword arguments" and the feature
    would have been unreachable from the primary 1.0 API.
    """

    async def _remember_task_sequence(self, flag_value):
        """Run remember() far enough to capture the pipeline it hands the executor."""
        captured = {}

        def _fake_executor(run_in_background=False):
            async def _run(**executor_kwargs):
                captured["tasks"] = [task.executable.__name__ for task in executor_kwargs["tasks"]]
                return {}

            return _run

        config = CognifyConfig(contradiction_detection=flag_value)
        with (
            patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}),
            patch.object(cognify_module, "get_cognify_config", return_value=config),
            patch.object(cognify_module, "get_pipeline_executor", _fake_executor),
            patch("cognee.modules.migrations.startup.run_migrations_and_block", new=AsyncMock()),
            patch("cognee.api.v1.serve.state.get_remote_client", return_value=None),
            patch("cognee.modules.engine.operations.setup.setup", new=AsyncMock()),
            patch("cognee.api.v1.add.add", new=AsyncMock()),
            patch(
                "cognee.modules.users.methods.get_default_user",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                remember_module,
                "resolve_authorized_user_datasets",
                new=AsyncMock(return_value=(object(), [])),
            ),
        ):
            await remember(
                "Alice was born in 1985.",
                self_improvement=False,
                chunk_size=1024,
                config={"ontology_config": {"ontology_resolver": None}},
            )
        return captured.get("tasks")

    @pytest.mark.asyncio
    async def test_remember_omits_detection_when_flag_off(self):
        assert await self._remember_task_sequence(False) == _BASE_SEQUENCE

    @pytest.mark.asyncio
    async def test_remember_runs_detection_when_flag_on(self):
        sequence = await self._remember_task_sequence(True)
        assert sequence == _BASE_SEQUENCE + ["detect_contradictions"]


class TestContradictionModels:
    """The structured detection response validates well-formed payloads and rejects bad ones."""

    def test_contradiction_list_validates_canned_payload(self):
        payload = {
            "contradictions": [
                {
                    "first_fact_id": "F0",
                    "second_fact_id": "F1",
                    "reason": "A person has a single birth year.",
                    "confidence": 0.95,
                }
            ]
        }
        result = ContradictionList.model_validate(payload)
        assert len(result.contradictions) == 1
        contradiction = result.contradictions[0]
        assert isinstance(contradiction, Contradiction)
        assert contradiction.first_fact_id == "F0"
        assert contradiction.second_fact_id == "F1"
        assert contradiction.confidence == 0.95

    def test_contradiction_list_defaults_to_empty(self):
        """A "no contradictions" reply need not carry the key at all."""
        assert ContradictionList.model_validate({}).contradictions == []

    def test_contradiction_rejects_out_of_range_confidence(self):
        """Confidence is bounded to [0.0, 1.0], so the threshold check is meaningful."""
        with pytest.raises(ValidationError):
            Contradiction.model_validate(
                {
                    "first_fact_id": "F0",
                    "second_fact_id": "F1",
                    "reason": "x",
                    "confidence": 1.5,
                }
            )

    def test_contradiction_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            Contradiction.model_validate(
                {
                    "first_fact_id": "F0",
                    "second_fact_id": "F1",
                    # reason missing
                    "confidence": 0.9,
                }
            )
