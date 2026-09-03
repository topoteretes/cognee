"""temporal_cognify is a flag on the standard task list, not a parallel pipeline (SDK-80)."""

import importlib
from unittest.mock import patch

import pytest

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.cognify.config import CognifyConfig

# The package re-exports the function under the same name; patch the module object.
cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")


async def _tasks(temporal_cognify, **config_overrides):
    config = CognifyConfig(**config_overrides)
    with patch.object(cognify_module, "get_cognify_config", return_value=config):
        return await get_default_tasks(
            config={"ontology_config": {"ontology_resolver": None}},
            chunk_size=1024,
            temporal_cognify=temporal_cognify,
        )


def _by_name(tasks):
    return {task.executable.__name__: task for task in tasks}


@pytest.mark.asyncio
async def test_temporal_flag_turns_on_the_event_lane_only():
    plain = await _tasks(temporal_cognify=False)
    temporal = await _tasks(temporal_cognify=True)

    assert [t.executable.__name__ for t in plain] == [t.executable.__name__ for t in temporal]
    assert (
        _by_name(plain)["extract_graph_and_summarize"].default_params["kwargs"]["extract_events"]
        is False
    )
    assert (
        _by_name(temporal)["extract_graph_and_summarize"].default_params["kwargs"]["extract_events"]
        is True
    )


@pytest.mark.asyncio
async def test_llm_event_entities_is_opt_in_and_temporal_only():
    off = await _tasks(temporal_cognify=True)
    assert "extract_knowledge_graph_from_events" not in _by_name(off)

    on = await _tasks(temporal_cognify=True, temporal_llm_event_entities=True)
    names = [t.executable.__name__ for t in on]
    assert names.index("extract_knowledge_graph_from_events") == (
        names.index("extract_graph_and_summarize") + 1
    )

    not_temporal = await _tasks(temporal_cognify=False, temporal_llm_event_entities=True)
    assert "extract_knowledge_graph_from_events" not in _by_name(not_temporal)


def test_get_temporal_tasks_is_gone():
    assert not hasattr(cognify_module, "get_temporal_tasks")
