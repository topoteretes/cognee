"""Regression tests for the opt-in canonicalization splice in get_default_tasks
(issue #3629, commit 5). The flag-OFF case is the critical invariant: the task
list must be element-for-element identical to the pre-canonicalization pipeline.
No real API keys are required (get_cognify_config is patched; config + chunk_size
are passed so no ontology/LLM setup runs)."""

import sys
from unittest.mock import patch

import pytest

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.cognify.config import CognifyConfig

# `from cognee.api.v1.cognify import cognify` would resolve to the re-exported
# cognify FUNCTION; grab the actual module object for patching get_cognify_config.
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]

# The canonical pre-canonicalization task order.
_BASE_SEQUENCE = [
    "classify_documents",
    "extract_chunks_from_documents",
    "extract_graph_and_summarize",
    "add_data_points",
    "extract_dlt_fk_edges",
]


async def _task_name_sequence(flag_value):
    """Build the default task list with the flag set, returning executable names."""
    config = CognifyConfig(entity_canonicalization=flag_value)
    with patch.object(cognify_module, "get_cognify_config", return_value=config):
        tasks = await get_default_tasks(
            # Pass a non-None config so the ontology-env branch is skipped, and an
            # explicit chunk_size so get_max_chunk_tokens() is never called.
            config={"ontology_config": {"ontology_resolver": None}},
            chunk_size=1024,
        )
    return [task.executable.__name__ for task in tasks]


@pytest.mark.asyncio
async def test_flag_off_task_list_is_byte_identical():
    """(f-OFF) With the flag off, the task list matches today's pipeline exactly."""
    sequence = await _task_name_sequence(False)
    assert sequence == _BASE_SEQUENCE
    assert "canonicalize_entities" not in sequence


@pytest.mark.asyncio
async def test_flag_on_inserts_canonicalize_between_summarize_and_store():
    """(f-ON) With the flag on, canonicalize_entities is spliced at index 3."""
    sequence = await _task_name_sequence(True)
    assert sequence == [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize",
        "canonicalize_entities",
        "add_data_points",
        "extract_dlt_fk_edges",
    ]
    # It sits strictly between extraction/summarization and storage.
    assert (
        sequence.index("canonicalize_entities") == sequence.index("extract_graph_and_summarize") + 1
    )
    assert sequence.index("canonicalize_entities") == sequence.index("add_data_points") - 1
