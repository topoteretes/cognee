"""Verify dataset-queue slot accounting under max_concurrent=1.

Runs the SDK calls inline (no isolation) so slots leaked by a call stay
visible to the next assertion. Expected to fail on the calls that hold a
slot past their own lifetime — fix those call sites and the test goes
green.

Covers: add/cognify, get_formatted_graph_data, datasets.delete_data,
datasets.empty_dataset.
"""

import os
import pathlib
import pytest
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.infrastructure.databases.dataset_queue import dataset_queue
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph.methods.get_formatted_graph_data import (
    get_formatted_graph_data,
)
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent


GET_DATASET_QUEUE_SETTINGS = (
    "cognee.infrastructure.databases.dataset_queue.queue.get_dataset_queue_settings"
)


def _mock_llm_output(text_input: str, system_prompt: str, response_model):
    if text_input == "test":
        return "test"
    if response_model == SummarizedContent:
        return SummarizedContent(summary="summary", description="summary")
    if response_model == KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[
                Node(
                    id="John",
                    name="John",
                    type="Person",
                    description="John is a person",
                    label="John",
                ),
                Node(
                    id="Apple",
                    name="Apple",
                    type="Company",
                    description="Apple is a company",
                    label="Apple",
                ),
            ],
            edges=[
                Edge(
                    source_node_id="John",
                    target_node_id="Apple",
                    relationship_name="works_for",
                ),
            ],
        )


def _assert_queue_empty(queue, label: str):
    """Assert no task holds any queue slot and the semaphore is full."""
    live = {tid: list(slots) for tid, slots in queue._task_slots.items() if slots}
    assert not live, f"After {label}: queue leaked slots — _task_slots still holds {live}"
    assert queue._semaphore._value == queue._max_concurrent, (
        f"After {label}: semaphore at {queue._semaphore._value}, "
        f"expected {queue._max_concurrent} (slot was never released)"
    )


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def test_max_concurrent_one_does_not_accumulate_slots(mock_llm):
    mock_llm.side_effect = _mock_llm_output

    data_dir = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_queue_max_concurrent_one"
    )
    system_dir = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_queue_max_concurrent_one"
    )
    cognee.config.data_root_directory(data_dir)
    cognee.config.system_root_directory(system_dir)

    with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
        mock_settings.return_value.enabled = True
        mock_settings.return_value.max_concurrent = 1

        # Reset singleton so the patched settings take effect.
        dataset_queue._instance = None
        queue = dataset_queue()
        assert queue._enabled is True
        assert queue._max_concurrent == 1

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        user = await get_default_user()

        # ---- add --------------------------------------------------------
        add_result = await cognee.add("John works for Apple.")
        data_id = add_result.data_ingestion_info[0]["data_id"]
        _assert_queue_empty(queue, "add")

        # ---- cognify ----------------------------------------------------
        cognify_result = await cognee.cognify()
        dataset_id = list(cognify_result.keys())[0]
        _assert_queue_empty(queue, "cognify")

        # ---- get_formatted_graph_data -----------------------------------
        graph_data = await get_formatted_graph_data(dataset_id, user)
        assert "nodes" in graph_data and "edges" in graph_data
        _assert_queue_empty(queue, "get_formatted_graph_data")

        # ---- delete_data ------------------------------------------------
        await datasets.delete_data(dataset_id, data_id, user)
        _assert_queue_empty(queue, "delete_data")

        # ---- empty_dataset ----------------------------------------------
        await datasets.empty_dataset(dataset_id, user)
        _assert_queue_empty(queue, "empty_dataset")
