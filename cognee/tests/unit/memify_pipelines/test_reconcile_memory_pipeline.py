from unittest.mock import AsyncMock, patch

import pytest

from cognee.memify_pipelines.reconcile_memory import reconcile_memory_pipeline


@pytest.mark.asyncio
async def test_reconcile_memory_pipeline_wires_memify_tasks():
    """The wrapper forwards its knobs onto the enrichment Task and calls memify with data=[{}] / dataset.

    reconcile_memory's tuning params are keyword-only and Task() stores them without validating against the
    target signature — so a kwarg typo (e.g. ``confidence_treshold=``) or a future signature drift would only
    surface during a live memify run. This pins the wiring with no graph / no LLM / no DB lock.
    """
    with patch("cognee.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock:
        result = await reconcile_memory_pipeline(
            confidence_threshold=0.8,
            prefer="feedback",
            demote_factor=0.3,
            max_pairs=7,
            protect_node_types=["EntityType"],
            dry_run=False,
            dataset="ds-1",
        )

    assert result == {"status": "ok"}
    memify_mock.assert_awaited_once()

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["dataset"] == "ds-1"
    assert kwargs["data"] == [{}]
    assert len(kwargs["extraction_tasks"]) == 1  # the _passthrough no-op
    assert len(kwargs["enrichment_tasks"]) == 1  # the reconcile_memory task

    # every decay/reconcile knob must land on the enrichment Task as the exact keyword-only kwargs
    enrichment_kwargs = kwargs["enrichment_tasks"][0].default_params["kwargs"]
    assert enrichment_kwargs["confidence_threshold"] == 0.8
    assert enrichment_kwargs["prefer"] == "feedback"
    assert enrichment_kwargs["demote_factor"] == 0.3
    assert enrichment_kwargs["max_pairs"] == 7
    assert enrichment_kwargs["protect_node_types"] == ["EntityType"]
    assert enrichment_kwargs["dry_run"] is False
