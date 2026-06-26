from unittest.mock import AsyncMock, patch

import pytest

from cognee.memify_pipelines.decay_memory import decay_memory_pipeline


@pytest.mark.asyncio
async def test_decay_memory_pipeline_wires_memify_tasks():
    """The wrapper forwards its knobs onto the enrichment Task and calls memify with data=[{}] / dataset.

    decay_memory's tuning params are keyword-only, and Task() stores them without validating against the
    target signature — so a kwarg typo (e.g. ``half_life=`` vs ``half_life_days=``) would otherwise only
    surface during a live memify run. This pins the wiring with no graph / no LLM / no DB lock.
    """
    with patch("cognee.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock:
        result = await decay_memory_pipeline(
            half_life_days=10,
            min_weight=0.2,
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
    assert len(kwargs["enrichment_tasks"]) == 1  # the decay_memory task

    # the decay knobs must land on the enrichment Task as the exact keyword-only kwargs
    enrichment_kwargs = kwargs["enrichment_tasks"][0].default_params["kwargs"]
    assert enrichment_kwargs["half_life_days"] == 10
    assert enrichment_kwargs["min_weight"] == 0.2
    assert enrichment_kwargs["protect_node_types"] == ["EntityType"]
    assert enrichment_kwargs["dry_run"] is False
