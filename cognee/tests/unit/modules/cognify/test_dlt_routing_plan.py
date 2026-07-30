"""Tests for cognify's DLT routing plan.

The planner never guesses: the standard plan is only returned when the probe
proves there is nothing to route; any probe failure raises instead of
silently sending manifest data through the LLM pipeline.
"""

import sys
from unittest.mock import AsyncMock, patch

import pytest

from cognee.exceptions import CogneeSystemError

cognify_module = sys.modules.get("cognee.api.v1.cognify.cognify") or __import__(
    "cognee.api.v1.cognify.cognify", fromlist=["cognify"]
)


@pytest.mark.asyncio
async def test_probe_failure_raises_instead_of_falling_back():
    with patch.object(
        cognify_module,
        "_probe_cognify_runs",
        new=AsyncMock(side_effect=RuntimeError("db unreachable")),
    ):
        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_module._plan_cognify_runs(["some_dataset"], user=None)

    assert "routing probe failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_successful_probe_result_passes_through():
    plan = [{"datasets": ["ds"], "sub_pipeline_kinds": None}]
    with patch.object(cognify_module, "_probe_cognify_runs", new=AsyncMock(return_value=plan)):
        assert await cognify_module._plan_cognify_runs(["ds"], user=None) == plan
