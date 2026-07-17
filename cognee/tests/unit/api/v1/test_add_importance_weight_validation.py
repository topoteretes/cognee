"""Unit tests for cognee.add() importance_weight input validation.

The validation runs as the first statement in add() (before serve()/DB routing),
so these tests exercise the pure boundary check without any database or LLM setup.
"""

import math

import pytest

from cognee.api.v1.add import add
from cognee.exceptions import CogneeValidationError


@pytest.mark.asyncio
async def test_add_rejects_importance_weight_above_range():
    """importance_weight > 2.0 would flip the ranking sign in triplet scoring."""
    with pytest.raises(CogneeValidationError, match="importance_weight"):
        await add("some text", dataset_name="ds", importance_weight=3.0)


@pytest.mark.asyncio
async def test_add_rejects_importance_weight_below_range():
    """Negative importance_weight is not a meaningful weight."""
    with pytest.raises(CogneeValidationError, match="importance_weight"):
        await add("some text", dataset_name="ds", importance_weight=-0.1)


@pytest.mark.asyncio
async def test_add_rejects_nan_importance_weight():
    """NaN corrupts heapq.nsmallest ordering during triplet ranking."""
    with pytest.raises(CogneeValidationError, match="importance_weight"):
        await add("some text", dataset_name="ds", importance_weight=float("nan"))


@pytest.mark.asyncio
async def test_add_rejects_infinite_importance_weight():
    """inf/-inf corrupt ranking arithmetic."""
    for bad in (float("inf"), float("-inf")):
        with pytest.raises(CogneeValidationError, match="importance_weight"):
            await add("some text", dataset_name="ds", importance_weight=bad)


@pytest.mark.asyncio
async def test_add_rejects_non_numeric_importance_weight():
    """Non-numeric weights cannot be persisted to a Float column meaningfully."""
    with pytest.raises(CogneeValidationError, match="importance_weight"):
        await add("some text", dataset_name="ds", importance_weight="very")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_add_accepts_boundary_importance_weights():
    """Weights at the [0.0, 2.0] boundaries pass validation.

    Validation is the first statement, so a passing weight proceeds past it. We
    patch the subsequent serve()/DB surface to avoid needing infrastructure: the
    point is simply that no CogneeValidationError is raised for valid weights.
    """
    from unittest.mock import patch

    # get_remote_client returns None -> falls through to local setup(); stub setup()
    # so we don't touch the DB. We only care that validation did NOT raise.
    with (
        patch(
            "cognee.api.v1.serve.state.get_remote_client",
            return_value=None,
        ),
        patch("cognee.api.v1.add.add.setup", create=True),
        patch(
            "cognee.api.v1.add.add.resolve_authorized_user_dataset",
            side_effect=RuntimeError("stop-after-validation"),
        ),
    ):
        for good in (0.0, 1.0, 2.0, 0.5):
            with pytest.raises(RuntimeError, match="stop-after-validation"):
                # Raises RuntimeError from the stub, proving we got PAST validation.
                await add("some text", dataset_name="ds", importance_weight=good)


def test_importance_weight_validation_bounds_are_documented_in_message():
    """Sanity: the error message names the valid range so callers can self-correct."""
    import asyncio

    try:
        asyncio.run(add("x", dataset_name="ds", importance_weight=math.nan))
    except CogneeValidationError as exc:
        assert "[0.0, 2.0]" in str(exc)
    else:
        pytest.fail("expected CogneeValidationError")
