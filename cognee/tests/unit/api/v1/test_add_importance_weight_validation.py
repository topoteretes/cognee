import pytest
import math
from cognee.api.v1.add import add

@pytest.mark.asyncio
async def test_add_rejects_importance_weight_above_range():
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight=2.1)

@pytest.mark.asyncio
async def test_add_rejects_importance_weight_below_range():
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight=-0.1)

@pytest.mark.asyncio
async def test_add_rejects_nan_importance_weight():
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight=float('nan'))

@pytest.mark.asyncio
async def test_add_rejects_infinite_importance_weight():
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight=float('inf'))

@pytest.mark.asyncio
async def test_add_rejects_non_numeric_importance_weight():
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight="high")
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        await add("test data", importance_weight=True)

@pytest.mark.asyncio
async def test_importance_weight_validation_bounds_are_documented_in_message():
    try:
        await add("test data", importance_weight=3.0)
    except ValueError as e:
        assert "0.0 and 2.0" in str(e)
