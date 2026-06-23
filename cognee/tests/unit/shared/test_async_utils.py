import asyncio

import pytest

from cognee.shared.async_utils import gather_with_concurrency_limit


class TestGatherWithConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_preserves_input_order(self):
        async def call(value: int) -> int:
            await asyncio.sleep(0.01 * (3 - value))
            return value

        calls = [lambda value=value: call(value) for value in range(4)]

        assert await gather_with_concurrency_limit(calls, limit=4) == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_limits_concurrent_calls(self):
        active_count = 0
        max_active_count = 0

        async def call(value: int) -> int:
            nonlocal active_count, max_active_count

            active_count += 1
            max_active_count = max(max_active_count, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return value

        calls = [lambda value=value: call(value) for value in range(6)]

        assert await gather_with_concurrency_limit(calls, limit=2) == [0, 1, 2, 3, 4, 5]
        assert max_active_count == 2

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_input(self):
        assert await gather_with_concurrency_limit([], limit=1) == []

    @pytest.mark.asyncio
    async def test_rejects_non_positive_limit(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            await gather_with_concurrency_limit([], limit=0)
