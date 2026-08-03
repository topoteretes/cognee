"""Global structured-output concurrency cap (CLO-409 Phase 0b).

``LLMGateway.acreate_structured_output`` is the single choke point every
structured-output call flows through, so one semaphore there bounds the whole
engine's concurrent LLM fan-out. ``llm_max_concurrent_requests == 0`` (the OSS
default) leaves it unbounded; a positive value caps peak concurrency.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

# Resolve the MODULE via sys.modules, not the dotted attribute: the llm package
# re-exports the LLMGateway *class*, so `import ....LLMGateway as gw` can bind the
# class instead of the module. sys.modules is always keyed by module name.
import cognee.infrastructure.llm.LLMGateway  # noqa: F401 — ensure it's imported

gw = sys.modules["cognee.infrastructure.llm.LLMGateway"]


def _peak_concurrency(limit: int, n: int, monkeypatch) -> int:
    monkeypatch.setattr(
        gw, "get_llm_config", lambda: SimpleNamespace(llm_max_concurrent_requests=limit)
    )

    async def scenario() -> int:
        state = {"cur": 0, "peak": 0}

        async def work() -> str:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
            await asyncio.sleep(0.01)  # force overlap
            state["cur"] -= 1
            return "ok"

        results = await asyncio.gather(*[gw._limit_concurrency(work()) for _ in range(n)])
        assert results == ["ok"] * n
        return state["peak"]

    return asyncio.run(scenario())


def test_unbounded_by_default(monkeypatch):
    # limit 0 -> no gating: all coroutines run at once.
    assert _peak_concurrency(0, 5, monkeypatch) == 5


def test_caps_peak_concurrency(monkeypatch):
    # limit 2 -> never more than 2 in flight, even with 6 scheduled.
    assert _peak_concurrency(2, 6, monkeypatch) == 2
