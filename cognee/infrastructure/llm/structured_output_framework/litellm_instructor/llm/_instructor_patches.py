"""Cache ``instructor.processing.function_calls.openai_schema`` by input class.

Upstream rebuilds a fresh Pydantic wrapper (via ``create_model``) on every
call — ``prepare_response_model`` invokes ``openai_schema(response_model)``
once per LLM request, and ``create_model`` makes a new class each time.
The new classes are never released (their metaclass/SchemaSerializer/
SchemaValidator are held by Pydantic/abc machinery), so a long-running
process accumulates ~35 Pydantic classes per cognify cycle, measured by
tracemalloc + gc.get_objects(). The wrapper is deterministic in ``cls`` —
memoizing collapses the Nth call to a dict lookup and pins the leak at one
wrapper per distinct response_model.
"""

from __future__ import annotations

from threading import Lock
from typing import Any


_APPLIED = False
_applied_lock = Lock()


def apply() -> None:
    global _APPLIED
    with _applied_lock:
        if _APPLIED:
            return

        from instructor.processing import function_calls as _fc

        original = _fc.openai_schema
        cache: dict[Any, Any] = {}
        cache_lock = Lock()

        def _cached_openai_schema(cls: Any) -> Any:
            wrapped = cache.get(cls)
            if wrapped is not None:
                return wrapped
            with cache_lock:
                wrapped = cache.get(cls)
                if wrapped is not None:
                    return wrapped
                wrapped = original(cls)
                cache[cls] = wrapped
                return wrapped

        _cached_openai_schema.__wrapped__ = original
        _fc.openai_schema = _cached_openai_schema

        # ``instructor.utils.core`` does ``from ..processing.function_calls
        # import ... openai_schema`` inside ``prepare_response_model`` at call
        # time, so its binding picks up our patched module attribute —
        # nothing to re-patch there.

        _APPLIED = True
