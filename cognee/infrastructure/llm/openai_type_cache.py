"""Cache openai-python's per-response type-introspection helpers.

On every HTTP response, the openai SDK rebuilds its ``ChatCompletion`` response
tree, repeatedly calling ``get_origin`` / ``get_args`` / ``is_annotated_type``
/ ``is_literal_type`` against the same static types. At cognify scale this
introspection dominates CPU. A small ``functools.lru_cache`` over each helper
collapses the cost to a single lookup per type.

The cache is installed by rebinding the helpers at every known import site
inside ``openai/*``. Patching the source module alone is insufficient because
most consumers use ``from ._compat import get_origin``, which captures the
name at import time.

Safe assumptions:
- Inputs to all four helpers are hashable type objects (PEP 484/585/604).
- The helpers are pure functions of their input.
- openai-python does not swap these names at runtime.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

_INSTALLED = False

_CACHE_MAXSIZE = 4096


def install() -> bool:
    """Install the type-introspection cache. Idempotent. Returns True on first install."""
    global _INSTALLED
    if _INSTALLED:
        return False

    try:
        from openai._utils import _compat as _ou_compat
        from openai._utils import _typing as _ou_typing
        from openai import _compat as _openai_compat
        from openai import _utils as _openai_utils
        from openai import (
            _base_client,
            _legacy_response,
            _models,
            _response,
        )
    except ImportError:
        return False

    def _cached(fn: Callable[..., Any]) -> Callable[..., Any]:
        wrapped = functools.lru_cache(maxsize=_CACHE_MAXSIZE)(fn)

        # Fall back to the uncached function if a caller ever passes an
        # unhashable argument — never block a real LLM response on caching.
        @functools.wraps(fn)
        def safe(arg: Any) -> Any:
            try:
                return wrapped(arg)
            except TypeError:
                return fn(arg)

        safe.cache_info = wrapped.cache_info  # type: ignore[attr-defined]
        safe.cache_clear = wrapped.cache_clear  # type: ignore[attr-defined]
        return safe

    cached_get_origin = _cached(_ou_compat.get_origin)
    cached_get_args = _cached(_ou_compat.get_args)
    cached_is_literal_type = _cached(_ou_compat.is_literal_type)
    cached_is_annotated_type = _cached(_ou_typing.is_annotated_type)

    # Rebind the cached versions everywhere the openai SDK has already imported
    # them. `from x import y` captures `y` at import time, so patching only the
    # source module would leave most call sites using the uncached originals.
    rebind_targets: list[tuple[Any, str, Callable[..., Any]]] = [
        (_ou_compat, "get_origin", cached_get_origin),
        (_ou_compat, "get_args", cached_get_args),
        (_ou_compat, "is_literal_type", cached_is_literal_type),
        (_ou_typing, "is_annotated_type", cached_is_annotated_type),
        (_openai_utils, "get_origin", cached_get_origin),
        (_openai_utils, "get_args", cached_get_args),
        (_openai_utils, "is_literal_type", cached_is_literal_type),
        (_openai_utils, "is_annotated_type", cached_is_annotated_type),
        (_openai_compat, "get_origin", cached_get_origin),
        (_openai_compat, "get_args", cached_get_args),
        (_openai_compat, "is_literal_type", cached_is_literal_type),
        (_models, "get_origin", cached_get_origin),
        (_models, "get_args", cached_get_args),
        (_models, "is_literal_type", cached_is_literal_type),
        (_models, "is_annotated_type", cached_is_annotated_type),
        (_response, "get_origin", cached_get_origin),
        (_response, "is_annotated_type", cached_is_annotated_type),
        (_legacy_response, "get_origin", cached_get_origin),
        (_legacy_response, "is_annotated_type", cached_is_annotated_type),
        (_base_client, "get_origin", cached_get_origin),
    ]

    for module, attr, replacement in rebind_targets:
        if hasattr(module, attr):
            setattr(module, attr, replacement)

    _INSTALLED = True
    return True
