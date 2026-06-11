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

This module touches openai-python's **private** modules (``openai._models``,
``openai._utils._compat``, ``openai._utils._typing``, ``openai._response``,
``openai._legacy_response``, ``openai._base_client``). They are not covered
by openai-python's compatibility guarantees and may move between releases. If
that happens, the import block fails fast and ``install()`` becomes a no-op —
cognee keeps working with the uncached originals.

Safe assumptions:
- Inputs to all four helpers are hashable type objects (PEP 484/585/604).
- The helpers are pure functions of their input.
- openai-python does not swap these names at runtime.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

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
    except ImportError as exc:
        logger.debug(
            "openai SDK not available or has shifted private modules; "
            "skipping type-cache install: %s",
            exc,
        )
        return False

    def _cached(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``fn`` with an LRU cache, falling back to ``fn`` on unhashable args.

        The helpers we patch all take a single positional argument; we keep the
        wrapper signature one-arg on purpose rather than ``*args, **kwargs`` so
        an accidental future use with a different shape fails loudly.
        """
        wrapped = functools.lru_cache(maxsize=_CACHE_MAXSIZE)(fn)

        @functools.wraps(fn)
        def safe(arg: Any) -> Any:
            """Cached call; falls back to the uncached ``fn`` on TypeError."""
            try:
                return wrapped(arg)
            except TypeError:
                # Unhashable input — caching can never block a real LLM call.
                return fn(arg)

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

    rebound = 0
    for module, attr, replacement in rebind_targets:
        if hasattr(module, attr):
            setattr(module, attr, replacement)
            rebound += 1

    _INSTALLED = True
    logger.debug(
        "openai type-introspection cache installed (maxsize=%d, %d/%d rebind sites)",
        _CACHE_MAXSIZE,
        rebound,
        len(rebind_targets),
    )
    return True
