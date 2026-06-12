"""In-process cache for lexical retriever corpora.

Loading the corpus is the dominant per-query cost of lexical retrieval: every
initialize() scans all DocumentChunk nodes from the graph, tokenizes them, and
(for BM25) rebuilds corpus statistics. Entries are keyed per graph context, so
multi-tenant setups stay isolated.

Staleness: write paths (add_data_points, deletes, prune) call invalidate() in
the same process; the TTL bounds staleness for writes this process cannot see
(other API workers, direct adapter writes).
"""

import asyncio
import os
import time
from collections import OrderedDict
from typing import Any, Optional

DEFAULT_TTL_SECONDS = 60.0
MAX_CACHED_CONTEXTS = 16

_cache: "OrderedDict[Any, tuple[float, dict]]" = OrderedDict()
_locks: dict[Any, asyncio.Lock] = {}


def get(key: Any) -> Optional[dict]:
    entry = _cache.get(key)
    if entry is None:
        return None

    built_at, state = entry
    if time.monotonic() - built_at > _ttl_seconds():
        _cache.pop(key, None)
        return None

    _cache.move_to_end(key)
    return state


def put(key: Any, state: dict) -> None:
    _cache[key] = (time.monotonic(), state)
    _cache.move_to_end(key)
    while len(_cache) > MAX_CACHED_CONTEXTS:
        _cache.popitem(last=False)


def lock(key: Any) -> asyncio.Lock:
    """Per-key build lock so concurrent first queries don't rebuild the same corpus.

    Keyed by the running event loop as well: an asyncio.Lock binds to the loop it is
    first awaited on, and reusing it from another loop (new loop per request/test)
    deadlocks or raises.
    """
    lock_key = (id(asyncio.get_running_loop()), key)
    if lock_key not in _locks:
        _locks[lock_key] = asyncio.Lock()
    return _locks[lock_key]


def invalidate() -> None:
    """Drop all cached corpora. Called from write paths after data changes."""
    _cache.clear()


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get("LEXICAL_CORPUS_CACHE_TTL", DEFAULT_TTL_SECONDS))
    except ValueError:
        return DEFAULT_TTL_SECONDS
