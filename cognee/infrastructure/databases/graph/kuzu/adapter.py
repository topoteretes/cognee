"""Legacy import path for the Ladybug graph adapter.

The graph adapter was renamed from ``KuzuAdapter`` to ``LadybugAdapter`` on
``dev``. This module preserves the historical import paths so call sites and
external tests that still spell things ``kuzu`` continue to work without
churn. New code should import from
``cognee.infrastructure.databases.graph.ladybug.adapter`` directly.
"""

from cognee.infrastructure.databases.graph.ladybug.adapter import (
    DEFAULT_KUZU_BUFFER_POOL_SIZE,
    DEFAULT_KUZU_MAX_DB_SIZE,
    LadybugAdapter,
)


KuzuAdapter = LadybugAdapter

__all__ = [
    "KuzuAdapter",
    "DEFAULT_KUZU_BUFFER_POOL_SIZE",
    "DEFAULT_KUZU_MAX_DB_SIZE",
]
