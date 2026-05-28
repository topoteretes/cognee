"""Schema view: D3-based schema graph visualization.

Renders the type-graph or DLT-schema graph in the Schema tab. Reads
``__SCHEMA_DATA__`` and ``__SCHEMA_GRAPH_DATA__`` tokens that the
orchestrator substitutes with the preprocessor's schema output.

Phase 1 ships the existing renderer verbatim — extracted from the original
2,769-line monolith but unchanged in behavior. Phase 2 reworks this into
a type-card surface (see plan).
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "schema_view.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
