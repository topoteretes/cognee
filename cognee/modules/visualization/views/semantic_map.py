"""Semantic view: the meaning-space scatter tab.

Reads the sibling ``semantic_map.js`` chunk. The chunk carries the
``__SEMANTIC_CLUSTERS__`` token that the orchestrator substitutes with the
clustering payload (positions arrive via the layout's ``__SEMANTIC_POSITIONS__``
token). Both fall back to ``null`` when there are no embeddings, and the view
shows a friendly empty state.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "semantic_map.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
