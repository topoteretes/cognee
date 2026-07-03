"""Semantic view: the meaning-space scatter tab.

Emits two concatenated chunks: the pure, d3-free ``semantic_core.js`` (the view's
decision layer, also unit-tested in Node) followed by ``semantic_map.js`` (the
d3/DOM shell). Order matters — the core defines ``window.SemanticCore`` that the
shell reads, so the core must come first.

The shell carries the ``__SEMANTIC_CLUSTERS__`` token that the orchestrator
substitutes with the clustering payload (positions arrive via the layout's
``__SEMANTIC_POSITIONS__`` token). Both fall back to ``null`` when there are no
embeddings, and the view shows a friendly empty state. The core is token-free.
"""

import os

_DIR = os.path.dirname(__file__)
_CORE_PATH = os.path.join(_DIR, "semantic_core.js")
_JS_PATH = os.path.join(_DIR, "semantic_map.js")


def emit_js(_preprocessed=None) -> str:
    with open(_CORE_PATH, "r", encoding="utf-8") as f:
        core = f.read()
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        shell = f.read()
    return core + "\n" + shell
