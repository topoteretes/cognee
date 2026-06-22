"""Story view: the main canvas-based graph renderer.

This is the largest JS chunk — the canvas/quadtree/minimap/FPS rendering
loop, ranked/organic layouts, label budget, hover/search/inspector wiring,
density layers, and all interaction state.

Phase 1 ships the existing renderer verbatim. The actual Story view
behavior (pipeline layout, edge bundling) is layered in by the
``pipeline_layout`` module — see ``layouts/pipeline_layout.py``.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "story_view.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
