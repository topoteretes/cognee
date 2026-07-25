"""Business view: the default tab — LOOM, the connections canvas.

Reads the sibling ``business_view.js`` chunk. The chunk consumes the
``__NODESET_COLORS__`` and ``__SEARCH_EVENTS__`` tokens (substituted
globally by the orchestrator) plus story_view's ``window._vizNodeById`` /
``window._vizLinks`` globals, so it adds no new data tokens of its own.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "business_view.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
