"""Memory view: the memory-map tab.

Reads the sibling ``memory_map.js`` chunk. The chunk carries the
``__MEMORY_DATA__`` and ``__SEARCH_EVENTS__`` tokens that the orchestrator
substitutes with the preprocessor's ``memory_map`` payload and the optional
``search_events`` argument.

STEP 1 ships a stub JS chunk (token wiring + a no-op
``window._renderMemoryView``) so the orchestrator assembly is complete
end-to-end; the deterministic-layout renderer lands in STEP 2.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "memory_map.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
