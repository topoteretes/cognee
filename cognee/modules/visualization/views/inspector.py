"""Inspector view: right-side detail panel for the selected node.

Phase 1 ships an empty stub — the legacy info-panel logic (inside the
story view's IIFE) continues to handle node detail display.

Phase 1e replaces this with a richer, collapsible-section panel
(Overview / Source / Provenance / Relations / Raw) reading the
preprocessor's enriched fields (``stage``, ``provenance``, ``visual_rank``,
etc.). See the plan section "Phase 1e".
"""


def emit_js(_preprocessed=None) -> str:
    return ""
