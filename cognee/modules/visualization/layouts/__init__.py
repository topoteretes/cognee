"""Layout modules for the cognee graph visualization.

Each submodule exports an ``emit_js(preprocessed) -> str`` function that
returns a JavaScript chunk implementing a layout strategy. The orchestrator
injects the chunk into the main story-view script block via a
``__PIPELINE_LAYOUT_JS__`` placeholder.
"""
from cognee.modules.visualization.layouts import semantic_layout
