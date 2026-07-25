"""Inspector view: schema type side panel (PR3).

Renders the JS that populates ``#schema-side-panel`` when a schema type box
is clicked. The panel shows the type name + true instance count, a bounded
set of sample-name chips with a "Show all" drill-down, and the per-type
outgoing relationship distribution. A "Highlight in graph" action switches
to the Graph tab and lights up the type's instance nodes.

Consumes the PR2 schema-node contract verbatim (``samples``, ``sample_size``,
``relationships``) via ``window._schemaTypeIndex`` populated in schema_view.js.
The highlight bridge calls ``window._highlightSchemaType`` exposed by
story_view.js. The empty stub it replaces returned "".
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "inspector.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
