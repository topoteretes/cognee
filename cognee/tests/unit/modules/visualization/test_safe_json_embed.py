"""Regression tests for ``_safe_json_embed``.

The helper embeds graph JSON into inline ``<script>`` blocks in
``template.html``. It must neutralise every HTML script-data breakout
sequence — not only ``</`` (a premature end tag) but also ``<!--``, which
drives the HTML tokenizer into script-data-(double-)escaped state and makes
the element's real ``</script>`` unrecognised, silently killing the graph
view (issue #4310).
"""

import asyncio
import json

from cognee.modules.visualization.cognee_network_visualization import (
    _safe_json_embed,
    cognee_network_visualization,
)


def test_safe_json_embed_neutralizes_endtag_and_comment():
    payload = {"text": "x <!-- c --> and </script> and a <script> tag example"}
    out = _safe_json_embed(payload)

    # Neither raw breakout sequence may survive into the embedded script text.
    assert "<!--" not in out
    assert "</" not in out

    # The escapes are JS string identity escapes: undoing them yields the exact
    # original object, so the embedded value is unchanged once the script runs.
    restored = out.replace("<\\!--", "<!--").replace("<\\/", "</")
    assert json.loads(restored) == payload


def test_safe_json_embed_plain_payload_roundtrips():
    payload = {"nodes": [1, 2, 3], "name": "A", "unicode": "café"}
    assert json.loads(_safe_json_embed(payload)) == payload


def test_rendered_html_escapes_comment_opener_from_node_text():
    """A node whose text contains ``<!-- ... <script>`` must be embedded with
    the ``<!--`` neutralised, so it cannot break the Graph tab's ``<script>``.

    The check is scoped to the injected text (the template itself contains
    legitimate ``<!--`` markup comments)."""
    marker = "example <!-- an html comment --> with a <script> snippet"
    nodes = [
        ("a", {"type": "Entity", "name": "A"}),
        ("b", {"type": "DocumentChunk", "text": marker}),
    ]
    edges = [("b", "a", "contains", {})]
    html = asyncio.run(cognee_network_visualization((nodes, edges)))

    # The raw opener from node text must not appear; only the escaped form.
    assert "example <!-- an html comment" not in html
    assert "example <\\!-- an html comment" in html
