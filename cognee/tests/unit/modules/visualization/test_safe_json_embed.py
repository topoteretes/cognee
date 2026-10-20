"""Regression tests for ``_safe_json_embed``.

The helper embeds graph JSON into inline ``<script>`` blocks in
``template.html``. It must neutralise every HTML script-data breakout
sequence — not only ``</`` (a premature end tag) but also ``<!--``, which
drives the HTML tokenizer into script-data-(double-)escaped state and makes
the element's real ``</script>`` unrecognised, silently killing the graph
view (issue #4310).

It does that by escaping ``<`` itself as ``\\u003c``, which is a legal escape
in JSON as well as in JS — so the output still round-trips through
``json.loads``, as ``test_orchestrator_assembly`` documents the data tokens
do.
"""

import asyncio
import json

from cognee.modules.visualization import cognee_network_visualization as orch
from cognee.modules.visualization.cognee_network_visualization import (
    _safe_json_embed,
    cognee_network_visualization,
)


def test_safe_json_embed_neutralizes_endtag_and_comment():
    payload = {"text": "x <!-- c --> and </script> and a <script> tag example"}
    out = _safe_json_embed(payload)

    # No raw ``<`` survives, so neither breakout sequence can start.
    assert "<" not in out
    assert "<!--" not in out
    assert "</" not in out

    # ``<`` is a JSON escape as well as a JS one, so the output stays
    # parseable and the embedded value is unchanged once the script runs.
    assert json.loads(out) == payload


def test_safe_json_embed_plain_payload_roundtrips():
    payload = {"nodes": [1, 2, 3], "name": "A", "unicode": "café"}
    assert json.loads(_safe_json_embed(payload)) == payload


def test_rendered_html_escapes_comment_opener_from_node_text(tmp_path, monkeypatch):
    """A node whose text contains ``<!-- ... <script>`` must be embedded with
    the ``<`` neutralised, so it cannot break the Graph tab's ``<script>``.

    The check is scoped to the injected text (the template itself contains
    legitimate ``<!--`` markup comments). The embedding seam is mocked so this
    needs no vector store or LLM, as ``test_semantic_map_assembly`` does.
    """

    async def fake_fetch(nodes, **kwargs):
        return {}

    monkeypatch.setattr(orch, "fetch_node_embeddings", fake_fetch)

    marker = "example <!-- an html comment --> with a <script> snippet"
    nodes = [
        ("a", {"type": "Entity", "name": "A"}),
        ("b", {"type": "DocumentChunk", "text": marker}),
    ]
    edges = [("b", "a", "contains", {})]
    html = asyncio.run(cognee_network_visualization((nodes, edges), str(tmp_path / "out.html")))

    # The raw opener from node text must not appear; only the escaped form.
    assert "example <!-- an html comment" not in html
    assert "example \\u003c!-- an html comment" in html
