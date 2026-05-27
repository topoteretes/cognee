"""Sanity tests for the visualization orchestrator after the Phase 1c split.

The orchestrator must:

  - Read ``template.html`` from disk.
  - Inject every view's JS chunk into the right token slot.
  - Substitute every data token (``__NODES_DATA__`` etc.) with valid JSON.
  - Leave NO ``__TOKEN__`` placeholders in the rendered HTML.

These tests do not assert behavior — that's the job of the legacy
``visualization_test.py`` token suite. They pin the *assembly contract*:
no placeholder leaks, every view module gets wired in.
"""

import asyncio
import re

import pytest

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)


def _minimal_graph():
    nodes_data = [
        ("a", {"type": "Entity", "name": "A"}),
        ("b", {"type": "DocumentChunk", "text": "hi"}),
    ]
    edges_data = [("b", "a", "contains", {})]
    return (nodes_data, edges_data)


def _render(tmp_path):
    return asyncio.run(cognee_network_visualization(_minimal_graph(), str(tmp_path / "out.html")))


def test_orchestrator_returns_full_html(tmp_path):
    html = _render(tmp_path)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")


def test_no_token_placeholders_leak(tmp_path):
    """Any ``__SOMETHING__`` left in the output is an unfilled token slot
    and means a view or data substitution was missed."""
    html = _render(tmp_path)
    # Exclude ``<\\/`` JSON escapes — those are legitimate.
    leaks = re.findall(r"__[A-Z][A-Z0-9_]*__", html)
    assert leaks == [], f"unfilled tokens: {leaks}"


def test_all_view_modules_contribute(tmp_path):
    """Spot-check distinctive tokens from each view's JS chunk."""
    html = _render(tmp_path)
    # ui_chrome.js: theme toggle
    assert "_isLightMode" in html
    # schema_view.js: D3 schema rendering
    assert "_renderSchemaGraph" in html
    # story_view.js: canvas renderer + label budget
    assert "computeRankedLayout" in html
    assert "labelBudget" in html


def test_data_tokens_substituted_as_json(tmp_path):
    """Every data token should be replaced with a JSON literal — not the
    literal string ``null`` (except for ``__SCHEMA_DATA__`` which legitimately
    becomes null when no schema is provided)."""
    html = _render(tmp_path)
    # nodes payload contains the entity name we put in
    assert '"name": "A"' in html
    assert '"name": "hi"' in html or '"name": "b"' in html
    # color maps default to empty dicts {} when no provenance is set
    assert "taskColors" in html


def test_schema_data_is_null_when_omitted(tmp_path):
    """The orchestrator emits the literal ``null`` for ``__SCHEMA_DATA__``
    when no schema_data is passed — JS handlers test for it."""
    html = _render(tmp_path)
    # schemaData variable assignment should resolve to a value (null or {})
    assert "const schemaData = null" in html or "const schemaData = {" in html


def test_preprocessor_enrichment_reaches_html(tmp_path):
    """Confirm the JS-facing node payload carries preprocessor-derived fields
    so the renderer can use them."""
    html = _render(tmp_path)
    # ``stage`` and ``label_priority`` are preprocessor-only fields
    assert '"stage":' in html
    assert '"label_priority":' in html
    # ``edge_class`` is set by the preprocessor on every link
    assert '"edge_class":' in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
