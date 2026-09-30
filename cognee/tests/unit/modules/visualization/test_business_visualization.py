import json
import re

import pytest

from cognee.modules.visualization.business_visualization import (
    build_business_visualization_html,
    cognee_business_visualization,
)

GRAPH_DATA = (
    [
        (
            "n1",
            {
                "type": "Entity",
                "name": "Apollo 11",
                "updated_at": 123,
                "created_at": 123,
                "source_node_set": "moon_docs",
            },
        ),
        (
            "n2",
            {
                "type": "DocumentChunk",
                "name": "chunk </script><b>x",
                "updated_at": 123,
                "created_at": 123,
                "source_node_set": "moon_docs",
            },
        ),
    ],
    [("n1", "n2", "mentioned_in", {})],
)


def _embedded_payload(html: str) -> dict:
    match = re.search(r"window\.__COGNEE_PAYLOAD__ = (.*?);</script>", html, re.DOTALL)
    assert match, "payload script missing"
    return json.loads(match.group(1))


def test_html_embeds_payload_and_bundle_without_cdn():
    html = build_business_visualization_html(GRAPH_DATA, dataset_name="apollo")

    assert html.startswith("<!doctype html>")
    assert "<title>cognee · apollo</title>" in html
    # No placeholder survives substitution.
    assert "__PAYLOAD__" not in html and "__BUSINESS_JS__" not in html
    # Self-contained: the renderer bundle is inlined, nothing is loaded from a CDN.
    assert "__COGNEE_PAYLOAD__" in html
    assert 'src="http' not in html
    assert len(html) > 100_000

    payload = _embedded_payload(html)
    assert payload["dataset_name"] == "apollo"
    assert {n["id"] for n in payload["nodes"]} == {"n1", "n2"}
    assert payload["links"][0]["source"] == "n1"
    assert "node_set" in payload["color_maps"]


def test_payload_cannot_break_out_of_script_tag():
    html = build_business_visualization_html(GRAPH_DATA, dataset_name="apollo")

    # `</script>` inside node names is neutralised, so the page still has
    # exactly the two script elements the template defines.
    assert html.count("</script>") == 2
    assert _embedded_payload(html)["nodes"][1]["name"] == "chunk </script><b>x"


@pytest.mark.asyncio
async def test_writes_file_and_returns_html(tmp_path):
    destination = tmp_path / "graph.html"

    html = await cognee_business_visualization(GRAPH_DATA, str(destination), dataset_name="apollo")

    assert destination.read_text(encoding="utf-8") == html
