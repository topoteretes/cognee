"""``graph_schema_to_graph_model`` must never fetch a URL or read a file named by ``$ref``."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cognee.shared import graph_model_utils
from cognee.shared.exceptions import ExternalSchemaReferenceError
from cognee.shared.graph_model_utils import graph_schema_to_graph_model


def _schema_with_items(ref: str) -> dict:
    return {
        "title": "Graph",
        "type": "object",
        "properties": {"nodes": {"type": "array", "items": {"$ref": ref}}},
    }


IN_DOCUMENT_SCHEMA = {
    "title": "Graph",
    "type": "object",
    "properties": {"nodes": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
    "$defs": {
        "Node": {
            "title": "Node",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    },
}


def test_in_document_ref_is_accepted():
    model = graph_schema_to_graph_model(IN_DOCUMENT_SCHEMA)
    assert "nodes" in model.model_fields


@pytest.mark.parametrize(
    "ref",
    [
        "http://169.254.169.254/latest/meta-data/",
        "https://example.com/schema.json",
        "/etc/passwd",
        "../.env",
        ".env",
        "file:///etc/hosts",
        "other.json#/Node",
    ],
)
def test_external_ref_is_rejected(ref):
    with pytest.raises(ExternalSchemaReferenceError) as exc_info:
        graph_schema_to_graph_model(_schema_with_items(ref))
    assert exc_info.value.status_code == 400
    assert ref in exc_info.value.message


def test_external_ref_nested_in_combinator_is_rejected():
    schema = {
        "title": "Graph",
        "type": "object",
        "properties": {
            "node": {"anyOf": [{"type": "null"}, {"$ref": "https://example.com/node.json"}]}
        },
    }
    with pytest.raises(ExternalSchemaReferenceError):
        graph_schema_to_graph_model(schema)


def test_url_ref_makes_no_request():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"type": "object"}).encode())

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/schema.json"
        with pytest.raises(ExternalSchemaReferenceError):
            graph_schema_to_graph_model(_schema_with_items(url))
    finally:
        server.shutdown()
    assert hits == []


def test_file_ref_does_not_open_the_file(tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"type": "object"}))
    opened = []
    real_open = open

    def spy_open(file, *args, **kwargs):
        if str(file) == str(secret):
            opened.append(file)
        return real_open(file, *args, **kwargs)

    import builtins

    builtins.open, saved = spy_open, builtins.open
    try:
        with pytest.raises(ExternalSchemaReferenceError):
            graph_schema_to_graph_model(_schema_with_items(str(secret)))
    finally:
        builtins.open = saved
    assert opened == []


def test_generator_is_told_not_to_follow_remote_refs(monkeypatch):
    seen = {}
    real_generate = graph_model_utils.generate

    def spy_generate(schema, config):
        seen["allow_remote_refs"] = config.allow_remote_refs
        return real_generate(schema, config=config)

    monkeypatch.setattr(graph_model_utils, "generate", spy_generate)
    graph_schema_to_graph_model(IN_DOCUMENT_SCHEMA)
    assert seen["allow_remote_refs"] is False
