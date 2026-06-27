"""In-process OpenAI-compatible mock for deployment e2e tests."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

GOLDEN_DOCUMENT = "Albert Einstein developed the theory of relativity."
GOLDEN_ENTITY = "Albert Einstein"
GOLDEN_ANSWER = "Albert Einstein developed the theory of relativity."


def _resolve_schema_name(req: dict) -> str:
    response_format = req.get("response_format") or {}
    json_schema = response_format.get("json_schema") or {}
    schema_name = json_schema.get("name", "")
    if schema_name:
        return schema_name

    tools = req.get("tools") or []
    if tools and isinstance(tools, list):
        schema_name = tools[0].get("function", {}).get("name", "")
        if schema_name:
            return schema_name

    functions = req.get("functions") or []
    if functions and isinstance(functions, list):
        schema_name = functions[0].get("name", "")
        if schema_name:
            return schema_name

    return ""


def build_chat_completion_content(req: dict) -> str:
    schema_name = _resolve_schema_name(req)

    if schema_name == "DefaultContentPrediction":
        return json.dumps(
            {
                "label": {
                    "type": "TEXTUAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES",
                    "subclass": ["News stories and blog posts"],
                }
            }
        )

    if schema_name == "SummarizedContent":
        return json.dumps(
            {
                "summary": (
                    "This document covers Albert Einstein and his development of "
                    "the theory of relativity."
                ),
                "description": "",
            }
        )

    if schema_name == "KnowledgeGraph":
        return json.dumps(
            {
                "nodes": [
                    {
                        "id": "einstein",
                        "name": GOLDEN_ENTITY,
                        "type": "Person",
                        "description": "A theoretical physicist.",
                    }
                ],
                "edges": [
                    {
                        "source_node_id": "einstein",
                        "target_node_id": "relativity",
                        "relationship_name": "developed",
                        "description": GOLDEN_ANSWER,
                    }
                ],
            }
        )

    if schema_name == "Answer":
        return json.dumps({"answer": GOLDEN_ANSWER})

    # Plain string completions (GRAPH_COMPLETION with response_model=str).
    return GOLDEN_ANSWER


def build_embeddings_response(req: dict) -> dict:
    model = req.get("model", "")
    dim = 3072 if "text-embedding-3-large" in model else 1536
    inputs = req.get("input", [])

    if isinstance(inputs, list):
        data_list = [
            {"object": "embedding", "index": idx, "embedding": [0.1] * dim}
            for idx in range(len(inputs))
        ]
    else:
        data_list = [{"object": "embedding", "index": 0, "embedding": [0.1] * dim}]

    return {"object": "list", "data": data_list, "model": model}


class MockLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            req = {}

        if self.path.endswith("/chat/completions"):
            content = build_chat_completion_content(req)
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "gpt-5-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.endswith("/embeddings"):
            payload = json.dumps(build_embeddings_response(req)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()


def start_mock_llm_server() -> tuple[HTTPServer, Thread, int]:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]

    server = HTTPServer(("0.0.0.0", port), MockLLMHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port


def stop_mock_llm_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()
