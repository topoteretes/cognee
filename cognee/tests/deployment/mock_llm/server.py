"""
Minimal OpenAI-compatible mock LLM + embedding server for deterministic CI.

Lets cognee run add -> cognify -> search with no real API key. It returns a
fixed entity ("Alice") for chat and a fixed vector for embeddings, so results
are deterministic.

cognee uses `instructor` tool-calling, so chat replies must be a tool_call whose
arguments fit the schema cognee sends. We reflect that schema and return a
minimal valid instance. Arrays are returned empty — this keeps models with
custom validators / unions (e.g. SessionTurnAnalysis) valid while still filling
the required fields the extraction models need.
"""
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

FIXED_ENTITY = os.environ.get("MOCK_ENTITY", "Einstein")
EMBED_DIM = int(os.environ.get("MOCK_EMBED_DIM", "1536"))


def _instance_from_schema(schema: dict):
    """Minimal deterministic instance satisfying a JSON schema.

    Strategy that stays valid across cognee's many models:
      - resolve $ref / allOf / anyOf / oneOf
      - honor const / enum (covers many constrained strings)
      - objects: fill ONLY required properties
      - arrays: emit [] (empty list satisfies min-less list fields and dodges
        discriminated-union / custom-validator item constraints)
      - scalars: deterministic placeholders; free strings -> FIXED_ENTITY
    """
    defs = schema.get("$defs", {}) or schema.get("definitions", {})

    def resolve(node, depth=0):
        if depth > 40 or not isinstance(node, dict):
            return FIXED_ENTITY

        if "const" in node:
            return node["const"]
        if node.get("enum"):
            return node["enum"][0]

        if "$ref" in node:
            ref = node["$ref"].split("/")[-1]
            return resolve(defs.get(ref, {}), depth + 1)
        if node.get("allOf"):
            return resolve(node["allOf"][0], depth + 1)
        for key in ("anyOf", "oneOf"):
            if node.get(key):
                non_null = [o for o in node[key] if o.get("type") != "null"]
                chosen = non_null[0] if non_null else node[key][0]
                return resolve(chosen, depth + 1)

        t = node.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), "string")

        if t == "object" or ("properties" in node and t is None):
            out = {}
            props = node.get("properties", {})
            required = node.get("required", [])
            for k in required:
                if k in props:
                    out[k] = resolve(props[k], depth + 1)
            return out
        if t == "array":
            # Empty list: satisfies optional/zero-min arrays and avoids
            # inventing items for discriminated unions / validated item models.
            return []
        if t == "integer":
            return 1
        if t == "number":
            return 1.0
        if t == "boolean":
            return True
        return FIXED_ENTITY

    return resolve(schema)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "gpt-4o-mini")
    tools = body.get("tools") or []

    if tools:
        fn = (tools[0] or {}).get("function", {})
        name = fn.get("name", "extract")
        schema = fn.get("parameters", {}) or {}
        args = _instance_from_schema(schema)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mock",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": FIXED_ENTITY}
        finish = "stop"

    return JSONResponse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inp = body.get("input", "")
    items = inp if isinstance(inp, list) else [inp]
    vector = [0.001] * EMBED_DIM
    data = [
        {"object": "embedding", "index": i, "embedding": vector}
        for i, _ in enumerate(items)
    ]
    return JSONResponse(
        {
            "object": "list",
            "model": body.get("model", "text-embedding-3-small"),
            "data": data,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "11434")))