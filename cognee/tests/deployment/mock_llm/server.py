from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import hashlib
import os
import numpy as np
from datetime import datetime
from typing import Optional
import asyncio
import httpx

app = FastAPI(title="Mock LLM Server")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    tools: Optional[list] = None
    response_format: Optional[dict] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list
    dimensions: int = 1536

# In-memory cache for chat responses
chat_cache: dict = {}
call_order: list = []  # Track order for multi-step cognify

def hash_prompt(messages: list) -> str:
    """Create SHA-256 hash of prompt for cassette lookup"""
    last = messages[-1]
    content = last.get("content", "") if isinstance(last, dict) else str(last)
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()

def resolve_ref(ref_str: str, root_schema: dict):
    parts = ref_str.lstrip("#/").split("/")
    curr = root_schema
    for p in parts:
        curr = curr.get(p, {})
    return curr

def generate_dummy_from_schema(schema: dict, root_schema: dict = None):
    """Generate dummy data matching a JSON schema"""
    if not root_schema:
        root_schema = schema

    if "$ref" in schema:
        schema = resolve_ref(schema["$ref"], root_schema)

    if "anyOf" in schema:
        return generate_dummy_from_schema(schema["anyOf"][0], root_schema)
    if "allOf" in schema:
        return generate_dummy_from_schema(schema["allOf"][0], root_schema)
    if "oneOf" in schema:
        return generate_dummy_from_schema(schema["oneOf"][0], root_schema)

    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    if "enum" in schema and isinstance(schema["enum"], list) and len(schema["enum"]) > 0:
        return schema["enum"][0]

    if "type" not in schema:
        if "properties" in schema:
            schema["type"] = "object"
        else:
            return "dummy"

    if schema["type"] == "object":
        return {k: generate_dummy_from_schema(v, root_schema) for k, v in schema.get("properties", {}).items()}
    elif schema["type"] == "array":
        items_schema = schema.get("items", {})
        if "$ref" in items_schema:
            items_schema = resolve_ref(items_schema["$ref"], root_schema)
        return [generate_dummy_from_schema(items_schema, root_schema)]
    elif schema["type"] == "string":
        if "rating" in str(schema).lower():
            return "helpful"
        return "Alice" if "name" in str(schema).lower() else "dummy text"
    elif schema["type"] == "integer" or schema["type"] == "number":
        return 1
    elif schema["type"] == "boolean":
        return True
    return "dummy"

from .cassette import cassette_manager

RECORD_MODE = os.getenv("MOCK_LLM_MODE", "replay")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt_hash = hash_prompt(request.messages)

    # --- REPLAY MODE (default, no API calls) ---
    if RECORD_MODE in ["replay", "none"]:
        cached = cassette_manager.replay_chat(prompt_hash)
        if cached:
            return cached
        # No cassette found → return canned dummy response
        return _make_dummy_response(request, prompt_hash)

    # --- ONCE mode: replay if exists, else record ---
    if RECORD_MODE == "once":
        cached = cassette_manager.replay_chat(prompt_hash)
        if cached:
            return cached
        # Fall through to record below

    # --- RECORD / NEW_EPISODES mode: call real LLM ---
    if RECORD_MODE in ["all", "new_episodes", "once"]:
        real_api_key = os.getenv("REAL_LLM_API_KEY")
        real_provider = os.getenv("REAL_LLM_PROVIDER", "groq")

        if not real_api_key:
            # No API key → return dummy canned response
            print(f"⚠️  MOCK_LLM_MODE={RECORD_MODE} but REAL_LLM_API_KEY not set. Using canned response.")
            return _make_dummy_response(request, prompt_hash)

        try:
            if real_provider == "groq":
                async with httpx.AsyncClient(
                    base_url="https://api.groq.com/openai/v1",
                    headers={"Authorization": f"Bearer {real_api_key}"},
                    timeout=60.0
                ) as groq_client:
                    resp = await groq_client.post(
                        "/chat/completions",
                        json={
                            "model": "llama3-8b-8192",
                            "messages": request.messages,
                            "temperature": request.temperature
                        }
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        response_dict = {
                            "model": request.model,
                            "id": data["id"],
                            "choices": [
                                {
                                    "message": {
                                        "content": choice["message"]["content"],
                                        "role": choice["message"]["role"]
                                    },
                                    "index": choice["index"]
                                }
                                for choice in data["choices"]
                            ],
                            "created": data["created"]
                        }
                        cassette_manager.record_chat(prompt_hash, response_dict)
                        return response_dict
                    else:
                        print(f"⚠️  Groq API error: {resp.status_code} {resp.text}. Using canned response.")

            elif real_provider == "openai":
                import openai
                client = openai.AsyncOpenAI(api_key=real_api_key)
                kwargs = {}
                if request.tools:
                    kwargs["tools"] = request.tools
                if request.response_format:
                    kwargs["response_format"] = request.response_format

                response = await client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    **kwargs
                )

                choices = []
                for choice in response.choices:
                    message_dict = {"role": choice.message.role, "content": choice.message.content}
                    if getattr(choice.message, "tool_calls", None):
                        message_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in choice.message.tool_calls
                        ]
                    choices.append({"message": message_dict, "index": choice.index})

                response_dict = {
                    "model": request.model,
                    "id": response.id,
                    "choices": choices,
                    "created": response.created
                }
                cassette_manager.record_chat(prompt_hash, response_dict)
                return response_dict

        except Exception as e:
            print(f"⚠️  Real LLM call failed: {e}. Using canned response.")

    # Fallback: canned dummy response
    return _make_dummy_response(request, prompt_hash)


def _make_dummy_response(request: ChatCompletionRequest, prompt_hash: str) -> dict:
    """Return a canned structured response. Handles tools and response_format."""
    # If tools are requested, return a tool_call response
    if request.tools:
        first_tool = request.tools[0]
        func = first_tool.get("function", {})
        params_schema = func.get("parameters", {})
        dummy_args = generate_dummy_from_schema(params_schema)
        return {
            "model": request.model,
            "id": f"mock-{prompt_hash[:8]}",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{prompt_hash[:8]}",
                        "type": "function",
                        "function": {
                            "name": func.get("name", "mock_tool"),
                            "arguments": json.dumps(dummy_args)
                        }
                    }]
                },
                "index": 0,
                "finish_reason": "tool_calls"
            }],
            "created": int(datetime.now().timestamp())
        }

    # If response_format is requested (structured output), generate dummy JSON
    if request.response_format:
        schema = request.response_format.get("json_schema", {}).get("schema", {})
        if schema:
            dummy = generate_dummy_from_schema(schema)
            content = json.dumps(dummy)
        else:
            content = json.dumps({"entities": [{"name": "Alice", "type": "PERSON"}],
                                  "relationships": [{"from": "Alice", "to": "Cognee", "type": "WORKS_AT"}]})
    else:
        content = json.dumps({
            "entities": [{"name": "Alice", "type": "PERSON"}],
            "relationships": [{"from": "Alice", "to": "Cognee", "type": "WORKS_AT"}]
        })

    response = {
        "model": request.model,
        "id": f"mock-{prompt_hash[:8]}",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content
            },
            "index": 0,
            "finish_reason": "stop"
        }],
        "created": int(datetime.now().timestamp())
    }

    # Cache for multi-step calls
    chat_cache[prompt_hash] = response
    return response


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = request.input if isinstance(request.input, list) else [request.input]

    data = []
    for i, inp in enumerate(inputs):
        input_hash = hashlib.sha256(str(inp).encode()).hexdigest()
        np.random.seed(hash(input_hash) % 2**32)
        vector = np.random.randn(request.dimensions).astype(float)
        data.append({
            "object": "embedding",
            "embedding": list(vector),
            "index": i
        })

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}
    }

if __name__ == "__main__":
    port = int(os.getenv("MOCK_LLM_PORT", "11434"))
    print(f"🚀 Starting mock LLM server on port {port}")
    print(f"  Mode: {RECORD_MODE} (default: no API calls)")
    uvicorn.run(app, host="0.0.0.0", port=port)
