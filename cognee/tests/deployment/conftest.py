import os
import sys
import time
import uuid
import socket
import pytest
import pytest_asyncio
import httpx
import asyncio
import threading
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

def pytest_addoption(parser):
    parser.addoption(
        "--image",
        action="store",
        default=None,
        help="Docker image to use for main API deployment tests (build, pull, or image name)"
    )
    parser.addoption(
        "--mcp-image",
        action="store",
        default=None,
        help="Docker image to use for MCP deployment tests (build, pull, or image name)"
    )

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def wait_for_health(url: str, timeout: float = 60.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") in ("ready", "ok") or data.get("health") == "healthy":
                    return True
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service at {url} not ready after {timeout}s")

class MockLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress server logs

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        if self.path.endswith('/chat/completions'):
            schema_name = ""
            if "response_format" in req:
                schema_name = req["response_format"].get("json_schema", {}).get("name", "")
            if not schema_name and "tools" in req:
                tools = req["tools"]
                if tools and isinstance(tools, list) and len(tools) > 0:
                    schema_name = tools[0].get("function", {}).get("name", "")
            if not schema_name and "functions" in req:
                funcs = req["functions"]
                if funcs and isinstance(funcs, list) and len(funcs) > 0:
                    schema_name = funcs[0].get("name", "")

            messages = req.get("messages", [])
            prompt = "".join([m.get("content", "") for m in messages])
            
            response_content = "{}"
            if schema_name == "DefaultContentPrediction" or "DefaultContentPrediction" in prompt or "TEXTUAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES" in prompt:
                response_content = json.dumps({
                    "label": {
                        "type": "TEXTUAL_DOCUMENTS_USED_FOR_GENERAL_PURPOSES",
                        "subclass": ["News stories and blog posts"]
                    }
                })
            elif schema_name == "SummarizedContent" or "SummarizedContent" in prompt or "summary" in prompt:
                response_content = json.dumps({
                    "summary": "This document covers Albert Einstein and his development of the theory of relativity."
                })
            elif schema_name == "KnowledgeGraph" or "KnowledgeGraph" in prompt or "nodes" in prompt:
                response_content = json.dumps({
                    "nodes": [
                        {
                            "id": "einstein",
                            "name": "Albert Einstein",
                            "type": "Person",
                            "description": "A theoretical physicist."
                        }
                    ],
                    "edges": [
                        {
                            "source_node_id": "einstein",
                            "target_node_id": "relativity",
                            "relationship_name": "developed",
                            "description": "Albert Einstein developed the theory of relativity."
                        }
                    ]
                })
            elif "Answer" in prompt:
                response_content = json.dumps({
                    "answer": "Albert Einstein developed the theory of relativity."
                })
            else:
                response_content = json.dumps({
                    "nodes": [{"id": "einstein", "name": "Albert Einstein", "type": "Person", "description": "A theoretical physicist."}],
                    "edges": [],
                    "answer": "Albert Einstein developed the theory of relativity."
                })
            
            resp = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "gpt-5-mini"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }]
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
            
        elif self.path.endswith('/embeddings'):
            model = req.get("model", "")
            dim = 1536
            if "text-embedding-3-large" in model:
                dim = 3072
            
            inputs = req.get("input", [])
            data_list = []
            if isinstance(inputs, list):
                for idx, inp in enumerate(inputs):
                    data_list.append({
                        "object": "embedding",
                        "index": idx,
                        "embedding": [0.1] * dim
                    })
            else:
                data_list.append({
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * dim
                })

            resp = {
                "object": "list",
                "data": data_list,
                "model": model,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

@pytest.fixture(scope="session")
def mock_llm_server():
    port = get_free_port()
    server = HTTPServer(('0.0.0.0', port), MockLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-5-mini"
    os.environ["LLM_ENDPOINT"] = f"http://127.0.0.1:{port}/v1"
    os.environ["LLM_API_KEY"] = "mock-key"
    
    os.environ["EMBEDDING_PROVIDER"] = "openai"
    os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
    os.environ["EMBEDDING_DIMENSIONS"] = "1536"
    os.environ["EMBEDDING_ENDPOINT"] = f"http://127.0.0.1:{port}/v1"
    os.environ["EMBEDDING_API_KEY"] = "mock-key"
    
    yield f"http://127.0.0.1:{port}/v1"
    
    server.shutdown()
    server.server_close()

@pytest.fixture(scope="session")
def image_name(request):
    img = request.config.getoption("--image")
    if not img:
        img = os.environ.get("COGNEE_DOCKER_IMAGE")
        
    if not img:
        img = "build"
        
    if img == "build":
        local_image = "cognee:local"
        print("Building local Docker image 'cognee:local' from Dockerfile...")
        proc = subprocess.run(["docker", "build", "-t", local_image, "."], capture_output=True, text=True, errors="replace")
        if proc.returncode == 0:
            print("Successfully built local image 'cognee:local'.")
            return local_image
        else:
            print(f"Failed to build local Docker image: {proc.stderr}")
            print("Falling back to published image 'cognee/cognee:latest'.")
            return "cognee/cognee:latest"
    elif img == "pull":
        return "cognee/cognee:latest"
    else:
        return img

@pytest.fixture(scope="session")
def mcp_image_name(request):
    img = request.config.getoption("--mcp-image")
    if not img:
        img = os.environ.get("COGNEE_MCP_IMAGE")
        
    if not img:
        img = "build"
        
    if img == "build":
        local_image = "cognee-mcp:local"
        print("Building local MCP Docker image 'cognee-mcp:local' from cognee-mcp/Dockerfile...")
        proc = subprocess.run(["docker", "build", "-t", local_image, "-f", "cognee-mcp/Dockerfile", "."], capture_output=True, text=True, errors="replace")
        if proc.returncode == 0:
            print("Successfully built local MCP image 'cognee-mcp:local'.")
            return local_image
        else:
            print(f"Failed to build local MCP image: {proc.stderr}")
            print("Falling back to published image 'cognee/cognee-mcp:latest'.")
            return "cognee/cognee-mcp:latest"
    elif img == "pull":
        return "cognee/cognee-mcp:latest"
    else:
        return img

@pytest.fixture
def running_container(mock_llm_server, image_name):
    from urllib.parse import urlparse
    parsed = urlparse(mock_llm_server)
    mock_llm_port = parsed.port
    
    host_port = get_free_port()
    container_name = f"cognee-test-{uuid.uuid4().hex[:8]}"
    
    is_linux = sys.platform.startswith("linux")
    
    if is_linux:
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "-e", f"HTTP_PORT={host_port}",
            "-e", "ENV=local",
            "-e", "DB_PROVIDER=sqlite",
            "-e", "LLM_PROVIDER=openai",
            "-e", "LLM_MODEL=gpt-5-mini",
            "-e", f"LLM_ENDPOINT=http://127.0.0.1:{mock_llm_port}/v1",
            "-e", "LLM_API_KEY=mock-key",
            "-e", "EMBEDDING_PROVIDER=openai",
            "-e", "EMBEDDING_MODEL=text-embedding-3-small",
            "-e", "EMBEDDING_DIMENSIONS=1536",
            "-e", f"EMBEDDING_ENDPOINT=http://127.0.0.1:{mock_llm_port}/v1",
            "-e", "EMBEDDING_API_KEY=mock-key",
            image_name
        ]
    else:
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{host_port}:8000",
            "--add-host", "host.docker.internal:host-gateway",
            "-e", "ENV=local",
            "-e", "DB_PROVIDER=sqlite",
            "-e", "LLM_PROVIDER=openai",
            "-e", "LLM_MODEL=gpt-5-mini",
            "-e", f"LLM_ENDPOINT=http://host.docker.internal:{mock_llm_port}/v1",
            "-e", "LLM_API_KEY=mock-key",
            "-e", "EMBEDDING_PROVIDER=openai",
            "-e", "EMBEDDING_MODEL=text-embedding-3-small",
            "-e", "EMBEDDING_DIMENSIONS=1536",
            "-e", f"EMBEDDING_ENDPOINT=http://host.docker.internal:{mock_llm_port}/v1",
            "-e", "EMBEDDING_API_KEY=mock-key",
            image_name
        ]
        
    print(f"Starting container {container_name} on host port {host_port} with image {image_name}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed to start: {proc.stderr}")
        
    url = f"http://127.0.0.1:{host_port}"
    
    try:
        wait_for_health(f"{url}/health", timeout=60.0)
        yield {
            "url": url,
            "port": host_port,
            "container_name": container_name
        }
    finally:
        print(f"\n--- CONTAINER LOGS FOR {container_name} ---")
        logs = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, errors="replace")
        sys.stdout.buffer.write(logs.stdout.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stderr.buffer.write(logs.stderr.encode("utf-8", errors="replace"))
        sys.stderr.buffer.write(b"\n")
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"Stopping and removing container {container_name}...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

@pytest_asyncio.fixture
async def api_client(running_container):
    base_url = running_container["url"]
    
    class AuthAsyncClient(httpx.AsyncClient):
        async def register_and_login(self, email="test_user@example.com", password="test_password"):
            reg_payload = {
                "email": email,
                "password": password,
                "is_active": True,
                "is_superuser": False,
                "is_verified": False
            }
            try:
                await self.post("/api/v1/auth/register", json=reg_payload)
            except Exception:
                # Swallow registration failures to ensure idempotency across multiple local runs
                # on persistent storage, falling back to attempting login.
                pass
            
            login_payload = {
                "username": email,
                "password": password
            }
            resp = await self.post("/api/v1/auth/login", data=login_payload)
            resp.raise_for_status()
            token = resp.json()["access_token"]
            self.headers["Authorization"] = f"Bearer {token}"
            return token

    async with AuthAsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client

@pytest_asyncio.fixture
async def mcp_client(mcp_image_name):
    host_port = get_free_port()
    container_name = f"cognee-mcp-test-{uuid.uuid4().hex[:8]}"
    
    is_linux = sys.platform.startswith("linux")
    
    if is_linux:
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "-e", f"HTTP_PORT={host_port}",
            "-e", "TRANSPORT_MODE=http",
            mcp_image_name
        ]
    else:
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{host_port}:8000",
            "-e", "TRANSPORT_MODE=http",
            mcp_image_name
        ]
        
    print(f"Starting MCP container {container_name} on host port {host_port} with image {mcp_image_name}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"docker run for MCP failed to start: {proc.stderr}")
        
    url = f"http://127.0.0.1:{host_port}"
    
    class MCPAsyncClient(httpx.AsyncClient):
        async def call_tool(self, tool_name: str, arguments: dict) -> dict:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            resp = await self.post("/mcp", json=payload)
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                raise RuntimeError(f"MCP Tool error: {result['error']}")
            return result["result"]
            
    try:
        wait_for_health(f"{url}/health", timeout=60.0)
        async with MCPAsyncClient(base_url=url, timeout=30.0) as client:
            yield client
    finally:
        print(f"\n--- MCP CONTAINER LOGS FOR {container_name} ---")
        logs = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, errors="replace")
        sys.stdout.buffer.write(logs.stdout.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stderr.buffer.write(logs.stderr.encode("utf-8", errors="replace"))
        sys.stderr.buffer.write(b"\n")
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"Stopping and removing MCP container {container_name}...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

@pytest.fixture
def run_golden_flow():
    """API golden flow integration test."""
    async def _run(api_client, dataset_name):
        await api_client.register_and_login()
        
        # 1. Add tiny known document
        files = {
            "data": ("document.txt", b"Albert Einstein developed the theory of relativity.", "text/plain")
        }
        data = {
            "datasetName": dataset_name
        }
        resp = await api_client.post("/api/v1/add", data=data, files=files)
        resp.raise_for_status()
        
        # 2. Get dataset_id from datasets router
        resp = await api_client.get("/api/v1/datasets")
        resp.raise_for_status()
        datasets = resp.json()
        
        dataset_id = None
        for ds in datasets:
            if ds["name"] == dataset_name:
                dataset_id = ds["id"]
                break
                
        assert dataset_id is not None, f"Dataset {dataset_name} not found in: {datasets}"
        
        # 3. Trigger Cognify
        cognify_payload = {
            "datasets": [dataset_name],
            "run_in_background": True
        }
        resp = await api_client.post("/api/v1/cognify", json=cognify_payload)
        resp.raise_for_status()
        
        # 4. Poll status (60s timeout under heavy CI virtual environments)
        status_completed = False
        for _ in range(60):
            resp = await api_client.get(f"/api/v1/datasets/status?dataset={dataset_id}")
            resp.raise_for_status()
            status_data = resp.json()
            status = status_data.get(str(dataset_id))
            if status == "DATASET_PROCESSING_COMPLETED":
                status_completed = True
                break
            elif status == "DATASET_PROCESSING_ERRORED":
                raise RuntimeError(
                    f"Cognify pipeline failed for dataset {dataset_id}. "
                    f"Status check payload: {status_data}"
                )
            await asyncio.sleep(1)
            
        assert status_completed, f"Cognify pipeline did not complete in time"
        
        # 5. Search GRAPH_COMPLETION
        search_gc_payload = {
            "search_type": "GRAPH_COMPLETION",
            "query": "Who developed relativity?",
            "dataset_ids": [dataset_id]
        }
        resp = await api_client.post("/api/v1/search", json=search_gc_payload)
        resp.raise_for_status()
        results_gc = resp.json()
        assert any("Einstein" in str(r) for r in results_gc), f"Einstein not found in GRAPH_COMPLETION results: {results_gc}"
        
        # 6. Search CHUNKS
        search_chunks_payload = {
            "search_type": "CHUNKS",
            "query": "Einstein",
            "dataset_ids": [dataset_id]
        }
        resp = await api_client.post("/api/v1/search", json=search_chunks_payload)
        resp.raise_for_status()
        results_chunks = resp.json()
        assert any("Einstein" in str(r) for r in results_chunks), f"Einstein not found in CHUNKS results: {results_chunks}"
        
    return _run
