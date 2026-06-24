import os
import time
import json
import socket
import pytest
import httpx
import asyncio
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def pytest_configure(config):
    config.addinivalue_line("markers", "deployment: Mark test as deployment E2E test")


def is_docker_available():
    try:
        result = subprocess.run(
            ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception:
        return False


# Resolve references helper
def resolve_ref(ref, root_schema):
    parts = ref.split("/")
    current = root_schema
    for part in parts:
        if part == "#":
            continue
        if part in current:
            current = current[part]
        else:
            return None
    return current


# Schema generator helper
def generate_mock_from_schema(schema, root_schema=None):
    if root_schema is None:
        root_schema = schema

    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], root_schema)
        if resolved:
            return generate_mock_from_schema(resolved, root_schema)
        return None

    if "type" not in schema:
        if "anyOf" in schema:
            for sub in schema["anyOf"]:
                if sub.get("type") != "null":
                    return generate_mock_from_schema(sub, root_schema)
        return None

    t = schema["type"]
    if isinstance(t, list):
        types = [x for x in t if x != "null"]
        t = types[0] if types else "null"

    if "enum" in schema:
        return schema["enum"][0]

    if t == "string":
        return "mock_value"
    elif t in ("integer", "number"):
        return 1
    elif t == "boolean":
        return True
    elif t == "array":
        item_schema = schema.get("items", {})
        val = generate_mock_from_schema(item_schema, root_schema)
        return [val] if val is not None else []
    elif t == "object":
        properties = schema.get("properties", {})
        res = {}
        for prop_name, prop_schema in properties.items():
            res[prop_name] = generate_mock_from_schema(prop_schema, root_schema)
        return res
    return None


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        req_body = json.loads(post_data.decode("utf-8"))

        path = self.path
        if path.endswith("/chat/completions"):
            arguments = "{}"
            if "tools" in req_body:
                tool = req_body["tools"][0]
                tool_name = tool["function"]["name"]
                parameters_schema = tool["function"]["parameters"]
                mock_data = generate_mock_from_schema(parameters_schema)
                arguments = json.dumps(mock_data)

                resp = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 123456789,
                    "model": req_body.get("model", "mock-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_mock",
                                        "type": "function",
                                        "function": {"name": tool_name, "arguments": arguments},
                                    }
                                ],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            else:
                resp_content = "This is a mock response from the mock OpenAI API."

                response_format = req_body.get("response_format")
                if response_format and response_format.get("type") == "json_object":
                    schema = response_format.get("json_schema", {}).get("schema")
                    if schema:
                        resp_content = json.dumps(generate_mock_from_schema(schema))
                    else:
                        resp_content = json.dumps(
                            {"summary": "Mock summary", "key_features": ["f1"]}
                        )

                resp = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 123456789,
                    "model": req_body.get("model", "mock-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": resp_content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif path.endswith("/embeddings"):
            input_data = req_body.get("input", [])
            if isinstance(input_data, str):
                input_data = [input_data]

            dimensions = 1536
            data_list = []
            for idx, item in enumerate(input_data):
                data_list.append(
                    {"object": "embedding", "index": idx, "embedding": [0.1] * dimensions}
                )

            resp = {
                "object": "list",
                "data": data_list,
                "model": req_body.get("model", "mock-embedding-model"),
                "usage": {
                    "prompt_tokens": len(input_data) * 2,
                    "total_tokens": len(input_data) * 2,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_openai_port():
    return get_free_port()


@pytest.fixture(scope="session", autouse=True)
def mock_openai_server(mock_openai_port):
    server = HTTPServer(("0.0.0.0", mock_openai_port), MockOpenAIHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Wait for mock server to be healthy
    healthy = False
    for _ in range(10):
        try:
            res = httpx.get(f"http://127.0.0.1:{mock_openai_port}")
            if res.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not healthy:
        raise RuntimeError("Failed to start mock OpenAI server")

    yield f"http://127.0.0.1:{mock_openai_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="session")
def build_cognee_image():
    if not is_docker_available():
        pytest.skip("Docker not available")

    print("\nBuilding Cognee Docker image...")
    build_process = subprocess.run(
        ["docker", "compose", "build", "cognee"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if build_process.returncode != 0:
        print("Docker compose build stdout:")
        print(build_process.stdout)
        print("Docker compose build stderr:")
        print(build_process.stderr)
        raise RuntimeError("Failed to build Cognee Docker image")
    print("Cognee Docker image built successfully.")


def wait_for_health(url: str, timeout: int = 60) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ready":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


@pytest.fixture
def running_container(mock_openai_port, build_cognee_image, tmp_path):
    if not is_docker_available():
        pytest.skip("Docker not available")

    # We will use this list to store clean up tasks
    cleanups = []

    def _run(db_stack="sqlite"):
        test_env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-3.5-turbo",
            "LLM_ENDPOINT": f"http://host.docker.internal:{mock_openai_port}/v1",
            "LLM_API_KEY": "mock-key",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_ENDPOINT": f"http://host.docker.internal:{mock_openai_port}/v1",
            "EMBEDDING_API_KEY": "mock-key",
            "MOCK_EMBEDDING": "true",
            "DB_PROVIDER": "sqlite",
            "GRAPH_DATABASE_PROVIDER": "ladybug",
            "VECTOR_DB_PROVIDER": "lancedb",
        }

        compose_cmd = ["docker", "compose"]

        if db_stack == "postgres":
            test_env.update(
                {
                    "DB_PROVIDER": "postgres",
                    "DB_HOST": "postgres",
                    "DB_PORT": "5432",
                    "DB_NAME": "cognee_db",
                    "DB_USERNAME": "cognee",
                    "DB_PASSWORD": "cognee",
                    "VECTOR_DB_PROVIDER": "pgvector",
                    "VECTOR_DB_URL": "postgresql+asyncpg://cognee:cognee@postgres:5432/cognee_db",
                    "GRAPH_DATABASE_PROVIDER": "postgres",
                }
            )
            compose_cmd.extend(["--profile", "postgres"])

        elif db_stack == "neo4j":
            test_env.update(
                {
                    "DB_PROVIDER": "postgres",
                    "DB_HOST": "postgres",
                    "DB_PORT": "5432",
                    "DB_NAME": "cognee_db",
                    "DB_USERNAME": "cognee",
                    "DB_PASSWORD": "cognee",
                    "VECTOR_DB_PROVIDER": "pgvector",
                    "VECTOR_DB_URL": "postgresql+asyncpg://cognee:cognee@postgres:5432/cognee_db",
                    "GRAPH_DATABASE_PROVIDER": "neo4j",
                    "GRAPH_DATABASE_URL": "bolt://neo4j:7687",
                    "GRAPH_DATABASE_USERNAME": "neo4j",
                    "GRAPH_DATABASE_PASSWORD": "pleaseletmein",
                }
            )
            compose_cmd.extend(["--profile", "postgres", "--profile", "neo4j"])

        # Write docker-compose.test.yml dynamically
        yaml_lines = ["services:", "  cognee:", "    environment:"]
        for k, v in test_env.items():
            yaml_lines.append(f"      - {k}={v}")

        test_compose_path = tmp_path / "docker-compose.test.yml"
        test_compose_path.write_text("\n".join(yaml_lines))

        # Start containers
        cmd = compose_cmd + ["-f", "docker-compose.yml", "-f", str(test_compose_path), "up", "-d"]

        # We need to start the correct services
        services_to_start = ["cognee"]
        if db_stack == "postgres":
            services_to_start.append("postgres")
        elif db_stack == "neo4j":
            services_to_start.extend(["postgres", "neo4j"])

        cmd.extend(services_to_start)

        # Register cleanup action
        def cleanup_action():
            print(f"\nTearing down containers for {db_stack}...")
            down_cmd = compose_cmd + [
                "-f",
                "docker-compose.yml",
                "-f",
                str(test_compose_path),
                "down",
                "-v",
            ]
            subprocess.run(down_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cleanups.append(cleanup_action)

        # Run startup command
        up_process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if up_process.returncode != 0:
            print("Docker compose up failed!")
            print("stdout:", up_process.stdout)
            print("stderr:", up_process.stderr)
            raise RuntimeError("Failed to spin up Docker containers")

        # Wait for health
        healthy = wait_for_health("http://localhost:8000/health", timeout=90)
        if not healthy:
            # Get logs to help debug
            logs_process = subprocess.run(
                compose_cmd
                + ["-f", "docker-compose.yml", "-f", str(test_compose_path), "logs", "cognee"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("\nCognee container failed to become healthy. Container logs:")
            print(logs_process.stdout)
            raise RuntimeError("Cognee API container failed health check")

        return "http://localhost:8000"

    yield _run

    # Run all cleanups in reverse order
    for cleanup in reversed(cleanups):
        try:
            cleanup()
        except Exception as e:
            print(f"Error during docker cleanup: {e}")


@pytest.fixture
def api_client():
    class AuthenticatedClient:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
            self.headers = {}

        async def register_and_login(self, email="user@example.com", password="password"):
            # Register user
            register_data = {
                "email": email,
                "password": password,
                "is_active": True,
                "is_superuser": False,
                "is_verified": True,
            }
            try:
                await self.client.post("/api/v1/auth/register", json=register_data)
            except Exception:
                pass  # Already exists or other register issues, proceed to login

            # Login
            login_data = {"username": email, "password": password}
            response = await self.client.post("/api/v1/auth/login", data=login_data)
            response.raise_for_status()
            token_data = response.json()
            token = token_data["access_token"]
            self.headers = {"Authorization": f"Bearer {token}"}
            self.client.headers.update(self.headers)
            return token

        async def post(self, url, **kwargs):
            return await self.client.post(url, **kwargs)

        async def get(self, url, **kwargs):
            return await self.client.get(url, **kwargs)

        async def close(self):
            await self.client.aclose()

    return AuthenticatedClient
