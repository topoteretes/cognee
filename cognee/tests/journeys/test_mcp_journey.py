"""Journey 9: the MCP server the way an agent client uses it.

Spawns ``cognee-mcp`` over stdio exactly like Claude Code or Codex would, lists
the tools, remembers into a fresh system inside a session (the first-session
dataset-creation regression, SDK-192), recalls it back, remembers permanently,
recalls from the graph, and checks the status tool. Then starts the same server
over HTTP and recalls the session fact again, proving state persisted across
transports. A stray print to stdout in stdio mode would corrupt the protocol and
fail the handshake, so a green run also proves stdout carries only frames.

Requires the ``mcp`` client and ``fastmcp`` in the running interpreter
(``uv run --with fastmcp --with mcp pytest ...``); skips otherwise.

In mock mode the server subprocess gets the deterministic AI stand-ins through a
``sitecustomize.py`` on ``PYTHONPATH`` so no network or key is needed.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from cognee.tests.journeys import _support

mcp = pytest.importorskip("mcp", reason="mcp client library not installed")
pytest.importorskip("fastmcp", reason="fastmcp (server runtime) not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SERVER = REPO_ROOT / "cognee-mcp" / "src" / "server.py"

DATASET = "mcp_journey"
SESSION_FACT = (
    "The Ashcombe signal box was staffed by Wilhelmina Prost until the line closed in 1994."
)
PERMANENT_FACT = (
    "Title: Ashcombe Ropeworks\n\nThe Ashcombe Ropeworks was founded by Cassius Delacroix-Nwosu and "
    "still twists hemp by hand."
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.skipif(not SERVER.exists(), reason="cognee-mcp not in tree"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text(result) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server_env(work: Path) -> dict:
    """Environment for the server subprocess: isolated roots, mocks in mock mode."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(
        {
            "DATA_ROOT_DIRECTORY": str(work / "data"),
            "SYSTEM_ROOT_DIRECTORY": str(work / "system"),
            "TELEMETRY_DISABLED": "1",
            "ENV": "dev",
        }
    )
    pythonpath = [str(REPO_ROOT)]
    if _support.IS_MOCK:
        shim = work / "shim"
        shim.mkdir(exist_ok=True)
        (shim / "mock_ai.py").write_bytes((HERE / "mock_ai.py").read_bytes())
        (shim / "sitecustomize.py").write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(__file__))\n"
            "import mock_ai\n"
            "mock_ai.install_all()\n"
        )
        pythonpath.insert(0, str(shim))
        env.update(
            {
                "LLM_API_KEY": "mock-key",
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "openai/gpt-5-mini",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "openai/text-embedding-3-small",
                "EMBEDDING_DIMENSIONS": "256",
                "EMBEDDING_API_KEY": "mock-key",
                "COGNEE_SKIP_PREFLIGHT": "1",
                "COGNEE_SKIP_CONNECTION_TEST": "true",
            }
        )
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _dataset_names(env: dict) -> list[str]:
    """Dataset names as a fresh process on the same roots sees them."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, json, cognee\n"
            "async def main():\n"
            "    rows = await cognee.datasets.list_datasets()\n"
            "    print('DATASETS=' + json.dumps([d.name for d in rows]))\n"
            "asyncio.run(main())",
        ],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in probe.stdout.splitlines():
        if line.startswith("DATASETS="):
            import json

            return json.loads(line[len("DATASETS=") :])
    raise AssertionError(f"dataset probe failed:\n{probe.stdout[-1000:]}\n{probe.stderr[-2000:]}")


@pytest.fixture
def work(tmp_path) -> Path:
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# stdio: the Claude Code / Codex path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stdio_client_journey(work):
    env = _server_env(work)
    stderr_path = work / "server.stderr"
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER), "--transport", "stdio"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    session_id = f"mcp-journey-{uuid.uuid4().hex[:8]}"

    with stderr_path.open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=120)

                # --- the memory API is advertised -----------------------------------
                tools = {tool.name for tool in (await session.list_tools()).tools}
                for required in ("remember", "recall", "forget"):
                    assert required in tools, (
                        f"tool {required!r} not advertised; got {sorted(tools)}"
                    )

                # --- first ever remember on a fresh system, inside a session ----------
                remembered = await session.call_tool(
                    "remember",
                    arguments={
                        "data": SESSION_FACT,
                        "session_id": session_id,
                        "dataset_name": DATASET,
                    },
                )
                assert not remembered.isError, _text(remembered)
                assert "session" in _text(remembered).lower(), _text(remembered)

                recalled = await session.call_tool(
                    "recall",
                    arguments={
                        "query": "Who staffed the Ashcombe signal box?",
                        "session_id": session_id,
                    },
                )
                assert not recalled.isError, _text(recalled)
                assert "prost" in _text(recalled).lower(), (
                    f"session recall over MCP did not return the fact: {_text(recalled)[:400]}"
                )

                # The session remember created its dataset up front (SDK-192):
                # visible to any other process on the same roots before any build.
                assert DATASET in _dataset_names(env), (
                    "session remember did not create its dataset before the background bridge"
                )

                # --- permanent memory: remember runs the build, recall reads the graph --
                permanent = await session.call_tool(
                    "remember", arguments={"data": PERMANENT_FACT, "dataset_name": DATASET}
                )
                assert not permanent.isError, _text(permanent)

                from_graph = await session.call_tool(
                    "recall",
                    arguments={"query": "Who founded the Ashcombe Ropeworks?", "datasets": DATASET},
                )
                assert not from_graph.isError, _text(from_graph)
                assert "delacroix" in _text(from_graph).lower(), (
                    f"graph recall over MCP did not return the fact: {_text(from_graph)[:400]}"
                )

                # --- status tool answers, forget validates its arguments ---------------
                status = await session.call_tool("cognify_status", arguments={})
                assert not status.isError, _text(status)
                assert _text(status).strip(), "cognify_status returned no text"

                bad_forget = await session.call_tool("forget", arguments={})
                assert "dataset" in _text(bad_forget).lower() or bad_forget.isError, (
                    "forget with no target neither errored nor explained itself"
                )

    stderr = stderr_path.read_text(errors="ignore")
    assert "Traceback" not in stderr, f"server logged a traceback:\n{stderr[-3000:]}"


# ---------------------------------------------------------------------------
# HTTP: the hosted / shared path, reading state written over stdio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_client_reads_state_written_over_stdio(work):
    try:
        from mcp.client.streamable_http import streamable_http_client as _http_client
    except ImportError:
        try:
            from mcp.client.streamable_http import streamablehttp_client as _http_client
        except ImportError:  # pragma: no cover - old mcp
            pytest.skip("mcp client lacks streamable HTTP support")

    env = _server_env(work)
    session_id = f"mcp-journey-{uuid.uuid4().hex[:8]}"

    # Write over stdio first.
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER), "--transport", "stdio"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    with (work / "stdio.stderr").open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=120)
                remembered = await session.call_tool(
                    "remember",
                    arguments={
                        "data": SESSION_FACT,
                        "session_id": session_id,
                        "dataset_name": DATASET,
                    },
                )
                assert not remembered.isError, _text(remembered)

    # Then read over HTTP from a separate server process on the same roots.
    port = _free_port()
    with (work / "http.log").open("w") as log:
        proc = subprocess.Popen(
            [
                str(sys.executable),
                str(SERVER),
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            import httpx

            deadline = time.monotonic() + 120
            healthy = False
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                        healthy = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            assert healthy, (
                f"MCP HTTP server did not become healthy:\n{(work / 'http.log').read_text()[-3000:]}"
            )

            async with _http_client(f"http://127.0.0.1:{port}/mcp") as (read, write, *_):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=60)
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    assert {"remember", "recall"} <= tools, sorted(tools)
                    recalled = await session.call_tool(
                        "recall",
                        arguments={
                            "query": "Who staffed the Ashcombe signal box?",
                            "session_id": session_id,
                        },
                    )
                    assert not recalled.isError, _text(recalled)
                    assert "prost" in _text(recalled).lower(), (
                        f"fact written over stdio was not readable over HTTP: {_text(recalled)[:400]}"
                    )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    log_text = (work / "http.log").read_text(errors="ignore")
    assert "Traceback" not in log_text, f"HTTP server logged a traceback:\n{log_text[-3000:]}"
