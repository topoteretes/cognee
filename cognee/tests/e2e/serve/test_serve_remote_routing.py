"""Functional test: after ``cognee.serve(url=...)`` the SDK's write/read path
runs on the remote instance, not the local store.

Regression suite for the report "``cognee.update()`` after ``cognee.serve()``
runs against the local store": ``update()`` resolved the LOCAL default user,
ran a local ``datasets.delete_data`` against the remote dataset UUID and
failed with ``Dataset '<uuid>' not found`` while the remote document stayed
untouched. The same report listed two gaps on the path — a ``DataItem``'s
pinned ``data_id`` never left the SDK, and ``datasets.list_data()`` read the
local store.

Setup: a real cognee API server in a subprocess (mocked LLM + embeddings, no
API keys, ``ENABLE_BACKEND_ACCESS_CONTROL=false`` so the default user answers
without a key), its own scratch data root; the SDK in this process with a
*different* scratch root that must stay empty. The scenario drives the public
SDK — ``remember`` → ``datasets.list_data`` → ``update`` (both the chunk-level
and the full-rebuild path, by id and by name) — and checks the server's raw
document after each step. Everything runs in one test coroutine: cognee's
cached engines and the aiohttp session bind to the running event loop.
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest

STARTUP_TIMEOUT_SECONDS = 240
DATASET = "serve_e2e"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _paragraph(i: int, tag: str = "") -> str:
    words = " ".join(f"w{i}{j:02d}" for j in range(30))
    return f"Paragraph {i}{tag} ENTP{i} ENTSHARED. {words}.\n"


TEXT_V1 = "".join(_paragraph(i) for i in range(6))
# One paragraph edited: the chunk-level update path replaces just that region.
TEXT_V2 = TEXT_V1.replace(_paragraph(3), _paragraph(3, " revised ENTNEW"))
TEXT_V3 = "A different document altogether ENTFINAL. " + "".join(
    _paragraph(i) for i in range(10, 13)
)


def _env_overrides(root: Path) -> dict:
    return {
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "VECTOR_DB_PROVIDER": "lancedb",
        "DB_PROVIDER": "sqlite",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
    }


@pytest.fixture(scope="module")
def remote_server():
    """A real API server in a subprocess, plus an isolated root for this SDK process."""
    root = Path(tempfile.mkdtemp(prefix="cognee_serve_e2e_"))
    server_root = root / "server"
    sdk_root = root / "sdk"
    server_root.mkdir()
    sdk_root.mkdir()

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    server_env = {
        **_env_overrides(server_root),
        # Single-user server: no API key needed, the default user answers.
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        "LLM_API_KEY": os.environ.get("LLM_API_KEY", "mock-key"),
    }
    env_file = root / "server_env.json"
    env_file.write_text(json.dumps(server_env))
    log_path = root / "server.log"

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cognee.tests.e2e.serve.server_runner",
                str(port),
                str(env_file),
            ],
            env={**os.environ, **server_env},
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(root),
        )

    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited early ({process.returncode}):\n{log_path.read_text()[-4000:]}"
                )
            try:
                if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"server not healthy after {STARTUP_TIMEOUT_SECONDS}s:\n"
                    f"{log_path.read_text()[-4000:]}"
                )
            time.sleep(0.5)

        # The SDK side: its own empty root. If anything below ran locally it
        # would land here — the final assertion checks it never did.
        import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

        os.environ.update(_env_overrides(sdk_root))
        from cognee.tests.e2e.serve.server_runner import _clear_config_caches

        _clear_config_caches()

        yield {"base_url": base_url, "sdk_root": sdk_root, "log_path": log_path}
    finally:
        # Captured by pytest, shown only when a test fails: the server-side
        # story (refusals, tracebacks) is otherwise lost with the scratch root.
        if log_path.exists():
            print("----- server log (tail) -----")
            print(log_path.read_text(errors="replace")[-8000:])
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        shutil.rmtree(root, ignore_errors=True)


async def _raw_document(base_url: str, dataset_id: str, data_id: UUID) -> str:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(f"{base_url}/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
        response.raise_for_status()
        return response.text


@pytest.mark.asyncio
async def test_sdk_writes_and_reads_go_to_the_remote_instance(remote_server):
    import cognee
    from cognee.api.v1.serve import state as serve_state
    from cognee.api.v1.serve import credentials as serve_credentials
    from cognee.tasks.ingestion.data_item import DataItem

    base_url = remote_server["base_url"]
    pinned = uuid4()

    # serve() persists credentials under ~/.cognee; keep the suite out of $HOME.
    with patch.object(serve_credentials, "save_credentials"):
        await cognee.serve(url=base_url)
    try:
        assert serve_state.is_remote_mode()

        # --- remember: the pinned id must be the id the server stores ---
        stored = await cognee.remember(
            DataItem(data=TEXT_V1, data_id=pinned),
            dataset_name=DATASET,
            node_set=["serve"],
        )
        assert stored["status"] == "completed", stored
        assert [item["id"] for item in stored["items"]] == [str(pinned)]
        dataset_id = stored["dataset_id"]
        assert dataset_id

        # --- list_data: routed to the server, reports the pinned document ---
        rows = await cognee.datasets.list_data(UUID(dataset_id))
        assert [str(row.id) for row in rows] == [str(pinned)]
        assert await _raw_document(base_url, dataset_id, pinned) == TEXT_V1

        # --- incremental load over the wire: re-sending the identical pinned
        # document resolves to the same row (no duplicate, content untouched) ---
        again = await cognee.remember(
            DataItem(data=TEXT_V1, data_id=pinned), dataset_name=DATASET, node_set=["serve"]
        )
        assert again["status"] == "completed", again
        rows = await cognee.datasets.list_data(UUID(dataset_id))
        assert [str(row.id) for row in rows] == [str(pinned)], "pinned re-ingest must not duplicate"
        assert await _raw_document(base_url, dataset_id, pinned) == TEXT_V1

        # --- identical content through update(): the chunk-level path is a no-op ---
        unchanged = await cognee.update(data_id=pinned, data=TEXT_V1, dataset_id=UUID(dataset_id))
        assert unchanged["status"] == "unchanged", unchanged
        assert unchanged["added_chunks"] == 0 and unchanged["deleted_chunks"] == 0, unchanged

        # --- update by id: chunk-level path, in place ---
        result = await cognee.update(data_id=pinned, data=TEXT_V2, dataset_id=UUID(dataset_id))
        assert result["status"] == "incremental", result
        assert await _raw_document(base_url, dataset_id, pinned) == TEXT_V2
        rows = await cognee.datasets.list_data(UUID(dataset_id))
        assert [str(row.id) for row in rows] == [str(pinned)], "update must replace, not duplicate"

        # --- update by name, with node_set: full-rebuild path keeps the id ---
        result = await cognee.update(
            data_id=pinned, data=TEXT_V3, dataset_name=DATASET, node_set=["serve"]
        )
        assert isinstance(result, dict) and result, result
        assert "status" not in result or result["status"] not in ("incremental", "unchanged")
        assert await _raw_document(base_url, dataset_id, pinned) == TEXT_V3
        rows = await cognee.datasets.list_data(UUID(dataset_id))
        assert [str(row.id) for row in rows] == [str(pinned)], "fallback must keep the id"

        # --- the original failure mode: an unknown id is the REMOTE's 404, not a
        # local "Dataset not found" from a store that never had the dataset ---
        with pytest.raises(RuntimeError, match=r"Remote update failed \(404\)"):
            await cognee.update(data_id=uuid4(), data="nope", dataset_id=UUID(dataset_id))

        # --- nothing ran locally ---
        local_files = [p for p in remote_server["sdk_root"].rglob("*") if p.is_file()]
        assert local_files == [], f"SDK touched its local store: {local_files}"
    finally:
        await cognee.disconnect()

    assert not serve_state.is_remote_mode()
