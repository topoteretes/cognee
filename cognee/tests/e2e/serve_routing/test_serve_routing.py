"""End-to-end coverage for ``cognee.serve()`` routing.

Every SDK call made while connected via ``serve()`` must reach the remote
instance over HTTP instead of touching the local store. Nothing verified that
before this suite: the unit tests around the proxy drive ``MagicMock`` sessions,
so they pin what the client *intends* to send and would keep passing if the
client and the FastAPI route disagreed about a parameter's name or placement
(query vs form vs multipart). Only a real request against a real route catches
that, which is what this suite does.

Setup: a real cognee instance in a subprocess (real routes, real
Ladybug/LanceDB/SQLite) with a mocked LLM and ``MOCK_EMBEDDING`` — offline and
deterministic, needing no third-party credentials.

The instance runs with access control ON, and the suite drives it with an API
key it mints over HTTP (login -> POST /auth/api-keys), so every proxied call
carries ``X-Api-Key`` exactly as the cloud path does. Running against an
unauthenticated instance would skip the layer most likely to break a proxied
call, so the suite also asserts that auth is genuinely enforced.

The client half runs in this process with its own scratch storage that must stay
EMPTY: an empty local store is the proof that every call was proxied, and a
routing regression shows up here as a local ``DatabaseNotCreatedError`` instead
of a silent wrong answer.
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import UUID, uuid4

import pytest

PORT = int(os.environ.get("SERVE_ROUTING_TEST_PORT", "8777"))
BASE_URL = f"http://127.0.0.1:{PORT}"
DATASET = "serve_routing_dataset"
V1 = "Alice lives in Berlin. She works as a cartographer."
V2 = "Alice lives in Bordeaux. She works as a sommelier."
SERVER_BOOT_TIMEOUT = 300
# Must match the credentials mock_instance.py pins on the instance.
DEFAULT_EMAIL = "serve-routing-e2e@example.com"
DEFAULT_PASSWORD = "serve-routing-e2e-password"


def _client_env(root: Path) -> dict:
    """Isolation for the CLIENT half — its local store must stay empty."""
    return dict(
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        DB_PROVIDER="sqlite",
        DB_NAME="serve_routing_client.db",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="ladybug",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        COGNEE_SKIP_CONNECTION_TEST="true",
        LLM_API_KEY="sk-mocked-never-called",
    )


def _reset_config_caches() -> None:
    """Config factories are lru_cached; drop them so the env above takes."""
    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(scope="module")
def api_key():
    """A live, access-controlled instance on ``BASE_URL``; yields its API key."""
    root = Path(tempfile.mkdtemp(prefix="cognee_serve_routing_"))

    server_env = os.environ.copy()
    server_env.pop("VECTOR_DB_URL", None)
    server_env.pop("GRAPH_DATABASE_URL", None)

    # Launched by PATH, not `-m`: `python -m cognee.tests...` imports the cognee
    # package first, so cognee/__init__.py's load_dotenv(override=True) would run
    # before the launcher could set its environment — and the auth posture latches
    # during that import. As a plain script the launcher runs first, as intended.
    launcher = Path(__file__).parent / "mock_instance.py"
    log_path = root / "instance.log"
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        [sys.executable, str(launcher), str(root), str(PORT)],
        env=server_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    deadline = time.time() + SERVER_BOOT_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"instance exited during boot (code {process.returncode}):\n"
                f"{log_path.read_text()[-3000:]}"
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    else:
        raise RuntimeError(
            f"instance did not become healthy within {SERVER_BOOT_TIMEOUT}s:\n"
            f"{log_path.read_text()[-3000:]}"
        )

    import cognee  # noqa: F401  (its import runs load_dotenv(override=True))

    os.environ.update(_client_env(root / "client"))
    _reset_config_caches()

    yield _mint_api_key()

    if hasattr(os, "killpg"):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    else:
        process.terminate()
    process.wait(timeout=30)
    log_file.close()
    shutil.rmtree(root, ignore_errors=True)


def _status_of(path: str, api_key: str = None) -> int:
    """Status code of a GET against the instance, optionally authenticated."""
    request = urllib.request.Request(f"{BASE_URL}{path}")
    if api_key is not None:
        request.add_header("X-Api-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _mint_api_key() -> str:
    """Log in with the default account and mint an API key, over HTTP.

    The same two calls a self-hosted user makes: POST /auth/login for a bearer
    token, then POST /auth/api-keys, which returns the raw key exactly once.
    """
    login = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/login",
        data=urllib.parse.urlencode(
            {"username": DEFAULT_EMAIL, "password": DEFAULT_PASSWORD}
        ).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(login, timeout=30) as response:
        token = json.load(response)["access_token"]

    mint = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/api-keys",
        data=json.dumps({"name": "serve-routing-e2e"}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(mint, timeout=30) as response:
        return json.load(response)["key"]


def _local_row_counts() -> tuple:
    """(datasets, data) in the CLIENT's own store. Both must stay zero."""
    import sqlite3

    from cognee.infrastructure.databases.relational.config import get_relational_config

    config = get_relational_config()
    path = os.path.join(config.db_path, config.db_name)
    if not os.path.exists(path):
        return 0, 0
    counts = []
    connection = sqlite3.connect(path)
    try:
        for table in ("datasets", "data"):
            try:
                counts.append(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                counts.append(0)
    finally:
        connection.close()
    return tuple(counts)


def test_serve_routes_every_proxied_endpoint(api_key):
    """Walk the proxied SDK surface against a live instance.

    One coroutine drives the whole scenario: cognee's cached engines bind to the
    running event loop, so the suite keeps a single loop (same reason the
    incremental-update e2e suite does). Every step records its own result so one
    failure does not hide the rest of the surface.
    """

    async def scenario():
        import cognee
        from cognee.modules.search.types import SearchType

        checks = []

        def record(name, ok, detail=""):
            checks.append((name, ok, detail))
            print(f"{'PASS' if ok else 'FAIL'} :: {name}" + (f" :: {detail}" if detail else ""))

        # Auth is genuinely on for this run — otherwise everything below would
        # pass against an open instance and prove nothing about the header path.
        record(
            "the instance rejects unauthenticated calls",
            _status_of("/api/v1/datasets") == 401,
            "GET /datasets without a key -> 401",
        )
        record(
            "the instance rejects a bad key",
            _status_of("/api/v1/datasets", "not-a-real-key") == 401,
            "GET /datasets with a bogus key -> 401",
        )

        client = await cognee.serve(url=BASE_URL, api_key=api_key)
        record("serve() connects with an API key", client is not None, "X-Api-Key accepted")

        # --- ingest ------------------------------------------------------
        # NOTE: cognify() with no arguments means "every dataset I own" locally
        # but the route rejects it (400), so the dataset is always explicit here.
        await cognee.add(V1, dataset_name=DATASET)
        cognify_result = await cognee.cognify(datasets=[DATASET])
        record(
            "add() + cognify() are proxied", bool(cognify_result), f"{len(cognify_result)} run(s)"
        )

        dataset_id = UUID(str(list(cognify_result.keys())[0]))

        datasets_local, data_local = _local_row_counts()
        record(
            "the dataset lives only on the instance",
            datasets_local == 0 and data_local == 0,
            f"local datasets={datasets_local} data={data_local}",
        )

        # --- list_data ---------------------------------------------------
        rows = await cognee.datasets.list_data(dataset_id)
        row = rows[0] if rows else None
        record(
            "list_data() returns rows shaped like the local call",
            row is not None
            and isinstance(row.id, UUID)
            and hasattr(row, "mime_type")
            and not hasattr(row, "mimeType"),
            "UUID id, snake_case fields" if row is not None else "no rows",
        )
        data_id = UUID(str(row.id))

        # --- update ------------------------------------------------------
        update_result = await cognee.update(data_id=data_id, data=V2, dataset_id=dataset_id)
        record("update() is proxied", update_result is not None, type(update_result).__name__)

        ids_after = [UUID(str(r.id)) for r in await cognee.datasets.list_data(dataset_id)]
        record(
            "update() replaces in place — the document id survives",
            data_id in ids_after,
            f"{len(ids_after)} row(s)",
        )

        # Lexical, not vector: MOCK_EMBEDDING makes every vector identical, so
        # similarity search returns nothing to assert on. Lexical retrieval
        # reads the stored chunk text, which is exactly the question here —
        # did the update replace what the instance will serve?
        chunks = await cognee.search(query_text="Alice", query_type=SearchType.CHUNKS)
        served = str(await cognee.search(query_text="Alice", query_type=SearchType.CHUNKS_LEXICAL))
        serves_new = "Bordeaux" in served or "sommelier" in served
        serves_old = "Berlin" in served or "cartographer" in served
        record(
            "the updated content is what retrieval serves",
            serves_new and not serves_old,
            f"new={serves_new} old={serves_old}",
        )

        await cognee.update(
            data_id=data_id, data=V2 + " Tagged.", dataset_id=dataset_id, node_set=["routing_tag"]
        )
        record("update() carries node_set", True, "route accepted the repeated form field")

        # --- the rest of the proxied surface ------------------------------
        record("search() is proxied", chunks is not None, "CHUNKS + CHUNKS_LEXICAL")

        recalled = await cognee.recall("Where does Alice live?")
        record("recall() is proxied", recalled is not None, type(recalled).__name__)

        remembered = await cognee.remember(
            "Alice also plays the cello.", dataset_name=DATASET, self_improvement=False
        )
        record("remember() is proxied", remembered is not None, type(remembered).__name__)

        improved = await cognee.improve(dataset=DATASET)
        record("improve() is proxied", improved is not None, type(improved).__name__)

        # --- client-side contract errors ----------------------------------
        try:
            await cognee.update(data_id=data_id, data=[V1, V2], dataset_id=dataset_id)
            record("update() rejects multiple documents", False, "no exception")
        except ValueError as error:
            record("update() rejects multiple documents", True, str(error)[:60])

        try:
            await cognee.update(data_id=data_id, data=12345, dataset_id=dataset_id)
            record("update() rejects an unsupported payload", False, "no exception")
        except TypeError as error:
            record("update() rejects an unsupported payload", True, str(error)[:60])

        try:
            await cognee.update(data_id=uuid4(), data=V2, dataset_id=dataset_id)
            record("update() surfaces the instance's error", False, "no exception")
        except RuntimeError as error:
            record("update() surfaces the instance's error", True, str(error)[:60])

        # --- forget, last: it deletes -------------------------------------
        forgotten = await cognee.forget(dataset_id=dataset_id, memory_only=True)
        record("forget() is proxied", forgotten is not None, type(forgotten).__name__)

        # --- nothing was ever written locally ------------------------------
        datasets_local, data_local = _local_row_counts()
        record(
            "the local store was never written to",
            datasets_local == 0 and data_local == 0,
            f"local datasets={datasets_local} data={data_local}",
        )

        await cognee.disconnect()

        failures = [f"{name} ({detail})" for name, ok, detail in checks if not ok]
        assert not failures, "proxied endpoints failed:\n  " + "\n  ".join(failures)

    asyncio.run(scenario())
