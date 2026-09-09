"""Journey 7: the HTTP API as a client sees it.

Register, log in, add text, cognify (blocking and background with polling),
search every common type, remember/recall, inspect datasets and data, confirm
another user cannot see the dataset, delete it. Along the way the route table is
compared to a checked-in snapshot so wire-shape changes are deliberate, and
error paths return proper 4xx codes instead of 500s.

Runs synchronously through FastAPI's TestClient: the app owns its own event
loop, so setup and teardown run in throwaway loops around it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from cognee.tests.journeys import _support

SNAPSHOT_PATH = Path(__file__).parent / "api_routes_snapshot.json"

DATASET = "journey_http"
DATASET_BG = "journey_http_background"
DOC_1 = (
    "Title: Harrowgate Lifeboat\n\nThe Harrowgate lifeboat is coxswained by Nkechi Aldermann "
    "and was launched 41 times last year."
)
DOC_2 = (
    "Title: Pellworth Cheese\n\nPellworth Cheese is a washed-rind cheese made by Ottoline Vasquez "
    "in the dairy at Ridge Farm."
)
DOC_3 = (
    "Title: Greywater Kayak Club\n\nThe Greywater Kayak Club is run by Soren Achterberg and paddles "
    "the estuary at dawn."
)
DOC_BG = (
    "Title: Bellmere Windmill\n\nThe Bellmere Windmill was rebuilt by carpenter Iolanthe Grieg "
    "and still grinds barley."
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


async def _reset(root: Path) -> None:
    import cognee
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    # Recreate the schema: the app's lifespan runs migrations once per process and
    # expects the relational DB to exist on every later startup.
    from cognee.modules.engine.operations.setup import setup as engine_setup

    await engine_setup()
    # Engines created in this loop must not leak into the app's loop.
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


@pytest.fixture
def api(tmp_path):
    root = Path(tmp_path)
    _run(_reset(root))
    from fastapi.testclient import TestClient

    from cognee.api.client import app

    with TestClient(app) as client:
        yield client
    _run(_reset(root))


class Session:
    """Thin helper: one authenticated user against the TestClient."""

    def __init__(self, client, email: str, password: str = "Journey-Passw0rd!"):
        self.client = client
        self.email = email
        self.password = password
        self.token = None

    def register_and_login(self) -> "Session":
        response = self.client.post(
            "/api/v1/auth/register", json={"email": self.email, "password": self.password}
        )
        assert response.status_code in (200, 201) or (
            response.status_code == 400 and "ALREADY_EXISTS" in response.text
        ), f"register failed: {response.status_code} {response.text}"
        response = self.client.post(
            "/api/v1/auth/login", data={"username": self.email, "password": self.password}
        )
        assert response.status_code == 200, f"login failed: {response.status_code} {response.text}"
        self.token = response.json()["access_token"]
        assert self.token
        return self

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def get(self, url, **kwargs):
        return self.client.get(url, headers=self.headers, **kwargs)

    def post(self, url, **kwargs):
        return self.client.post(url, headers=self.headers, **kwargs)

    def delete(self, url, **kwargs):
        return self.client.delete(url, headers=self.headers, **kwargs)

    # -- domain helpers -------------------------------------------------------

    def dataset(self, name: str) -> dict | None:
        response = self.get("/api/v1/datasets")
        assert response.status_code == 200, response.text
        return next((d for d in response.json() if d["name"] == name), None)

    def status(self, dataset_id: str, pipeline: str = "cognify_pipeline") -> str | None:
        response = self.get(
            "/api/v1/datasets/status", params={"dataset": dataset_id, "pipeline": pipeline}
        )
        assert response.status_code == 200, response.text
        value = response.json().get(dataset_id)
        if isinstance(value, dict):
            value = value.get(pipeline)
        return value

    def search(self, query: str, search_type: str, dataset: str) -> str:
        response = self.post(
            "/api/v1/search",
            json={"searchType": search_type, "query": query, "datasets": [dataset], "topK": 5},
        )
        assert response.status_code == 200, (
            f"search {search_type}: {response.status_code} {response.text}"
        )
        body = response.json()
        assert isinstance(body, list) and body, f"search {search_type} returned {body!r}"
        return json.dumps(body, default=str).lower()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.journey
def test_http_api_user_journey(api):
    unique = uuid.uuid4().hex[:8]

    # --- liveness ---------------------------------------------------------------
    assert api.get("/").status_code == 200
    health = api.get("/health")
    assert health.status_code == 200, health.text

    # --- unauthenticated access is refused, not crashed -----------------------------
    anonymous = api.get("/api/v1/datasets")
    assert anonymous.status_code in (401, 403), (
        f"expected auth failure, got {anonymous.status_code}"
    )

    alice = Session(api, f"alice-{unique}@cognee-journeys.org").register_and_login()

    # --- add two texts -------------------------------------------------------------
    added = alice.post("/api/v1/add", data={"raw_data": [DOC_1, DOC_2], "datasetName": DATASET})
    assert added.status_code in (200, 201), f"add: {added.status_code} {added.text}"

    dataset = alice.dataset(DATASET)
    assert dataset is not None, "dataset not listed after add"
    dataset_id = dataset["id"]
    for key in ("id", "name", "createdAt", "ownerId"):  # OutDTO is camelCase on the wire
        assert key in dataset, f"dataset DTO lost field {key}: {dataset}"

    rows = alice.get(f"/api/v1/datasets/{dataset_id}/data")
    assert rows.status_code == 200, rows.text
    assert len(rows.json()) == 2, f"expected 2 data rows, got {rows.json()}"

    # --- blocking cognify, then the status endpoint agrees ----------------------------
    cognified = alice.post(
        "/api/v1/cognify", json={"datasets": [DATASET], "runInBackground": False}
    )
    assert cognified.status_code in (200, 201), f"cognify: {cognified.status_code} {cognified.text}"
    assert alice.status(dataset_id) == "DATASET_PROCESSING_COMPLETED", alice.status(dataset_id)

    # --- search every common type returns the fact ------------------------------------
    for search_type in ("CHUNKS", "RAG_COMPLETION", "GRAPH_COMPLETION"):
        text = alice.search("Who coxswains the Harrowgate lifeboat?", search_type, DATASET)
        assert "aldermann" in text, f"{search_type} did not surface the fact: {text[:300]}"

    # --- remember / recall over HTTP ---------------------------------------------------
    remembered = alice.post("/api/v1/remember", data={"raw_data": [DOC_3], "datasetName": DATASET})
    assert remembered.status_code in (200, 201), (
        f"remember: {remembered.status_code} {remembered.text}"
    )

    recalled = alice.post(
        "/api/v1/recall",
        json={"query": "Who runs the Greywater Kayak Club?", "datasets": [DATASET]},
    )
    assert recalled.status_code == 200, f"recall: {recalled.status_code} {recalled.text}"
    assert isinstance(recalled.json(), list) and recalled.json(), recalled.text
    assert "achterberg" in json.dumps(recalled.json(), default=str).lower(), recalled.text[:300]

    rows = alice.get(f"/api/v1/datasets/{dataset_id}/data").json()
    assert len(rows) == 3, f"remember did not add a data row: {len(rows)}"

    history = alice.get("/api/v1/search")
    assert history.status_code == 200, history.text
    assert isinstance(history.json(), list) and len(history.json()) >= 3, (
        "search history does not record the searches just made"
    )

    # --- background cognify with status polling ----------------------------------------
    added_bg = alice.post("/api/v1/add", data={"raw_data": [DOC_BG], "datasetName": DATASET_BG})
    assert added_bg.status_code in (200, 201), added_bg.text
    bg_id = alice.dataset(DATASET_BG)["id"]
    started = alice.post(
        "/api/v1/cognify", json={"datasets": [DATASET_BG], "runInBackground": True}
    )
    assert started.status_code in (200, 201, 202), (
        f"background cognify: {started.status_code} {started.text}"
    )

    deadline = time.monotonic() + 180
    final = None
    while time.monotonic() < deadline:
        final = alice.status(bg_id)
        if final in ("DATASET_PROCESSING_COMPLETED", "DATASET_PROCESSING_ERRORED"):
            break
        time.sleep(1)
    assert final == "DATASET_PROCESSING_COMPLETED", f"background cognify ended as {final}"
    text = alice.search("Who rebuilt the Bellmere Windmill?", "CHUNKS", DATASET_BG)
    assert "grieg" in text, text[:300]

    # --- another user sees none of it -----------------------------------------------------
    bob = Session(api, f"bob-{unique}@cognee-journeys.org").register_and_login()
    assert bob.dataset(DATASET) is None, "another user can list a dataset they do not own"
    leak = bob.post(
        "/api/v1/search",
        json={
            "searchType": "CHUNKS",
            "query": "Who coxswains the Harrowgate lifeboat?",
            "datasetIds": [dataset_id],
        },
    )
    assert leak.status_code in (200, 403, 404), leak.text
    if leak.status_code == 200:
        assert "aldermann" not in json.dumps(leak.json(), default=str).lower(), (
            "another user retrieved content from a dataset they have no permission on"
        )

    # --- validation errors are 4xx, never 500 -------------------------------------------------
    bad_type = alice.post("/api/v1/search", json={"searchType": "NOT_A_TYPE", "query": "x"})
    assert bad_type.status_code in (400, 422), f"invalid searchType -> {bad_type.status_code}"
    no_dataset = alice.post("/api/v1/add", data={"raw_data": ["x"]})
    assert 400 <= no_dataset.status_code < 500, f"add without dataset -> {no_dataset.status_code}"
    missing = alice.get(f"/api/v1/datasets/{uuid.uuid4()}/data")
    assert missing.status_code in (403, 404), f"unknown dataset -> {missing.status_code}"

    # --- delete, and it is gone -------------------------------------------------------------------
    deleted = alice.delete(f"/api/v1/datasets/{dataset_id}")
    assert deleted.status_code in (200, 204), f"delete: {deleted.status_code} {deleted.text}"
    assert alice.dataset(DATASET) is None, "dataset still listed after delete"
    assert alice.get(f"/api/v1/datasets/{dataset_id}/data").status_code in (403, 404)


@pytest.mark.journey
def test_route_table_matches_snapshot(api):
    """The set of public routes is part of the product contract.

    Regenerate deliberately with ``COGNEE_UPDATE_API_SNAPSHOT=1``.
    """
    # Read the table from the served OpenAPI document rather than app.routes:
    # newer FastAPI keeps included routers as lazy objects, and the schema is
    # what clients actually generate against.
    schema = api.get("/openapi.json").json()
    routes = sorted(
        f"{method.upper()} {path}"
        for path, operations in schema.get("paths", {}).items()
        for method in operations
        if method.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert len(routes) > 50, f"suspiciously small route table: {routes}"

    if os.getenv("COGNEE_UPDATE_API_SNAPSHOT") == "1" or not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(json.dumps(routes, indent=2) + "\n")
        if not os.getenv("COGNEE_UPDATE_API_SNAPSHOT"):
            pytest.fail(
                f"route snapshot was missing; wrote {SNAPSHOT_PATH.name}, commit it and rerun"
            )

    expected = json.loads(SNAPSHOT_PATH.read_text())
    added = sorted(set(routes) - set(expected))
    removed = sorted(set(expected) - set(routes))
    assert not added and not removed, (
        "public route table changed.\n"
        + ("  added:\n    " + "\n    ".join(added) + "\n" if added else "")
        + ("  removed:\n    " + "\n    ".join(removed) + "\n" if removed else "")
        + "If intentional, run with COGNEE_UPDATE_API_SNAPSHOT=1 and commit the snapshot."
    )


@pytest.mark.journey
def test_openapi_schema_is_served_and_consistent(api):
    response = api.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = set(schema.get("paths", {}))
    for required in (
        "/api/v1/add",
        "/api/v1/cognify",
        "/api/v1/search",
        "/api/v1/datasets",
        "/api/v1/remember",
        "/api/v1/recall",
    ):
        assert required in paths, f"{required} missing from OpenAPI schema"
    assert _support.MODE in ("mock", "llm")
