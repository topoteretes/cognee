"""
Release E2E test: 10 parallel users exercising a live Cognee HTTP server.

Boots ``uvicorn cognee.api.client:app`` with ``ENABLE_BACKEND_ACCESS_CONTROL=True``
and fresh storage roots, registers 10 users over HTTP, then runs their scenarios
concurrently:

- users 0-2 (writer):           one dataset, sync cognify, CHUNKS search
- users 3-4 (multi):            two datasets, background cognify + status polling
- user  5   (item_deleter):     three documents, delete one data item, its chunk disappears
- users 6-7 (dataset_deleter):  delete the dataset, build a new one under a different name
- user  8   (forgetter):        forget the dataset, recreate the SAME name from scratch
- user  9   (memory_forgetter): forget memoryOnly=True, re-cognify without re-adding files

Every document embeds a unique sentinel token. Search verification is LLM-free:
SearchType.CHUNKS only queries the vector store and returns raw chunk text, so a
user's results must contain their own sentinels and never a foreign one. Dataset
listings, cross-user search (403) and status probes are asserted both before and
after the delete/forget churn, while users 0-4 keep searching during the churn to
catch neighbour interference.

Cognify itself needs real LLM/embedding credentials in the environment.

Usage:
    python cognee/tests/test_release_multi_user_e2e.py
    pytest cognee/tests/test_release_multi_user_e2e.py
"""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

import httpx

HOST = "127.0.0.1"
PORT = int(os.environ.get("HTTP_API_PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}"
RUN_ID = uuid.uuid4().hex[:8]
NUM_USERS = 10
COGNIFY_TIMEOUT_SECONDS = 1800.0
SEARCH_TOP_K = 25

TOPICS = [
    "the history of alpine glaciers",
    "deep sea bioluminescent creatures",
    "the architecture of gothic cathedrals",
    "fermentation in traditional cooking",
    "the physics of violin acoustics",
    "desert irrigation techniques",
    "the evolution of postal systems",
    "volcanic soil and viticulture",
    "polar navigation before satellites",
    "the printing press and literacy",
]

# Every sentinel ever created, per user index — used to detect cross-user content leaks.
ALL_SENTINELS: dict[int, set[str]] = {index: set() for index in range(NUM_USERS)}


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)


def make_document(user_index: int, doc_index: int) -> tuple[str, str]:
    """Return (sentinel, document text); the sentinel is registered for leak checks."""
    topic = TOPICS[user_index]
    sentinel = f"SENTINEL-{RUN_ID}-U{user_index}-D{doc_index}"
    ALL_SENTINELS[user_index].add(sentinel)
    text = (
        f"This report covers {topic}. It was written for release validation and describes "
        f"{topic} across several sentences so that cognify can build a small knowledge graph "
        f"with entities, places and events related to {topic}. Practitioners agree that "
        f"{topic} rewards careful study, and this document summarizes the essentials. "
        f"The internal reference code of this report is {sentinel}."
    )
    return sentinel, text


class UserSession:
    """One simulated user: an authenticated HTTP client plus what the user owns."""

    def __init__(self, index: int):
        self.index = index
        self.email = f"release_user_{index}_{RUN_ID}@example.com"
        self.password = f"Release-Test-{RUN_ID}-{index}!"
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(COGNIFY_TIMEOUT_SECONDS, connect=15.0),
        )
        self.expected_datasets: set[str] = set()
        self.sentinels: set[str] = set()  # sentinels currently retrievable via search
        self.main_dataset: str = ""
        self.victim_sentinel: str = ""  # item_deleter only: sentinel of the deleted document

    async def close(self) -> None:
        await self.client.aclose()

    async def register_and_login(self) -> None:
        response = await self.client.post(
            "/api/v1/auth/register", json={"email": self.email, "password": self.password}
        )
        assert response.status_code == 201, (
            f"user {self.index}: register failed: {response.status_code} {response.text}"
        )
        response = await self.client.post(
            "/api/v1/auth/login", data={"username": self.email, "password": self.password}
        )
        assert response.status_code == 200, (
            f"user {self.index}: login failed: {response.status_code} {response.text}"
        )
        self.client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    async def add_text(self, dataset_name: str, filename: str, text: str) -> None:
        response = await self.client.post(
            "/api/v1/add",
            data={"datasetName": dataset_name},
            files=[("data", (filename, text.encode("utf-8"), "text/plain"))],
        )
        assert response.status_code == 200, (
            f"user {self.index}: add to '{dataset_name}' failed:"
            f" {response.status_code} {response.text}"
        )

    async def list_datasets(self) -> dict[str, str]:
        """Return {dataset name: dataset id} for every dataset visible to this user."""
        response = await self.client.get("/api/v1/datasets")
        assert response.status_code == 200, (
            f"user {self.index}: dataset listing failed: {response.status_code} {response.text}"
        )
        return {dataset["name"]: dataset["id"] for dataset in response.json()}

    async def dataset_id(self, dataset_name: str) -> str:
        datasets = await self.list_datasets()
        assert dataset_name in datasets, (
            f"user {self.index}: dataset '{dataset_name}' missing from listing {sorted(datasets)}"
        )
        return datasets[dataset_name]

    async def list_data(self, dataset_id: str) -> list[dict]:
        response = await self.client.get(f"/api/v1/datasets/{dataset_id}/data")
        assert response.status_code == 200, (
            f"user {self.index}: data listing failed: {response.status_code} {response.text}"
        )
        return response.json()

    async def cognify(self, dataset_names: list[str], background: bool = False) -> None:
        response = await self.client.post(
            "/api/v1/cognify",
            json={"datasets": dataset_names, "runInBackground": background},
        )
        assert response.status_code == 200, (
            f"user {self.index}: cognify of {dataset_names} failed:"
            f" {response.status_code} {response.text}"
        )

    async def wait_for_cognify(self, dataset_id: str) -> None:
        deadline = time.time() + COGNIFY_TIMEOUT_SECONDS
        while time.time() < deadline:
            response = await self.client.get(
                "/api/v1/datasets/status", params={"dataset": dataset_id}
            )
            assert response.status_code == 200, (
                f"user {self.index}: status check failed: {response.status_code} {response.text}"
            )
            status = response.json().get(dataset_id)
            if status is not None:
                assert "ERRORED" not in status, (
                    f"user {self.index}: cognify errored for dataset {dataset_id}"
                )
                if "COMPLETED" in status:
                    return
            await asyncio.sleep(3)
        raise AssertionError(
            f"user {self.index}: cognify of dataset {dataset_id} did not complete"
            f" within {COGNIFY_TIMEOUT_SECONDS}s"
        )

    async def search_chunks(
        self,
        dataset_names: list[str] | None = None,
        dataset_ids: list[str] | None = None,
        query: str = "release validation report",
    ) -> httpx.Response:
        payload: dict = {"searchType": "CHUNKS", "query": query, "topK": SEARCH_TOP_K}
        if dataset_names is not None:
            payload["datasets"] = dataset_names
        if dataset_ids is not None:
            payload["datasetIds"] = dataset_ids
        return await self.client.post("/api/v1/search", json=payload)

    async def delete_dataset(self, dataset_id: str) -> None:
        response = await self.client.delete(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200, (
            f"user {self.index}: dataset delete failed: {response.status_code} {response.text}"
        )

    async def delete_data_item(self, dataset_id: str, data_id: str) -> None:
        response = await self.client.delete(f"/api/v1/datasets/{dataset_id}/data/{data_id}")
        assert response.status_code == 200, (
            f"user {self.index}: data item delete failed: {response.status_code} {response.text}"
        )

    async def forget(self, dataset_name: str, memory_only: bool = False) -> None:
        response = await self.client.post(
            "/api/v1/forget", json={"dataset": dataset_name, "memoryOnly": memory_only}
        )
        assert response.status_code == 200, (
            f"user {self.index}: forget of '{dataset_name}' failed:"
            f" {response.status_code} {response.text}"
        )


def search_blob(user: UserSession, response: httpx.Response) -> str:
    """Flatten a search response to one string, robust to result wrapper shape."""
    assert response.status_code == 200, (
        f"user {user.index}: search failed: {response.status_code} {response.text}"
    )
    return json.dumps(response.json())


def assert_search_content(
    user: UserSession,
    blob: str,
    expected_present: set[str],
    expected_absent: set[str] = frozenset(),
) -> None:
    """Assert own sentinels are found, retired ones are gone, and nothing foreign leaked."""
    for sentinel in expected_present:
        assert sentinel in blob, (
            f"user {user.index}: own sentinel {sentinel} not found in search results"
        )
    for sentinel in expected_absent:
        assert sentinel not in blob, (
            f"user {user.index}: retired sentinel {sentinel} still in search results"
        )
    for other_index, foreign_sentinels in ALL_SENTINELS.items():
        if other_index == user.index:
            continue
        for sentinel in foreign_sentinels:
            assert sentinel not in blob, (
                f"user {user.index}: search results leaked sentinel {sentinel}"
                f" belonging to user {other_index}"
            )


async def build_and_verify_dataset(
    user: UserSession, dataset_name: str, doc_index: int, background: bool = False
) -> str:
    """Add one document, cognify, and verify its sentinel is retrievable via CHUNKS."""
    sentinel, text = make_document(user.index, doc_index)
    await user.add_text(dataset_name, f"user{user.index}_doc{doc_index}.txt", text)
    dataset_id = await user.dataset_id(dataset_name)
    assert len(await user.list_data(dataset_id)) >= 1
    await user.cognify([dataset_name], background=background)
    await user.wait_for_cognify(dataset_id)
    user.expected_datasets.add(dataset_name)
    user.sentinels.add(sentinel)
    blob = search_blob(user, await user.search_chunks([dataset_name], query=TOPICS[user.index]))
    assert_search_content(user, blob, {sentinel})
    return sentinel


async def build_writer(user: UserSession) -> None:
    user.main_dataset = f"writer_{user.index}_{RUN_ID}"
    await build_and_verify_dataset(user, user.main_dataset, doc_index=0)
    log(f"user {user.index} (writer): built '{user.main_dataset}'")


async def build_multi(user: UserSession) -> None:
    """Two datasets, cognified in the background and awaited via the status endpoint."""
    names = [f"multi_{suffix}_{user.index}_{RUN_ID}" for suffix in ("a", "b")]
    sentinels = {}
    for doc_index, name in enumerate(names):
        sentinel, text = make_document(user.index, doc_index)
        sentinels[name] = sentinel
        await user.add_text(name, f"user{user.index}_doc{doc_index}.txt", text)
    await user.cognify(names, background=True)
    datasets = await user.list_datasets()
    await asyncio.gather(*(user.wait_for_cognify(datasets[name]) for name in names))
    user.expected_datasets.update(names)
    user.sentinels.update(sentinels.values())
    user.main_dataset = names[0]
    for name in names:
        blob = search_blob(user, await user.search_chunks([name], query=TOPICS[user.index]))
        assert_search_content(user, blob, {sentinels[name]})
    log(f"user {user.index} (multi): built {names} via background cognify")


async def build_item_deleter(user: UserSession) -> None:
    """Three documents in one dataset; one of them ('victim') gets deleted during churn."""
    user.main_dataset = f"trimmed_{user.index}_{RUN_ID}"
    filenames = ["keeper_one.txt", "victim.txt", "keeper_two.txt"]
    for doc_index, filename in enumerate(filenames):
        sentinel, text = make_document(user.index, doc_index)
        if filename == "victim.txt":
            user.victim_sentinel = sentinel
        await user.add_text(user.main_dataset, filename, text)
        user.sentinels.add(sentinel)
    dataset_id = await user.dataset_id(user.main_dataset)
    assert len(await user.list_data(dataset_id)) == 3
    await user.cognify([user.main_dataset])
    await user.wait_for_cognify(dataset_id)
    user.expected_datasets.add(user.main_dataset)
    blob = search_blob(
        user, await user.search_chunks([user.main_dataset], query=TOPICS[user.index])
    )
    assert_search_content(user, blob, user.sentinels)
    log(f"user {user.index} (item_deleter): built '{user.main_dataset}' with 3 documents")


BUILDERS = {
    0: build_writer,
    1: build_writer,
    2: build_writer,
    3: build_multi,
    4: build_multi,
    5: build_item_deleter,
    6: build_writer,
    7: build_writer,
    8: build_writer,
    9: build_writer,
}


async def churn_item_deleter(user: UserSession) -> None:
    """Delete a single data item and verify only its chunk disappears."""
    dataset_id = await user.dataset_id(user.main_dataset)
    victim = next(item for item in await user.list_data(dataset_id) if "victim" in item["name"])
    await user.delete_data_item(dataset_id, victim["id"])
    user.sentinels.discard(user.victim_sentinel)
    assert len(await user.list_data(dataset_id)) == 2
    blob = search_blob(
        user, await user.search_chunks([user.main_dataset], query=TOPICS[user.index])
    )
    assert_search_content(user, blob, user.sentinels, expected_absent={user.victim_sentinel})
    log(f"user {user.index} (item_deleter): deleted one document, siblings intact")


async def churn_dataset_deleter(user: UserSession) -> None:
    """Delete the whole dataset, then build a fresh one under a different name."""
    old_name = user.main_dataset
    old_sentinels = set(user.sentinels)
    await user.delete_dataset(await user.dataset_id(old_name))
    user.expected_datasets.discard(old_name)
    user.sentinels.clear()
    assert old_name not in await user.list_datasets(), (
        f"user {user.index}: dataset '{old_name}' still listed after delete"
    )
    user.main_dataset = f"reborn_{user.index}_{RUN_ID}"
    new_sentinel = await build_and_verify_dataset(user, user.main_dataset, doc_index=1)
    blob = search_blob(
        user, await user.search_chunks([user.main_dataset], query=TOPICS[user.index])
    )
    assert_search_content(user, blob, {new_sentinel}, expected_absent=old_sentinels)
    log(f"user {user.index} (dataset_deleter): deleted '{old_name}', built '{user.main_dataset}'")


async def churn_forgetter(user: UserSession) -> None:
    """Forget the dataset, then recreate it under the SAME name from scratch."""
    name = user.main_dataset
    old_sentinels = set(user.sentinels)
    await user.forget(name)
    user.expected_datasets.discard(name)
    user.sentinels.clear()
    assert name not in await user.list_datasets(), (
        f"user {user.index}: dataset '{name}' still listed after forget"
    )
    new_sentinel = await build_and_verify_dataset(user, name, doc_index=1)
    blob = search_blob(user, await user.search_chunks([name], query=TOPICS[user.index]))
    assert_search_content(user, blob, {new_sentinel}, expected_absent=old_sentinels)
    log(f"user {user.index} (forgetter): forgot and recreated '{name}'")


async def churn_memory_forgetter(user: UserSession) -> None:
    """Forget memory only, then re-cognify the preserved raw files."""
    name = user.main_dataset
    await user.forget(name, memory_only=True)
    dataset_id = await user.dataset_id(name)  # dataset record must survive a memory-only forget
    assert len(await user.list_data(dataset_id)) == 1, (
        f"user {user.index}: raw data lost by memory-only forget"
    )
    # Memory is wiped: search must not return the sentinel; the vector collections may
    # be gone entirely, which surfaces as 422 (search prerequisites not met).
    response = await user.search_chunks([name], query=TOPICS[user.index])
    if response.status_code == 200:
        blob = json.dumps(response.json())
        for sentinel in user.sentinels:
            assert sentinel not in blob, (
                f"user {user.index}: sentinel {sentinel} still retrievable after memory-only forget"
            )
    else:
        assert response.status_code == 422, (
            f"user {user.index}: unexpected search failure after memory-only forget:"
            f" {response.status_code} {response.text}"
        )
    await user.cognify([name])
    await user.wait_for_cognify(dataset_id)
    blob = search_blob(user, await user.search_chunks([name], query=TOPICS[user.index]))
    assert_search_content(user, blob, user.sentinels)
    log(f"user {user.index} (memory_forgetter): wiped memory and re-cognified '{name}'")


async def steady_reader(user: UserSession, stop: asyncio.Event) -> None:
    """Keep searching while other users churn; results must stay correct throughout."""
    iterations = 0
    while not stop.is_set():
        blob = search_blob(
            user,
            await user.search_chunks(sorted(user.expected_datasets), query=TOPICS[user.index]),
        )
        assert_search_content(user, blob, user.sentinels)
        iterations += 1
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=2.0)
    log(f"user {user.index}: {iterations} steady reads during churn, all consistent")


async def verify_isolation(users: list[UserSession]) -> None:
    """Each user sees exactly their own datasets and cannot touch anyone else's."""
    listings = {user.index: await user.list_datasets() for user in users}
    for user in users:
        assert set(listings[user.index]) == user.expected_datasets, (
            f"user {user.index}: expected datasets {sorted(user.expected_datasets)},"
            f" listing shows {sorted(listings[user.index])}"
        )

    async def probe_foreign(user: UserSession) -> None:
        for other in users:
            if other is user:
                continue
            for name, dataset_id in listings[other.index].items():
                response = await user.search_chunks(dataset_ids=[dataset_id])
                assert response.status_code == 403, (
                    f"user {user.index} searching '{name}' of user {other.index}:"
                    f" expected 403, got {response.status_code} {response.text[:200]}"
                )
                response = await user.client.get(
                    "/api/v1/datasets/status", params={"dataset": dataset_id}
                )
                # Denial surfaces either as 409 or as a response omitting the dataset.
                if response.status_code == 200:
                    assert dataset_id not in response.json(), (
                        f"user {user.index} got status of foreign dataset '{name}'"
                    )
                else:
                    assert response.status_code == 409, (
                        f"user {user.index} status probe on '{name}':"
                        f" unexpected {response.status_code} {response.text[:200]}"
                    )

    await asyncio.gather(*(probe_foreign(user) for user in users))
    log("isolation verified: listings exact, cross-user search 403, status denied")


async def run_scenarios() -> None:
    users = [UserSession(index) for index in range(NUM_USERS)]
    try:
        log(f"registering {NUM_USERS} users")
        await asyncio.gather(*(user.register_and_login() for user in users))

        log("phase 1: parallel build (add + cognify + search)")
        await asyncio.gather(*(BUILDERS[user.index](user) for user in users))

        log("phase 2: isolation matrix")
        await verify_isolation(users)

        log("phase 3: churn (delete / forget / recreate) with concurrent readers")
        stop = asyncio.Event()

        async def churn() -> None:
            try:
                await asyncio.gather(
                    churn_item_deleter(users[5]),
                    churn_dataset_deleter(users[6]),
                    churn_dataset_deleter(users[7]),
                    churn_forgetter(users[8]),
                    churn_memory_forgetter(users[9]),
                )
            finally:
                stop.set()

        await asyncio.gather(churn(), *(steady_reader(user, stop) for user in users[:5]))

        log("phase 4: final isolation matrix and content check")
        await verify_isolation(users)
        for user in users:
            blob = search_blob(
                user,
                await user.search_chunks(sorted(user.expected_datasets), query=TOPICS[user.index]),
            )
            assert_search_content(user, blob, user.sentinels)
    finally:
        await asyncio.gather(*(user.close() for user in users), return_exceptions=True)


def wait_for_server(url: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Cognee server at {url} did not become ready within {timeout}s")


def start_server() -> subprocess.Popen:
    """Start uvicorn with access control on and fresh storage roots.

    A repo .env (loaded by cognee with override=True) wins over these env vars;
    CI has no .env, so the job env fully controls the configuration there.
    """
    storage_root = tempfile.mkdtemp(prefix=f"cognee_release_e2e_{RUN_ID}_")
    env = {
        **os.environ,
        "ENABLE_BACKEND_ACCESS_CONTROL": "True",
        "DATA_ROOT_DIRECTORY": os.path.join(storage_root, "data"),
        "SYSTEM_ROOT_DIRECTORY": os.path.join(storage_root, "system"),
    }
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cognee.api.client:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        env=env,
        start_new_session=True,
    )


def main() -> None:
    log(f"starting Cognee server on {BASE_URL} (run id {RUN_ID})")
    server = start_server()
    try:
        wait_for_server(f"{BASE_URL}/health")
        log("server ready")
        asyncio.run(run_scenarios())
        log("release multi-user E2E passed")
    finally:
        log("shutting down server")
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)


def test_release_multi_user() -> None:
    main()


if __name__ == "__main__":
    main()
