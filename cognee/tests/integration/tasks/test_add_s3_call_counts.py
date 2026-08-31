"""S3 calls per added file: the COG-6241 budget, as an executable regression.

Before the fix, one add()-ed file cost 13 S3 HTTP calls (2 PUT, 6 HEAD, 5 GET)
in steady state — the payload was uploaded twice and downloaded and re-hashed
four times. The fix's acceptance budget is at most 1 PUT, 2 HEAD, 1 GET per
file. This test runs the real add() twice against a local moto S3 server (run
2 is the steady state the cloud measures) with every s3fs call counted, and
fails if any verb exceeds the budget — so a change that quietly re-introduces
a duplicate upload or read-back is caught by CI-style tooling, not by a cloud
benchmark after the fact.

Requires ``moto[server]`` (and s3fs/boto3 from the aws extra); skipped when
either is absent. Local run:

    ~/cognee/cognee-clo590/.venv/bin/python -m pytest \\
        cognee/tests/integration/tasks/test_add_s3_call_counts.py -q
"""

import collections
import io
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

moto = pytest.importorskip("moto", reason="requires moto[server] (pip install 'moto[server]')")
s3fs = pytest.importorskip("s3fs", reason="requires s3fs (pip install 'cognee[aws]')")

pytestmark = pytest.mark.asyncio

# The COG-6241 acceptance budget, per file, steady state (run 2).
MAX_PUT_PER_FILE = 1
MAX_HEAD_PER_FILE = 2
MAX_GET_PER_FILE = 1

FILES = 6


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def s3_env():
    import boto3

    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    moto_proc = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    for _ in range(80):
        try:
            client.list_buckets()
            break
        except Exception:
            time.sleep(0.25)
    else:
        moto_proc.terminate()
        pytest.skip("moto server did not come up")
    client.create_bucket(Bucket="bench")

    root = Path(tempfile.mkdtemp(prefix="cognee_s3_counts_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    def clear_config_caches():
        import importlib

        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.files.storage.s3_config", "get_s3_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
            ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
            ("cognee.infrastructure.llm.config", "get_llm_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

    mp = pytest.MonkeyPatch()
    for key, value in dict(
        DB_PROVIDER="sqlite",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        STORAGE_BACKEND="s3",
        STORAGE_BUCKET_NAME="bench",
        AWS_REGION="us-east-1",
        AWS_ACCESS_KEY_ID="testing",
        AWS_SECRET_ACCESS_KEY="testing",
        AWS_ENDPOINT_URL=endpoint,
        DATA_ROOT_DIRECTORY="s3://bench/tenant-x/data",
        # System (sqlite, cache) stays local: this test budgets the DATA path.
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        CACHE_ROOT_DIRECTORY=str(root / "cache"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
        COGNEE_SKIP_CONNECTION_TEST="true",
    ).items():
        mp.setenv(key, value)
    clear_config_caches()

    # Count every S3 API call, through both the async client and the sync bridge.
    counts = collections.Counter()
    original_async = s3fs.S3FileSystem._call_s3
    original_sync = s3fs.S3FileSystem.call_s3

    async def counted_async(self, method, *args, **kwargs):
        counts[method] += 1
        return await original_async(self, method, *args, **kwargs)

    def counted_sync(self, method, *args, **kwargs):
        counts[method] += 1
        return original_sync(self, method, *args, **kwargs)

    s3fs.S3FileSystem._call_s3 = counted_async
    s3fs.S3FileSystem.call_s3 = counted_sync

    yield counts

    s3fs.S3FileSystem._call_s3 = original_async
    s3fs.S3FileSystem.call_s3 = original_sync
    mp.undo()
    clear_config_caches()
    moto_proc.terminate()
    shutil.rmtree(root, ignore_errors=True)


def _uploads():
    from starlette.datastructures import Headers, UploadFile

    return [
        UploadFile(
            file=io.BytesIO(f"call count fixture {index}\n".encode() * 200),
            filename=f"fixture_{index}.txt",
            headers=Headers({"content-type": "application/octet-stream"}),
        )
        for index in range(FILES)
    ]


async def test_steady_state_add_stays_within_the_s3_call_budget(s3_env):
    import cognee

    counts = s3_env

    # Run 1 warms content-addressed storage; run 2 (a fresh dataset over the
    # same content) is the steady state the cloud benchmark measures.
    await cognee.add(_uploads(), "s3_budget_run1")
    counts.clear()
    await cognee.add(_uploads(), "s3_budget_run2")

    per_file = {verb: count / FILES for verb, count in sorted(counts.items())}
    # A budget check over an idle meter proves nothing: if the S3 backend was
    # never engaged (config regression, missing extra), fail loudly instead of
    # passing trivially.
    assert per_file.get("put_object", 0) >= 1, (
        f"S3 was not engaged by add() — counts: {per_file or '{}'}; "
        "the storage backend fell back to local and this test measured nothing."
    )
    budget = {
        "put_object": MAX_PUT_PER_FILE,
        "head_object": MAX_HEAD_PER_FILE,
        "get_object": MAX_GET_PER_FILE,
    }
    over = {
        verb: (per_file.get(verb, 0), limit)
        for verb, limit in budget.items()
        if per_file.get(verb, 0) > limit
    }
    assert not over, (
        f"S3 calls per file over budget: {over}; all counts: {per_file}. "
        "A duplicate upload or read-back has crept back into the add() path "
        "(see COG-6241)."
    )
