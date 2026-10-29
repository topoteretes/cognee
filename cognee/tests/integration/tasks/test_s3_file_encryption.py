"""S3 data root with file encryption off and on: same behaviour, same S3 call budget.

Runs the real ``add()`` against a local moto S3 server twice per mode (run 2 is
the steady state), with every s3fs call counted, and checks for both modes:

* the objects in the bucket are what the mode promises — plaintext, or sealed
  envelopes that leak none of the text;
* every data item reads back through ``open_data_file`` as the document that was
  added (the text item byte for byte, the PDF as its extracted text), and
  ``get_size`` reports the plaintext size;
* re-adding the same content creates no new rows and uploads nothing, so the
  content hash is of the plaintext;
* the raw-download route streams the plaintext, never the envelope;
* the per-file S3 call budget of ``test_add_s3_call_counts.py`` holds unchanged,
  so the encryption layer adds no S3 round trips on either path.

Requires ``moto[server]`` and the ``aws`` extra; skipped when either is absent.
"""

import base64
import collections
import io
import logging
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest

logger = logging.getLogger(__name__)

moto = pytest.importorskip("moto", reason="requires moto[server] (pip install 'moto[server]')")
s3fs = pytest.importorskip("s3fs", reason="requires s3fs (pip install 'cognee[aws]')")

pytestmark = pytest.mark.asyncio

# The COG-6241 budget, per file, steady state — the same numbers
# test_add_s3_call_counts.py enforces for the plaintext path.
MAX_PUT_PER_FILE = 1
MAX_HEAD_PER_FILE = 2
MAX_GET_PER_FILE = 1

FILES = 6
TEXT = (
    "The Orca deployment platform is run by Mira Novak; releases ship every Tuesday "
    "after the 10:00 change review.\n    An indented second line. Code word: ZEBRA-7731.\n"
)
PDF = pathlib.Path(__file__).parents[2] / "test_data" / "artificial-intelligence.pdf"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _clear_config_caches():
    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.files.storage.s3_config", "get_s3_config"),
        ("cognee.infrastructure.files.encryption.config", "get_file_encryption_config"),
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
    from cognee.infrastructure.files.encryption import reset_key_providers

    reset_key_providers()


@pytest.fixture(scope="module")
def s3_server():
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
            logger.debug("Ignoring exception in s3_server", exc_info=True)
            time.sleep(0.25)
    else:
        moto_proc.terminate()
        pytest.skip("moto server did not come up")
    client.create_bucket(Bucket="bench")
    yield endpoint, client
    moto_proc.terminate()


@pytest.fixture(params=["plaintext", "sealed"])
def s3_cognee(request, s3_server):
    """cognee on an S3 data root, encryption off (plaintext) or on (sealed).

    Each mode gets its own bucket prefix and system root, so the sealed run never
    meets the plaintext run's objects as legacy pass-through.
    """
    endpoint, client = s3_server
    mode = request.param
    root = pathlib.Path(tempfile.mkdtemp(prefix=f"cognee_s3_{mode}_"))

    import cognee  # cognee's import runs load_dotenv(override=True)

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "CACHE_BACKEND": "sqlite",
        "MOCK_EMBEDDING": "true",
        "TELEMETRY_DISABLED": "1",
        "STORAGE_BACKEND": "s3",
        "STORAGE_BUCKET_NAME": "bench",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_ENDPOINT_URL": endpoint,
        "DATA_ROOT_DIRECTORY": f"s3://bench/{mode}/data",
        # System (sqlite, cache) stays local: the DATA path is what is under test.
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "CACHE_ROOT_DIRECTORY": str(root / "cache"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "FILE_ENCRYPTION_ENABLED": "true" if mode == "sealed" else "false",
        "FILE_ENCRYPTION_KEY": base64.b64encode(b"\x42" * 32).decode(),
    }.items():
        mp.setenv(key, value)
    _clear_config_caches()

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

    yield {"mode": mode, "sealed": mode == "sealed", "counts": counts, "client": client}

    s3fs.S3FileSystem._call_s3 = original_async
    s3fs.S3FileSystem.call_s3 = original_sync
    mp.undo()
    _clear_config_caches()
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


async def _dataset_rows(dataset_name):
    from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    dataset = (await get_datasets_by_name([dataset_name], user.id))[0]
    return dataset, await get_dataset_data(dataset.id)


def _objects(client, mode):
    listing = client.list_objects_v2(Bucket="bench", Prefix=f"{mode}/data")
    return {
        item["Key"]: client.get_object(Bucket="bench", Key=item["Key"])["Body"].read()
        for item in listing.get("Contents", [])
    }


async def test_s3_objects_match_the_mode_and_read_back_as_the_added_documents(s3_cognee):
    import cognee
    from cognee.infrastructure.files.encryption.envelope import MAGIC
    from cognee.infrastructure.files.storage import get_file_storage
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    sealed, client, mode = s3_cognee["sealed"], s3_cognee["client"], s3_cognee["mode"]
    dataset_name = f"s3_{mode}_documents"

    await cognee.add([TEXT, str(PDF)], dataset_name)

    objects = _objects(client, mode)
    assert objects, "add() stored nothing under the S3 data root."
    for key, body in objects.items():
        if sealed:
            assert body.startswith(MAGIC), f"{key} is stored in plaintext."
            assert b"ZEBRA-7731" not in body and b"Artificial" not in body, (
                f"{key} leaks plaintext."
            )
        else:
            assert not body.startswith(MAGIC), f"{key} is sealed with encryption off."

    _, rows = await _dataset_rows(dataset_name)
    assert len(rows) == 2
    read_back = {}
    for row in rows:
        assert row.raw_data_location.startswith("s3://")
        async with open_data_file(row.raw_data_location, mode="rb") as plaintext:
            content = plaintext.read()
        assert not content.startswith(MAGIC), f"{row.name} read back as ciphertext."
        directory, _, name = row.raw_data_location.rpartition("/")
        size = await get_file_storage(directory).get_size(name)
        assert size == len(content), f"{row.name}: get_size {size} != plaintext {len(content)}"
        read_back[row.name] = content
    text_rows = [name for name in read_back if name.startswith("text_")]
    assert len(text_rows) == 1, f"expected one text item, got {sorted(read_back)}"
    assert read_back[text_rows[0]].decode("utf-8") == TEXT
    assert b"Artificial intelligence" in read_back["artificial-intelligence"]

    # Re-adding the same content: the content hash and the content-addressed object
    # names are computed on the plaintext, so no new row and no new object appears.
    # (A text item is re-stored under its existing name on every add, sealed or not.)
    await cognee.add([TEXT, str(PDF)], dataset_name)
    _, rows_after = await _dataset_rows(dataset_name)
    assert len(rows_after) == 2, "re-adding identical content created new rows"
    assert set(_objects(client, mode)) == set(objects), (
        "re-adding identical content stored new objects: the hash is not of the plaintext"
    )


async def test_raw_download_streams_plaintext_from_s3(s3_cognee):
    import httpx

    import cognee
    from cognee.api.client import app
    from cognee.infrastructure.files.encryption.envelope import MAGIC
    from cognee.modules.users.methods import get_authenticated_user, get_default_user

    mode = s3_cognee["mode"]
    dataset_name = f"s3_{mode}_download"
    await cognee.add([TEXT], dataset_name)
    dataset, rows = await _dataset_rows(dataset_name)

    # The auth posture is fixed when cognee is first imported, which another test
    # module may have done under the default (authentication required) posture;
    # the route under test is the download, not the login, so act as the owner.
    user = await get_default_user()
    app.dependency_overrides[get_authenticated_user] = lambda: user
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.get(f"/api/v1/datasets/{dataset.id}/data/{rows[0].id}/raw")
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)

    assert response.status_code == 200, response.text
    assert not response.content.startswith(MAGIC), "the download served the envelope"
    assert response.content.decode("utf-8") == TEXT


async def test_steady_state_add_stays_within_the_s3_call_budget(s3_cognee):
    import cognee

    counts, mode = s3_cognee["counts"], s3_cognee["mode"]

    # Run 1 warms content-addressed storage; run 2 is the steady state.
    await cognee.add(_uploads(), f"s3_{mode}_budget_run1")
    counts.clear()
    await cognee.add(_uploads(), f"s3_{mode}_budget_run2")

    per_file = {verb: count / FILES for verb, count in sorted(counts.items())}
    assert per_file.get("put_object", 0) >= 1, (
        f"S3 was not engaged by add() — counts: {per_file or '{}'}"
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
        f"[{mode}] S3 calls per file over budget: {over}; all counts: {per_file}. "
        "The encryption layer must add no S3 round trips on either path."
    )
