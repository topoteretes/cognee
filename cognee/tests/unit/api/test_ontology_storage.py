"""The ontology store keeps its files in cognee's file storage.

Covers:
- an ``s3://`` data root is handed to the storage layer as it is, and no local
  folder named ``s3:`` appears (before the fix the store treated the URL as a
  relative local path)
- the layout on a local data root is unchanged, so earlier uploads are found
- a user id cannot leave the data root: it is refused before the storage layer is reached
- a storage failure is raised, not papered over with a local write
"""

import hashlib
import json
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.ontologies import ontologies as ontologies_module
from cognee.api.v1.ontologies.ontologies import OntologyService

OWL = (
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:owl="http://www.w3.org/2002/07/owl#">'
    b'<owl:Class rdf:about="http://example.org/Car"/></rdf:RDF>'
)
S3_DATA_ROOT = "s3://my-bucket/prefix/data"


class InMemoryStorage:
    """Stand-in for ``StorageManager``: objects in a dict, keyed by storage-relative path."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    async def store(self, file_path, data, overwrite=False):
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif hasattr(data, "read"):
            data = data.read()
        if overwrite or file_path not in self.objects:
            self.objects[file_path] = data
        return file_path

    @asynccontextmanager
    async def open(self, file_path, mode="r"):
        if file_path not in self.objects:
            raise FileNotFoundError(file_path)
        yield BytesIO(self.objects[file_path])

    async def file_exists(self, file_path):
        return file_path in self.objects

    async def is_file(self, file_path):
        return file_path in self.objects

    async def remove(self, file_path):
        self.objects.pop(file_path, None)


def _upload_file(content: bytes = OWL, filename: str = "cars.owl"):
    upload = SimpleNamespace(filename=filename, content_type="application/rdf+xml")
    upload.read = AsyncMock(return_value=content)
    return upload


def _storage_name(ontology_key: str) -> str:
    return hashlib.blake2s(ontology_key.encode("utf-8"), digest_size=16).hexdigest() + ".owl"


@pytest.fixture
def s3_store(monkeypatch, tmp_path):
    """An ontology store on an s3:// data root, with the storage layer replaced by a stand-in.

    Returns ``(service, storage, requested_roots)``. The working directory is a
    temp dir, so a stray local folder would be created there.
    """
    monkeypatch.chdir(tmp_path)
    storage = InMemoryStorage()
    requested_roots = []

    def fake_get_file_storage(storage_path):
        requested_roots.append(storage_path)
        return storage

    fake_config = SimpleNamespace(data_root_directory=S3_DATA_ROOT)
    with (
        patch.object(ontologies_module, "get_base_config", return_value=fake_config),
        patch.object(ontologies_module, "get_file_storage", fake_get_file_storage),
    ):
        yield OntologyService(), storage, requested_roots


@pytest.fixture
def local_store(tmp_path):
    fake_config = SimpleNamespace(data_root_directory=tmp_path)
    with patch.object(ontologies_module, "get_base_config", return_value=fake_config):
        yield OntologyService()


@pytest.mark.asyncio
async def test_s3_data_root_is_stored_through_the_storage_layer(s3_store, tmp_path):
    service, storage, requested_roots = s3_store
    user = SimpleNamespace(id="user-1")

    await service.upload_ontology("cars", _upload_file(), user)

    assert set(requested_roots) == {S3_DATA_ROOT}
    assert set(storage.objects) == {"user-1/metadata.json", f"user-1/{_storage_name('cars')}"}
    assert storage.objects[f"user-1/{_storage_name('cars')}"] == OWL
    assert not (tmp_path / "s3:").exists(), "the s3:// data root was used as a local path"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_s3_data_root_reads_lists_and_deletes_through_the_storage_layer(s3_store):
    service, storage, _ = s3_store
    user = SimpleNamespace(id="user-1")
    await service.upload_ontology("cars", _upload_file(), user)

    (ontology_file,) = await service.get_ontology_files(["cars"], user)
    assert ontology_file.read() == OWL
    assert (ontology_file.filename, ontology_file.rdf_format) == ("cars.owl", "xml")
    assert set(await service.list_ontologies(user)) == {"cars"}
    assert len(await service.load_ontology_graph("cars", user)) == 1

    await service.delete_ontology("cars", user)

    assert set(storage.objects) == {"user-1/metadata.json"}
    assert await service.list_ontologies(user) == {}
    with pytest.raises(ValueError, match="not found"):
        await service.get_ontology_files(["cars"], user)


@pytest.mark.asyncio
async def test_local_data_root_keeps_its_layout(local_store, tmp_path):
    """<data root>/<user id>/metadata.json and <blake2s of the key>.owl, as before."""
    user = SimpleNamespace(id="user-1")

    await local_store.upload_ontology("cars", _upload_file(), user)

    user_dir = tmp_path / "user-1"
    assert sorted(path.name for path in user_dir.iterdir()) == sorted(
        ["metadata.json", _storage_name("cars")]
    )
    assert (user_dir / _storage_name("cars")).read_bytes() == OWL
    index = json.loads((user_dir / "metadata.json").read_text(encoding="utf-8"))
    assert index["cars"]["filename"] == "cars.owl"
    assert index["cars"]["format"] == "xml"
    assert index["cars"]["size_bytes"] == len(OWL)


@pytest.mark.asyncio
async def test_ontology_uploaded_before_the_storage_layer_is_still_found(local_store, tmp_path):
    """Files written with plain local calls, by the store as it was, are read back."""
    user_dir = tmp_path / "user-1"
    user_dir.mkdir()
    (user_dir / _storage_name("cars")).write_bytes(OWL)
    with open(user_dir / "metadata.json", "w", encoding="utf-8") as index_file:
        json.dump(
            {"cars": {"filename": "cars.owl", "size_bytes": len(OWL), "uploaded_at": "2026-01-01"}},
            index_file,
            indent=2,
        )

    (ontology_file,) = await local_store.get_ontology_files(["cars"], SimpleNamespace(id="user-1"))

    assert ontology_file.read() == OWL
    assert ontology_file.filename == "cars.owl"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["..", "../other-user", "/etc", "user-1/../../other-user"])
async def test_user_id_cannot_leave_the_data_root(s3_store, user_id):
    """Refused before the storage layer is reached, so the rule holds for every backend."""
    service, storage, requested_roots = s3_store
    user = SimpleNamespace(id=user_id)

    with pytest.raises(ValueError, match="Invalid user id"):
        await service.list_ontologies(user)
    with pytest.raises(ValueError, match="Invalid user id"):
        await service.upload_ontology("cars", _upload_file(), user)
    with pytest.raises(ValueError, match="Invalid user id"):
        await service.delete_ontology("cars", user)

    assert requested_roots == []
    assert storage.objects == {}


@pytest.mark.asyncio
async def test_storage_failure_is_raised(s3_store, tmp_path):
    """No fallback to a local write when the storage layer fails."""
    service, storage, _ = s3_store
    storage.store = AsyncMock(side_effect=PermissionError("Access Denied"))

    with pytest.raises(PermissionError):
        await service.upload_ontology("cars", _upload_file(), SimpleNamespace(id="user-1"))

    assert list(tmp_path.iterdir()) == []
