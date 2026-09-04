"""The add and remember endpoints accept a folder as multipart parts whose
filenames carry paths relative to the folder. The router writes the tree
server-side and passes the directory to add()/remember() in place of the
parts, after flat uploads and before raw_data entries. Labels and
external_metadata are rejected with a folder (a folder expands to many
records), and remember only does this for normal ingestion — the skills path
handles relative SKILL.md names itself.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cognee.tasks.ingestion.folder_uploads as folder_uploads
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.exceptions import InvalidFolderUploadError

add_pkg = importlib.import_module("cognee.api.v1.add")
remember_pkg = importlib.import_module("cognee.api.v1.remember")

USER_ID = str(uuid.uuid4())
MOCK_USER = SimpleNamespace(
    id=USER_ID, email="test@example.com", is_active=True, tenant_id=str(uuid.uuid4())
)
FOLDER_PARTS = [
    ("data", ("proj/pyproject.toml", b'[project]\nname = "proj"\n', "text/plain")),
    ("data", ("proj/src/app.py", b"def main():\n    pass\n", "text/plain")),
    ("data", ("proj/README.md", b"# proj\n", "text/markdown")),
]
FLAT_PART = ("data", ("notes.txt", b"plain notes", "text/plain"))


@pytest.fixture
def uploads_root(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    monkeypatch.setattr(folder_uploads, "folder_uploads_root", lambda: root)
    return root


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/api/v1/add")
    app.include_router(get_remember_router(), prefix="/api/v1/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def pipeline_run_completed():
    return PipelineRunCompleted(
        pipeline_run_id=uuid.uuid4(), dataset_id=uuid.uuid4(), dataset_name="test_dataset"
    )


def remember_completed():
    return SimpleNamespace(status="completed", to_dict=lambda: {"status": "completed"})


# ── add ─────────────────────────────────────────────────────────────────────


def test_add_materializes_the_folder_and_passes_its_directory(client, uploads_root):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=[FLAT_PART, *FOLDER_PARTS],
            data={"datasetName": "test_dataset", "raw_data": ["a note"]},
        )

        assert response.status_code == 200, response.text
        sent = mock_add.call_args.args[0]
        folder = uploads_root / USER_ID / "proj"
        # flat uploads, then the folder, then raw_data
        assert sent[0].filename == "notes.txt"
        assert sent[1] == str(folder)
        assert sent[2] == "a note"
        assert (folder / "pyproject.toml").exists()
        assert (folder / "src" / "app.py").read_bytes() == b"def main():\n    pass\n"
        assert (folder / "README.md").read_bytes() == b"# proj\n"


def test_add_rejects_labels_with_a_folder_upload(client, uploads_root):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        response = client.post(
            "/api/v1/add",
            files=FOLDER_PARTS,
            data={"datasetName": "test_dataset", "labels": '["code"]'},
        )

        assert response.status_code == 400
        assert "folder uploads" in response.json()["error"]
        mock_add.assert_not_awaited()


def test_add_rejects_unsafe_folder_names_and_writes_nothing(client, uploads_root):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        # No global CogneeApiError handler on the bare router app: the 400-class
        # error surfaces as the exception itself.
        with pytest.raises(InvalidFolderUploadError):
            client.post(
                "/api/v1/add",
                files=[*FOLDER_PARTS, ("data", ("../escape.py", b"x", "text/plain"))],
                data={"datasetName": "test_dataset"},
            )

        mock_add.assert_not_awaited()
        assert not uploads_root.exists()


# ── remember ────────────────────────────────────────────────────────────────


def test_remember_materializes_the_folder_for_normal_ingestion(client, uploads_root):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember", files=FOLDER_PARTS, data={"datasetName": "test_dataset"}
        )

        assert response.status_code == 200, response.text
        assert mock_remember.call_args.args[0] == [str(uploads_root / USER_ID / "proj")]


def test_remember_leaves_skills_uploads_alone(client, uploads_root):
    """content_type='skills' owns relative SKILL.md names; the folder path must not intercept."""
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember",
            files=[("data", ("my-skill/SKILL.md", b"# skill", "text/markdown"))],
            data={"datasetName": "test_dataset", "content_type": "skills"},
        )

        assert response.status_code == 200, response.text
        sent = mock_remember.call_args.args[0]
        assert sent[0].filename == "my-skill/SKILL.md"
        assert not uploads_root.exists()


def test_remember_rejects_external_metadata_with_a_folder_upload(client, uploads_root):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        response = client.post(
            "/api/v1/remember",
            files=FOLDER_PARTS,
            data={"datasetName": "test_dataset", "external_metadata": '[{"source": "crm"}]'},
        )

        assert response.status_code == 400
        assert "folder uploads" in response.json()["detail"]
        mock_remember.assert_not_awaited()
