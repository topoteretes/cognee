"""The add and remember routers hand folder uploads to the pipeline untouched.

A folder arrives as multipart parts whose filenames carry paths relative to
the folder. The router does not materialize anything — resolve_data_directories
does that inside add(), after authorization and keyed by dataset — but it
validates the names and rejects labels/external_metadata with a folder so a
bad request is a 400 rather than a pipeline error. remember only validates for
normal ingestion; the skills path owns relative SKILL.md names.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.ingestion.exceptions import InvalidFolderUploadError

add_pkg = importlib.import_module("cognee.api.v1.add")
remember_pkg = importlib.import_module("cognee.api.v1.remember")

MOCK_USER = SimpleNamespace(
    id=str(uuid.uuid4()), email="test@example.com", is_active=True, tenant_id=str(uuid.uuid4())
)
FOLDER_PARTS = [
    ("data", ("proj/pyproject.toml", b'[project]\nname = "proj"\n', "text/plain")),
    ("data", ("proj/src/app.py", b"def main():\n    pass\n", "text/plain")),
    ("data", ("proj/README.md", b"# proj\n", "text/markdown")),
]
FLAT_PART = ("data", ("notes.txt", b"plain notes", "text/plain"))


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


def test_add_passes_folder_parts_through_with_their_relative_names(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=[FLAT_PART, *FOLDER_PARTS],
            data={"datasetName": "test_dataset", "raw_data": ["a note"]},
        )

        assert response.status_code == 200, response.text
        sent = mock_add.call_args.args[0]
        assert [getattr(item, "filename", item) for item in sent] == [
            "notes.txt",
            "proj/pyproject.toml",
            "proj/src/app.py",
            "proj/README.md",
            "a note",
        ]


def test_add_rejects_labels_with_a_folder_upload(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        # No global CogneeApiError handler on the bare router app: the 400-class
        # error surfaces as the exception itself.
        with pytest.raises(InvalidFolderUploadError, match="labels"):
            client.post(
                "/api/v1/add",
                files=FOLDER_PARTS,
                data={"datasetName": "test_dataset", "labels": '["code", "code", "code"]'},
            )

        mock_add.assert_not_awaited()


def test_add_rejects_unsafe_folder_names_before_the_pipeline(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        with pytest.raises(InvalidFolderUploadError):
            client.post(
                "/api/v1/add",
                files=[*FOLDER_PARTS, ("data", ("../escape.py", b"x", "text/plain"))],
                data={"datasetName": "test_dataset"},
            )

        mock_add.assert_not_awaited()


def test_add_labels_still_pair_with_flat_uploads(client):
    with patch.object(add_pkg, "add", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = pipeline_run_completed()

        response = client.post(
            "/api/v1/add",
            files=[FLAT_PART],
            data={"datasetName": "test_dataset", "labels": '["notes"]'},
        )

        assert response.status_code == 200, response.text
        assert mock_add.call_args.args[0][0].label == "notes"


# ── remember ────────────────────────────────────────────────────────────────


def test_remember_passes_folder_parts_through(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember", files=FOLDER_PARTS, data={"datasetName": "test_dataset"}
        )

        assert response.status_code == 200, response.text
        sent = mock_remember.call_args.args[0]
        assert [item.filename for item in sent] == [
            "proj/pyproject.toml",
            "proj/src/app.py",
            "proj/README.md",
        ]


def test_remember_does_not_validate_skills_uploads_as_folders(client):
    """content_type='skills' owns relative SKILL.md names, including ones the folder rules
    would reject as unsafe."""
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        mock_remember.return_value = remember_completed()

        response = client.post(
            "/api/v1/remember",
            files=[("data", ("my-skill/SKILL.md", b"# skill", "text/markdown"))],
            data={"datasetName": "test_dataset", "content_type": "skills"},
        )

        assert response.status_code == 200, response.text
        assert mock_remember.call_args.args[0][0].filename == "my-skill/SKILL.md"


def test_remember_rejects_external_metadata_with_a_folder_upload(client):
    with patch.object(remember_pkg, "remember", new_callable=AsyncMock) as mock_remember:
        with pytest.raises(InvalidFolderUploadError, match="external_metadata"):
            client.post(
                "/api/v1/remember",
                files=FOLDER_PARTS,
                data={
                    "datasetName": "test_dataset",
                    "external_metadata": '[{"source": "crm"}, null, null]',
                },
            )

        mock_remember.assert_not_awaited()
