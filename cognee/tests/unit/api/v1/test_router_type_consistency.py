"""
Tests for API router type consistency with core function signatures.

Verifies fixes for issue #2049:
- add/update routers accept Optional[List[UploadFile]] (not bare List[UploadFile])
- cognify/search DTOs accept single string/UUID values (auto-normalized to lists)
"""

import pytest
from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models import PipelineRunCompleted
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.update.routers.get_update_router import get_update_router
from cognee.api.v1.cognify.routers.get_cognify_router import (
    get_cognify_router,
    CognifyPayloadDTO,
)
from cognee.api.v1.search.routers.get_search_router import (
    get_search_router,
    SearchPayloadDTO,
)


MOCK_USER = SimpleNamespace(
    id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4()
)
MOCK_DATASET_ID = uuid4()
MOCK_PIPELINE_RUN_ID = uuid4()


def _make_completed(**kwargs):
    defaults = dict(
        pipeline_run_id=MOCK_PIPELINE_RUN_ID,
        dataset_id=MOCK_DATASET_ID,
        dataset_name="test_dataset",
        status="completed",
    )
    defaults.update(kwargs)
    return PipelineRunCompleted(**defaults)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/add")
    app.include_router(get_update_router(), prefix="/update")
    app.include_router(get_cognify_router(), prefix="/cognify")
    app.include_router(get_search_router(), prefix="/search")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# CognifyPayloadDTO validator tests
# ---------------------------------------------------------------------------


class TestCognifyPayloadDTOValidators:
    """Test that CognifyPayloadDTO normalizes single values to lists."""

    def test_single_dataset_string_normalized_to_list(self):
        dto = CognifyPayloadDTO(datasets="my_dataset")
        assert dto.datasets == ["my_dataset"]

    def test_dataset_list_passed_through(self):
        dto = CognifyPayloadDTO(datasets=["ds1", "ds2"])
        assert dto.datasets == ["ds1", "ds2"]

    def test_datasets_none_remains_none(self):
        dto = CognifyPayloadDTO(datasets=None)
        assert dto.datasets is None

    def test_single_dataset_id_uuid_normalized_to_list(self):
        uid = uuid4()
        dto = CognifyPayloadDTO(dataset_ids=uid)
        assert dto.dataset_ids == [uid]

    def test_single_dataset_id_string_normalized_to_list(self):
        uid = uuid4()
        dto = CognifyPayloadDTO(dataset_ids=str(uid))
        assert dto.dataset_ids == [uid]

    def test_dataset_ids_list_passed_through(self):
        uids = [uuid4(), uuid4()]
        dto = CognifyPayloadDTO(dataset_ids=uids)
        assert dto.dataset_ids == uids

    def test_dataset_ids_none_remains_none(self):
        dto = CognifyPayloadDTO(dataset_ids=None)
        assert dto.dataset_ids is None


# ---------------------------------------------------------------------------
# SearchPayloadDTO validator tests
# ---------------------------------------------------------------------------


class TestSearchPayloadDTOValidators:
    """Test that SearchPayloadDTO normalizes single values to lists."""

    def test_single_dataset_string_normalized_to_list(self):
        dto = SearchPayloadDTO(datasets="my_dataset", query="test")
        assert dto.datasets == ["my_dataset"]

    def test_dataset_list_passed_through(self):
        dto = SearchPayloadDTO(datasets=["ds1", "ds2"], query="test")
        assert dto.datasets == ["ds1", "ds2"]

    def test_datasets_none_remains_none(self):
        dto = SearchPayloadDTO(query="test")
        assert dto.datasets is None

    def test_single_dataset_id_uuid_normalized_to_list(self):
        uid = uuid4()
        dto = SearchPayloadDTO(dataset_ids=uid, query="test")
        assert dto.dataset_ids == [uid]

    def test_single_dataset_id_string_normalized_to_list(self):
        uid = uuid4()
        dto = SearchPayloadDTO(dataset_ids=str(uid), query="test")
        assert dto.dataset_ids == [uid]

    def test_dataset_ids_list_passed_through(self):
        uids = [uuid4(), uuid4()]
        dto = SearchPayloadDTO(dataset_ids=uids, query="test")
        assert dto.dataset_ids == uids

    def test_dataset_ids_none_remains_none(self):
        dto = SearchPayloadDTO(query="test")
        assert dto.dataset_ids is None


# ---------------------------------------------------------------------------
# Add endpoint: data parameter is Optional
# ---------------------------------------------------------------------------


class TestAddDataOptional:
    """Test that the add endpoint properly treats data as Optional."""

    def test_add_no_data_with_dataset_name_does_not_422(self, client):
        """Sending no files should not cause a 422 validation error.

        Before the fix, ``data: List[UploadFile] = File(default=None)`` was
        misleading because the type annotation was not Optional, which could
        confuse API documentation generators and linters.  The fix changes
        the annotation to ``Optional[List[UploadFile]]``.
        """
        import cognee.api.v1.add as add_pkg

        add_pkg.add = AsyncMock(return_value=_make_completed())

        resp = client.post("/add", data={"datasetName": "test_dataset"})
        # Should not get a 422 validation error
        assert resp.status_code != 422

    def test_add_with_file_still_works(self, client):
        """Standard file upload path should remain functional."""
        import cognee.api.v1.add as add_pkg

        add_pkg.add = AsyncMock(return_value=_make_completed())

        resp = client.post(
            "/add",
            data={"datasetName": "test_dataset"},
            files={"data": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Update endpoint: data parameter is Optional
# ---------------------------------------------------------------------------


class TestUpdateDataOptional:
    """Test that the update endpoint properly treats data as Optional."""

    def test_update_no_data_does_not_422(self, client):
        """Sending no files should not cause a 422 validation error."""
        import cognee.api.v1.update as update_pkg

        update_pkg.update = AsyncMock(return_value={"run": _make_completed()})

        resp = client.patch(
            "/update",
            params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
        )
        assert resp.status_code != 422

    def test_update_with_file_still_works(self, client):
        """Standard file upload path should remain functional."""
        import cognee.api.v1.update as update_pkg

        update_pkg.update = AsyncMock(return_value={"run": _make_completed()})

        resp = client.patch(
            "/update",
            params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
            files={"data": ("test.txt", b"updated content", "text/plain")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cognify endpoint: single-value parameters via API
# ---------------------------------------------------------------------------


class TestCognifySingleValues:
    """Test that the cognify endpoint accepts single string/UUID values."""

    def test_cognify_single_dataset_string(self, client):
        """API should accept a single dataset name string, not just a list."""
        import cognee.api.v1.cognify as cognify_pkg

        completed = _make_completed()
        cognify_pkg.cognify = AsyncMock(return_value={str(MOCK_DATASET_ID): completed})

        resp = client.post(
            "/cognify",
            json={"datasets": "my_dataset"},
        )
        assert resp.status_code == 200

    def test_cognify_dataset_list_still_works(self, client):
        """Existing list-based usage should continue to work."""
        import cognee.api.v1.cognify as cognify_pkg

        completed = _make_completed()
        cognify_pkg.cognify = AsyncMock(return_value={str(MOCK_DATASET_ID): completed})

        resp = client.post(
            "/cognify",
            json={"datasets": ["ds1", "ds2"]},
        )
        assert resp.status_code == 200

    def test_cognify_single_dataset_id_uuid(self, client):
        """API should accept a single UUID string for dataset_ids."""
        import cognee.api.v1.cognify as cognify_pkg

        uid = uuid4()
        completed = _make_completed()
        cognify_pkg.cognify = AsyncMock(return_value={str(uid): completed})

        resp = client.post(
            "/cognify",
            json={"dataset_ids": str(uid)},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Search endpoint: single-value parameters via API
# ---------------------------------------------------------------------------


class TestSearchSingleValues:
    """Test that the search endpoint accepts single string/UUID values."""

    def test_search_single_dataset_string(self, client):
        """API should accept a single dataset name string, not just a list."""
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(return_value=[])

        resp = client.post(
            "/search",
            json={"datasets": "my_dataset", "query": "test query"},
        )
        assert resp.status_code == 200

    def test_search_dataset_list_still_works(self, client):
        """Existing list-based usage should continue to work."""
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(return_value=[])

        resp = client.post(
            "/search",
            json={"datasets": ["ds1", "ds2"], "query": "test query"},
        )
        assert resp.status_code == 200

    def test_search_single_dataset_id_uuid(self, client):
        """API should accept a single UUID string for dataset_ids."""
        import cognee.api.v1.search as search_pkg

        search_pkg.search = AsyncMock(return_value=[])

        uid = uuid4()
        resp = client.post(
            "/search",
            json={"dataset_ids": str(uid), "query": "test query"},
        )
        assert resp.status_code == 200
