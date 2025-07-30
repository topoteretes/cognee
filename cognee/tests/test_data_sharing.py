"""
Tests for User Data Sharing Pipeline (Kuzu + LanceDB → Kuzu + LanceDB)
"""
import json
import pytest
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from cognee.api.client import app
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create mock user"""
    return User(
        id=uuid4(),
        email="test@example.com",
        tenant_id=uuid4(),
        is_active=True,
        is_verified=True
    )


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    return Dataset(
        id=uuid4(),
        name="test_dataset",
        owner_id=uuid4()
    )


@pytest.fixture
def sample_export_data():
    """Sample export data structure"""
    dataset_id = str(uuid4())
    user_id = str(uuid4())
    
    return {
        "dataset_id": dataset_id,
        "metadata": {
            "dataset": {
                "id": dataset_id,
                "name": "test_dataset",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": None,
                "owner_id": user_id
            },
            "data_items": [
                {
                    "id": str(uuid4()),
                    "name": "test_file.txt",
                    "extension": "txt",
                    "mime_type": "text/plain",
                    "content_hash": "abc123",
                    "external_metadata": {},
                    "node_set": None,
                    "token_count": 100,
                    "data_size": 1024,
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": None
                }
            ]
        },
        "graph_data": {
            "nodes": [
                {
                    "id": str(uuid4()),
                    "data": {
                        "name": "Test Node",
                        "type": "Document",
                        "properties": "{\"content\": \"test\"}"
                    }
                }
            ],
            "edges": [
                {
                    "from_node": str(uuid4()),
                    "to_node": str(uuid4()),
                    "edge_label": "CONTAINS",
                    "data": {
                        "relationship_name": "CONTAINS",
                        "properties": "{\"weight\": 1.0}"
                    }
                }
            ],
            "node_count": 1,
            "edge_count": 1
        },
        "vector_data": {
            "collections": {
                "test_collection": {
                    "name": "test_collection",
                    "schema": "vector_schema",
                    "data": "vector_data",
                    "metadata": {}
                }
            },
            "collection_count": 1
        },
        "metastore_data": {
            "permissions": [
                {
                    "id": str(uuid4()),
                    "principal_id": user_id,
                    "permission_id": str(uuid4()),
                    "created_at": "2023-01-01T00:00:00"
                }
            ],
            "dataset_database_config": {
                "owner_id": user_id,
                "vector_database_name": "test_vector_db",
                "graph_database_name": "test_graph_db",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": None
            }
        },
        "created_at": "2023-01-01T00:00:00",
        "source_user_id": user_id,
        "version": "1.0"
    }


class TestDataExportEndpoint:
    """Test the /v1/data/export/{dataset_id} endpoint"""
    
    @patch('cognee.modules.users.methods.get_authenticated_user')
    @patch('cognee.modules.users.permissions.methods.check_permission_on_dataset')
    @patch('cognee.modules.data.methods.export_dataset_data')
    def test_export_dataset_success(
        self, 
        mock_export, 
        mock_check_permission, 
        mock_get_user,
        client, 
        mock_user, 
        sample_export_data
    ):
        """Test successful dataset export"""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_check_permission.return_value = None  # No exception = permission granted
        mock_export.return_value = sample_export_data
        
        dataset_id = uuid4()
        
        # Make request
        response = client.get(f"/api/v1/data/export/{dataset_id}")
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["dataset_id"] == sample_export_data["dataset_id"]
        assert response_data["version"] == "1.0"
        assert "graph_data" in response_data
        assert "vector_data" in response_data
        assert "metastore_data" in response_data
        
        # Verify mocks were called
        mock_check_permission.assert_called_once_with(mock_user, "share", dataset_id)
        mock_export.assert_called_once_with(dataset_id, mock_user)
    
    @patch('cognee.modules.users.methods.get_authenticated_user')
    @patch('cognee.modules.users.permissions.methods.check_permission_on_dataset')
    def test_export_dataset_permission_denied(
        self, 
        mock_check_permission, 
        mock_get_user,
        client, 
        mock_user
    ):
        """Test export with insufficient permissions"""
        from cognee.modules.users.exceptions import PermissionDeniedError
        
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_check_permission.side_effect = PermissionDeniedError("No share permission")
        
        dataset_id = uuid4()
        
        # Make request
        response = client.get(f"/api/v1/data/export/{dataset_id}")
        
        # Verify response
        assert response.status_code == 500  # Permission error gets wrapped as 500
        assert "error" in response.json()["detail"].lower()


class TestDataImportEndpoint:
    """Test the /v1/data/import endpoint"""
    
    @patch('cognee.modules.users.methods.get_authenticated_user')
    @patch('cognee.modules.data.methods.import_dataset_data')
    def test_import_dataset_success(
        self, 
        mock_import, 
        mock_get_user,
        client, 
        mock_user, 
        sample_export_data
    ):
        """Test successful dataset import"""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_import.return_value = {
            "success": True,
            "dataset_id": uuid4(),
            "message": "Successfully imported dataset 'test_dataset_imported'",
            "imported_nodes": 5,
            "imported_edges": 3,
            "imported_vectors": 100
        }
        
        # Prepare test data
        import_request = {
            "target_dataset_name": "imported_dataset",
            "preserve_relationships": True
        }
        
        # Create transfer file content
        transfer_content = json.dumps(sample_export_data).encode('utf-8')
        
        # Make request with multipart form data
        files = {
            "transfer_file": ("transfer.json", transfer_content, "application/json")
        }
        data = {
            "import_request": json.dumps(import_request)
        }
        
        response = client.post("/api/v1/data/import", files=files, data=data)
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["success"] is True
        assert response_data["imported_nodes"] == 5
        assert response_data["imported_edges"] == 3
        assert response_data["imported_vectors"] == 100
        assert "dataset_id" in response_data
        
        # Verify mock was called
        mock_import.assert_called_once()
    
    @patch('cognee.modules.users.methods.get_authenticated_user')
    def test_import_dataset_invalid_file(
        self, 
        mock_get_user,
        client, 
        mock_user
    ):
        """Test import with invalid file format"""
        # Setup mocks
        mock_get_user.return_value = mock_user
        
        # Prepare invalid file
        files = {
            "transfer_file": ("transfer.txt", b"invalid content", "text/plain")
        }
        data = {
            "import_request": json.dumps({
                "target_dataset_name": "test",
                "preserve_relationships": True
            })
        }
        
        response = client.post("/api/v1/data/import", files=files, data=data)
        
        # Verify response
        assert response.status_code == 400
        assert "invalid transfer file format" in response.json()["detail"].lower()


class TestDataSharingIntegration:
    """Integration tests for the complete data sharing pipeline"""
    
    @patch('cognee.modules.users.methods.get_authenticated_user')
    @patch('cognee.modules.users.permissions.methods.check_permission_on_dataset')
    @patch('cognee.modules.data.methods.export_dataset_data')
    @patch('cognee.modules.data.methods.import_dataset_data')
    def test_full_export_import_pipeline(
        self,
        mock_import,
        mock_export,
        mock_check_permission,
        mock_get_user,
        client,
        sample_export_data
    ):
        """Test the complete export-import pipeline"""
        # Create two different users
        source_user = User(id=uuid4(), email="source@example.com")
        target_user = User(id=uuid4(), email="target@example.com")
        
        dataset_id = uuid4()
        
        # Test export phase
        mock_get_user.return_value = source_user
        mock_check_permission.return_value = None
        mock_export.return_value = sample_export_data
        
        export_response = client.get(f"/api/v1/data/export/{dataset_id}")
        assert export_response.status_code == 200
        
        exported_data = export_response.json()
        
        # Test import phase
        mock_get_user.return_value = target_user
        mock_import.return_value = {
            "success": True,
            "dataset_id": uuid4(),
            "message": "Successfully imported dataset",
            "imported_nodes": 1,
            "imported_edges": 1,
            "imported_vectors": 1
        }
        
        # Prepare import request
        import_request = {
            "target_dataset_name": "imported_dataset",
            "preserve_relationships": True
        }
        
        transfer_content = json.dumps(exported_data).encode('utf-8')
        files = {
            "transfer_file": ("transfer.json", transfer_content, "application/json")
        }
        data = {
            "import_request": json.dumps(import_request)
        }
        
        import_response = client.post("/api/v1/data/import", files=files, data=data)
        assert import_response.status_code == 200
        
        import_result = import_response.json()
        assert import_result["success"] is True
        
        # Verify data transformation occurred
        mock_export.assert_called_once_with(dataset_id, source_user)
        mock_import.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
