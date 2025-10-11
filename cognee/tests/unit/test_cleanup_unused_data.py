"""Unit tests for cleanup_unused_data functionality."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from cognee.tasks.cleanup_unused_data import cleanup_unused_data
from cognee.modules.data.access_tracking import AccessTracker


class TestCleanupUnusedData:
    """Test suite for cleanup_unused_data task."""

    @pytest.fixture
    def mock_db_interface(self):
        """Create a mock database interface."""
        db = Mock()
        db.get_datasets_older_than = AsyncMock(return_value=[])
        db.get_documents_older_than = AsyncMock(return_value=[])
        db.delete_dataset = AsyncMock()
        db.delete_document = AsyncMock()
        return db

    @pytest.fixture
    def mock_access_tracker(self):
        """Create a mock access tracker."""
        tracker = Mock(spec=AccessTracker)
        tracker.get_last_access = AsyncMock(return_value=None)
        tracker.mark_accessed = AsyncMock()
        return tracker

    @pytest.mark.asyncio
    async def test_cleanup_with_no_old_data(self, mock_db_interface, mock_access_tracker):
        """Test cleanup when no data is older than threshold."""
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30)
                
                assert result['datasets_deleted'] == 0
                assert result['documents_deleted'] == 0
                assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_datasets(self, mock_db_interface, mock_access_tracker):
        """Test that old datasets are properly deleted."""
        old_date = datetime.now() - timedelta(days=40)
        mock_datasets = [
            {'id': 'dataset1', 'name': 'Old Dataset 1', 'last_modified': old_date},
            {'id': 'dataset2', 'name': 'Old Dataset 2', 'last_modified': old_date}
        ]
        mock_db_interface.get_datasets_older_than.return_value = mock_datasets
        
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30)
                
                assert result['datasets_deleted'] == 2
                assert mock_db_interface.delete_dataset.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_respects_access_tracking(self, mock_db_interface, mock_access_tracker):
        """Test that recently accessed data is not deleted despite old modification date."""
        old_date = datetime.now() - timedelta(days=40)
        recent_access = datetime.now() - timedelta(days=5)
        
        mock_datasets = [
            {'id': 'dataset1', 'name': 'Recently Accessed', 'last_modified': old_date}
        ]
        mock_db_interface.get_datasets_older_than.return_value = mock_datasets
        mock_access_tracker.get_last_access.return_value = recent_access
        
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30)
                
                assert result['datasets_deleted'] == 0
                assert mock_db_interface.delete_dataset.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_invalid_threshold(self, mock_db_interface, mock_access_tracker):
        """Test that cleanup raises error with invalid threshold."""
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                with pytest.raises(ValueError):
                    await cleanup_unused_data(days_threshold=-1)

    @pytest.mark.asyncio
    async def test_cleanup_handles_deletion_errors(self, mock_db_interface, mock_access_tracker):
        """Test that cleanup handles deletion errors gracefully."""
        old_date = datetime.now() - timedelta(days=40)
        mock_datasets = [
            {'id': 'dataset1', 'name': 'Dataset 1', 'last_modified': old_date}
        ]
        mock_db_interface.get_datasets_older_than.return_value = mock_datasets
        mock_db_interface.delete_dataset.side_effect = Exception("Database error")
        
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30)
                
                assert result['status'] == 'partial_success' or result['status'] == 'error'
                assert 'errors' in result

    @pytest.mark.asyncio
    async def test_cleanup_dry_run_mode(self, mock_db_interface, mock_access_tracker):
        """Test cleanup in dry-run mode doesn't actually delete data."""
        old_date = datetime.now() - timedelta(days=40)
        mock_datasets = [
            {'id': 'dataset1', 'name': 'Old Dataset', 'last_modified': old_date}
        ]
        mock_db_interface.get_datasets_older_than.return_value = mock_datasets
        
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30, dry_run=True)
                
                assert result['datasets_to_delete'] == 1
                assert mock_db_interface.delete_dataset.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_documents(self, mock_db_interface, mock_access_tracker):
        """Test cleanup handles documents as well as datasets."""
        old_date = datetime.now() - timedelta(days=40)
        mock_documents = [
            {'id': 'doc1', 'name': 'Old Doc 1', 'last_modified': old_date},
            {'id': 'doc2', 'name': 'Old Doc 2', 'last_modified': old_date},
            {'id': 'doc3', 'name': 'Old Doc 3', 'last_modified': old_date}
        ]
        mock_db_interface.get_documents_older_than.return_value = mock_documents
        
        with patch('cognee.tasks.cleanup_unused_data.get_db_interface', return_value=mock_db_interface):
            with patch('cognee.tasks.cleanup_unused_data.AccessTracker', return_value=mock_access_tracker):
                result = await cleanup_unused_data(days_threshold=30)
                
                assert result['documents_deleted'] == 3
                assert mock_db_interface.delete_document.call_count == 3
