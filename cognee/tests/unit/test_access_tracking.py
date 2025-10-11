"""Unit tests for access tracking functionality."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from cognee.modules.data.access_tracking import AccessTracker


class TestAccessTracker:
    """Test suite for AccessTracker class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database connection."""
        db = Mock()
        db.execute = AsyncMock()
        db.fetchone = AsyncMock(return_value=None)
        db.fetchall = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def tracker(self, mock_db):
        """Create an AccessTracker instance with mock database."""
        with patch('cognee.modules.data.access_tracking.get_db_connection', return_value=mock_db):
            return AccessTracker()

    @pytest.mark.asyncio
    async def test_mark_accessed_creates_entry(self, tracker, mock_db):
        """Test that marking a resource as accessed creates a log entry."""
        resource_id = 'dataset_123'
        resource_type = 'dataset'
        
        await tracker.mark_accessed(resource_id, resource_type)
        
        # Verify database insert was called
        assert mock_db.execute.called
        call_args = mock_db.execute.call_args[0][0]
        assert 'INSERT' in call_args.upper()
        assert 'access_logs' in call_args.lower()

    @pytest.mark.asyncio
    async def test_get_last_access_returns_timestamp(self, tracker, mock_db):
        """Test retrieving last access timestamp for a resource."""
        resource_id = 'dataset_123'
        expected_time = datetime.now() - timedelta(days=5)
        
        mock_db.fetchone.return_value = {'last_accessed': expected_time}
        
        result = await tracker.get_last_access(resource_id, 'dataset')
        
        assert result == expected_time
        assert mock_db.execute.called

    @pytest.mark.asyncio
    async def test_get_last_access_returns_none_for_new_resource(self, tracker, mock_db):
        """Test that get_last_access returns None for resources never accessed."""
        mock_db.fetchone.return_value = None
        
        result = await tracker.get_last_access('never_accessed', 'dataset')
        
        assert result is None

    @pytest.mark.asyncio
    async def test_mark_accessed_updates_existing_entry(self, tracker, mock_db):
        """Test that marking an already-accessed resource updates the timestamp."""
        resource_id = 'dataset_123'
        old_time = datetime.now() - timedelta(days=10)
        
        mock_db.fetchone.return_value = {'last_accessed': old_time}
        
        await tracker.mark_accessed(resource_id, 'dataset')
        
        # Should call UPDATE instead of INSERT
        call_args = str(mock_db.execute.call_args)
        assert 'UPDATE' in call_args.upper() or 'INSERT' in call_args.upper()

    @pytest.mark.asyncio
    async def test_get_unused_resources(self, tracker, mock_db):
        """Test retrieving resources not accessed within threshold."""
        threshold_days = 30
        old_resources = [
            {'resource_id': 'dataset_1', 'resource_type': 'dataset', 'last_accessed': datetime.now() - timedelta(days=45)},
            {'resource_id': 'dataset_2', 'resource_type': 'dataset', 'last_accessed': datetime.now() - timedelta(days=60)}
        ]
        
        mock_db.fetchall.return_value = old_resources
        
        result = await tracker.get_unused_resources(threshold_days)
        
        assert len(result) == 2
        assert all(r['resource_id'] in ['dataset_1', 'dataset_2'] for r in result)

    @pytest.mark.asyncio
    async def test_mark_accessed_with_invalid_resource_type(self, tracker):
        """Test that marking with invalid resource type raises error."""
        with pytest.raises(ValueError):
            await tracker.mark_accessed('resource_123', 'invalid_type')

    @pytest.mark.asyncio
    async def test_mark_accessed_with_empty_resource_id(self, tracker):
        """Test that marking with empty resource ID raises error."""
        with pytest.raises(ValueError):
            await tracker.mark_accessed('', 'dataset')

    @pytest.mark.asyncio
    async def test_get_access_count(self, tracker, mock_db):
        """Test getting the number of times a resource was accessed."""
        resource_id = 'dataset_123'
        expected_count = 5
        
        mock_db.fetchone.return_value = {'access_count': expected_count}
        
        result = await tracker.get_access_count(resource_id, 'dataset')
        
        assert result == expected_count

    @pytest.mark.asyncio
    async def test_clear_access_logs(self, tracker, mock_db):
        """Test clearing access logs for a specific resource."""
        resource_id = 'dataset_123'
        
        await tracker.clear_access_logs(resource_id, 'dataset')
        
        # Verify DELETE was called
        assert mock_db.execute.called
        call_args = mock_db.execute.call_args[0][0]
        assert 'DELETE' in call_args.upper()

    @pytest.mark.asyncio
    async def test_get_most_accessed_resources(self, tracker, mock_db):
        """Test retrieving most frequently accessed resources."""
        most_accessed = [
            {'resource_id': 'dataset_1', 'access_count': 100},
            {'resource_id': 'dataset_2', 'access_count': 75},
            {'resource_id': 'dataset_3', 'access_count': 50}
        ]
        
        mock_db.fetchall.return_value = most_accessed
        
        result = await tracker.get_most_accessed_resources(limit=3)
        
        assert len(result) == 3
        assert result[0]['access_count'] >= result[1]['access_count']
        assert result[1]['access_count'] >= result[2]['access_count']

    @pytest.mark.asyncio
    async def test_mark_accessed_handles_db_errors(self, tracker, mock_db):
        """Test that database errors during mark_accessed are handled gracefully."""
        mock_db.execute.side_effect = Exception("Database connection lost")
        
        with pytest.raises(Exception):
            await tracker.mark_accessed('dataset_123', 'dataset')

    @pytest.mark.asyncio
    async def test_batch_mark_accessed(self, tracker, mock_db):
        """Test marking multiple resources as accessed in batch."""
        resources = [
            ('dataset_1', 'dataset'),
            ('dataset_2', 'dataset'),
            ('doc_1', 'document')
        ]
        
        await tracker.batch_mark_accessed(resources)
        
        # Should have been called for each resource
        assert mock_db.execute.call_count >= len(resources)

    @pytest.mark.asyncio
    async def test_get_access_history(self, tracker, mock_db):
        """Test retrieving complete access history for a resource."""
        resource_id = 'dataset_123'
        history = [
            {'timestamp': datetime.now() - timedelta(days=1), 'action': 'read'},
            {'timestamp': datetime.now() - timedelta(days=5), 'action': 'read'},
            {'timestamp': datetime.now() - timedelta(days=10), 'action': 'read'}
        ]
        
        mock_db.fetchall.return_value = history
        
        result = await tracker.get_access_history(resource_id, 'dataset')
        
        assert len(result) == 3
        assert all('timestamp' in entry for entry in result)
