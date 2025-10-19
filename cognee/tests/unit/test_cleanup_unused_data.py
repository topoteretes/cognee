"""Unit tests for schema-agnostic cleanup_unused_data functionality."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy import MetaData, Table, Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from cognee.tasks.cleanup_unused_data import (
    cleanup_unused_data,
    get_cleanup_preview,
    get_data_usage_statistics,
    CleanupResult
)
from cognee.modules.data.cleanup_unused_data import (
    discover_tracked_tables,
    delete_unused_data,
    get_unused_data_counts,
    get_table_statistics
)
from cognee.modules.data.access_tracking import (
    update_last_accessed,
    mark_entity_accessed,
    bulk_update_last_accessed
)


class TestDiscoverTrackedTables:
    """Tests for table discovery functionality."""

    def test_discover_default_graph_tables(self):
        """Test discovering tables in default graph schema."""
        metadata = MetaData()
        
        # Create mock tables with last_accessed column
        Table('document_chunks', metadata,
              Column('id', PGUUID, primary_key=True),
              Column('last_accessed', DateTime))
        Table('entities', metadata,
              Column('id', PGUUID, primary_key=True),
              Column('last_accessed', DateTime))
        # Table without last_accessed should not be included
        Table('other_table', metadata,
              Column('id', PGUUID, primary_key=True))
        
        tracked = discover_tracked_tables(metadata)
        
        assert len(tracked) == 2
        tracked_names = [t.name for t in tracked]
        assert 'document_chunks' in tracked_names
        assert 'entities' in tracked_names
        assert 'other_table' not in tracked_names

    def test_discover_custom_graph_tables(self):
        """Test discovering tables in custom graph schema."""
        metadata = MetaData()
        
        # Create tables in custom schema
        Table('custom_nodes', metadata,
              Column('id', PGUUID, primary_key=True),
              Column('last_accessed', DateTime),
              schema='my_custom_graph')
        Table('custom_edges', metadata,
              Column('id', PGUUID, primary_key=True),
              Column('last_accessed', DateTime),
              schema='my_custom_graph')
        # Table in different schema
        Table('other_nodes', metadata,
              Column('id', PGUUID, primary_key=True),
              Column('last_accessed', DateTime),
              schema='other_graph')
        
        tracked = discover_tracked_tables(metadata, schema='my_custom_graph')
        
        assert len(tracked) == 2
        tracked_names = [t.name for t in tracked]
        assert 'custom_nodes' in tracked_names
        assert 'custom_edges' in tracked_names
        assert 'other_nodes' not in tracked_names


class TestCleanupUnusedData:
    """Test suite for cleanup_unused_data task."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock database adapter."""
        adapter = Mock()
        session = AsyncMock()
        adapter.get_async_session = MagicMock(return_value=session)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return adapter, session

    @pytest.mark.asyncio
    async def test_cleanup_dry_run_default_graph(self, mock_adapter):
        """Test dry run cleanup on default graph."""
        adapter, session = mock_adapter
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.delete_unused_data') as mock_delete:
                mock_delete.return_value = {
                    'document_chunks': 5,
                    'entities': 3,
                    'summaries': 2
                }
                
                result = await cleanup_unused_data(
                    days_threshold=30,
                    dry_run=True
                )
                
                assert isinstance(result, CleanupResult)
                assert result.status == 'success'
                assert result.dry_run is True
                assert result.deleted_counts['document_chunks'] == 5
                assert result.deleted_counts['entities'] == 3
                assert result.deleted_counts['summaries'] == 2
                assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_cleanup_actual_deletion_custom_graph(self, mock_adapter):
        """Test actual deletion on custom graph."""
        adapter, session = mock_adapter
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.delete_unused_data') as mock_delete:
                mock_delete.return_value = {
                    'custom_nodes': 10,
                    'custom_edges': 15
                }
                
                result = await cleanup_unused_data(
                    days_threshold=60,
                    dry_run=False,
                    schema='my_custom_graph'
                )
                
                assert result.status == 'success'
                assert result.dry_run is False
                assert result.deleted_counts['custom_nodes'] == 10
                assert result.deleted_counts['custom_edges'] == 15
                # Verify schema was passed to delete function
                mock_delete.assert_called_once()
                call_kwargs = mock_delete.call_args[1]
                assert call_kwargs['schema'] == 'my_custom_graph'

    @pytest.mark.asyncio
    async def test_cleanup_handles_errors(self, mock_adapter):
        """Test that cleanup handles errors gracefully."""
        adapter, session = mock_adapter
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.delete_unused_data') as mock_delete:
                mock_delete.side_effect = Exception('Database connection error')
                
                result = await cleanup_unused_data(
                    days_threshold=30,
                    dry_run=True
                )
                
                assert result.status == 'error'
                assert len(result.errors) > 0
                assert 'Database connection error' in result.errors[0]

    @pytest.mark.asyncio
    async def test_get_cleanup_preview(self, mock_adapter):
        """Test getting cleanup preview without deletion."""
        adapter, session = mock_adapter
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.get_unused_data_counts') as mock_counts:
                mock_counts.return_value = {
                    'document_chunks': 100,
                    'entities': 50,
                    'summaries': 25
                }
                
                result = await get_cleanup_preview(days_threshold=30)
                
                assert result['document_chunks'] == 100
                assert result['entities'] == 50
                assert result['summaries'] == 25

    @pytest.mark.asyncio
    async def test_get_data_usage_statistics(self, mock_adapter):
        """Test getting data usage statistics."""
        adapter, session = mock_adapter
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.get_table_statistics') as mock_stats:
                mock_stats.return_value = {
                    'document_chunks': {
                        'total_records': 1000,
                        'never_accessed': 200,
                        'recently_accessed': 600,
                        'accessed_percentage': 80.0
                    }
                }
                
                result = await get_data_usage_statistics()
                
                assert 'document_chunks' in result
                assert result['document_chunks']['total_records'] == 1000
                assert result['document_chunks']['accessed_percentage'] == 80.0


class TestAccessTracking:
    """Test suite for schema-agnostic access tracking."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.run_sync = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_update_last_accessed_single_entity(self, mock_session):
        """Test updating last_accessed for single entity."""
        entity_id = uuid4()
        
        # Mock metadata reflection
        def mock_reflect(func):
            metadata = MetaData()
            Table('document_chunks', metadata,
                  Column('id', PGUUID, primary_key=True),
                  Column('last_accessed', DateTime))
            func(Mock(bind=Mock()))
        
        mock_session.run_sync.side_effect = mock_reflect
        mock_session.execute.return_value = Mock(rowcount=1)
        
        with patch('cognee.modules.data.access_tracking.MetaData') as mock_metadata_class:
            mock_metadata = Mock()
            mock_metadata.tables = {
                'document_chunks': Mock(
                    columns={'id': Mock(), 'last_accessed': Mock()},
                    c=Mock(id=Mock(in_=Mock(return_value=Mock())))
                )
            }
            mock_metadata_class.return_value = mock_metadata
            
            await update_last_accessed(
                session=mock_session,
                entity_ids=entity_id,
                table_name='document_chunks'
            )
            
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_update_last_accessed(self, mock_session):
        """Test bulk updating last_accessed across multiple tables."""
        updates = [
            {'table_name': 'document_chunks', 'entity_ids': [uuid4(), uuid4()]},
            {'table_name': 'entities', 'entity_ids': [uuid4()]}
        ]
        
        with patch('cognee.modules.data.access_tracking.update_last_accessed') as mock_update:
            mock_update.return_value = None
            
            await bulk_update_last_accessed(
                session=mock_session,
                updates=updates
            )
            
            assert mock_update.call_count == 2

    @pytest.mark.asyncio
    async def test_mark_entity_accessed(self, mock_session):
        """Test marking single entity as accessed."""
        entity_id = uuid4()
        
        with patch('cognee.modules.data.access_tracking.update_last_accessed') as mock_update:
            mock_update.return_value = None
            
            result = await mark_entity_accessed(
                session=mock_session,
                entity_id=entity_id,
                table_name='document_chunks'
            )
            
            assert result is True
            mock_update.assert_called_once_with(
                session=mock_session,
                entity_ids=[entity_id],
                table_name='document_chunks',
                schema=None
            )


class TestCustomGraphSupport:
    """Test suite specifically for custom graph schema support."""

    @pytest.mark.asyncio
    async def test_cleanup_with_custom_schema(self):
        """Test cleanup operation with custom schema parameter."""
        adapter = Mock()
        session = AsyncMock()
        adapter.get_async_session = MagicMock(return_value=session)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('cognee.tasks.cleanup_unused_data.get_relational_engine', return_value=adapter):
            with patch('cognee.modules.data.cleanup_unused_data.delete_unused_data') as mock_delete:
                mock_delete.return_value = {'custom_table': 5}
                
                result = await cleanup_unused_data(
                    days_threshold=30,
                    dry_run=True,
                    schema='custom_schema'
                )
                
                # Verify schema was passed correctly
                call_kwargs = mock_delete.call_args[1]
                assert call_kwargs['schema'] == 'custom_schema'

    @pytest.mark.asyncio
    async def test_access_tracking_with_custom_schema(self):
        """Test access tracking with custom schema."""
        session = AsyncMock()
        entity_id = uuid4()
        
        with patch('cognee.modules.data.access_tracking.update_last_accessed') as mock_update:
            mock_update.return_value = None
            
            await mark_entity_accessed(
                session=session,
                entity_id=entity_id,
                table_name='custom_nodes',
                schema='my_graph'
            )
            
            # Verify schema was passed
            call_kwargs = mock_update.call_args[1]
            assert call_kwargs['schema'] == 'my_graph'
