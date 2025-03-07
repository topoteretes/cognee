import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.data.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods.get_data import get_data
from cognee.modules.data.models.Data import Data


@pytest.fixture
def mock_data():
    data = MagicMock(spec=Data)
    data.id = uuid.uuid4()
    data.name = "test_data.txt"
    data.extension = "txt"
    data.mime_type = "text/plain"
    data.raw_data_location = "/path/to/data"
    data.owner_id = uuid.uuid4()
    data.content_hash = "abc123"
    data.external_metadata = {"key": "value"}
    data.token_count = 100
    data.created_at = datetime.now(timezone.utc)
    data.updated_at = None
    data.__tablename__ = "data"
    return data


@pytest.mark.asyncio
@patch("cognee.modules.data.methods.get_data.get_relational_engine")
async def test_get_data_success(mock_get_relational_engine, mock_data):
    # Setup
    user_id = mock_data.owner_id
    data_id = mock_data.id
    
    # Mock the database session
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = mock_data
    
    # Mock the database engine
    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value.__aenter__.return_value = mock_session
    mock_get_relational_engine.return_value = mock_engine
    
    # Execute
    result = await get_data(user_id, data_id)
    
    # Verify
    mock_session.get.assert_called_once_with(Data, data_id)
    assert result == mock_data


@pytest.mark.asyncio
@patch("cognee.modules.data.methods.get_data.get_relational_engine")
async def test_get_data_unauthorized(mock_get_relational_engine, mock_data):
    # Setup
    user_id = uuid.uuid4()  # Different from mock_data.owner_id
    data_id = mock_data.id
    
    # Mock the database session
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = mock_data
    
    # Mock the database engine
    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value.__aenter__.return_value = mock_session
    mock_get_relational_engine.return_value = mock_engine
    
    # Execute and verify
    with pytest.raises(UnauthorizedDataAccessError) as excinfo:
        await get_data(user_id, data_id)
    
    mock_session.get.assert_called_once_with(Data, data_id)
    assert f"User {user_id} is not authorized to access data {data_id}" in str(excinfo.value)


@pytest.mark.asyncio
@patch("cognee.modules.data.methods.get_data.get_relational_engine")
async def test_get_data_not_found(mock_get_relational_engine):
    # Setup
    user_id = uuid.uuid4()
    data_id = uuid.uuid4()
    
    # Mock the database session
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = None  # Data not found
    
    # Mock the database engine
    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value.__aenter__.return_value = mock_session
    mock_get_relational_engine.return_value = mock_engine
    
    # Execute
    result = await get_data(user_id, data_id)
    
    # Verify
    mock_session.get.assert_called_once_with(Data, data_id)
    assert result is None 