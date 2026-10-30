from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from cognee.api.v1.responses.routers.get_responses_router import get_responses_router
from cognee.api.v1.responses.models import ResponseRequest

@pytest.mark.asyncio
async def test_dynamic_model_routing():
    # Setup mock client
    mock_client = AsyncMock()
    mock_responses_create = AsyncMock()
    # Mock responses.create response format
    mock_responses_create.return_value = MagicMock()
    mock_responses_create.return_value.model_dump.return_value = {
        "id": "resp_test_123",
        "output": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        }
    }
    mock_client.responses.create = mock_responses_create
    
    mock_user = MagicMock()

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        router = get_responses_router()
        # Find the route for POST "/"
        route = [r for r in router.routes if r.path == "/"][0]
        
        # Test case 1: Default / "cognee-v1" model (resolves to gpt-4o)
        request = ResponseRequest(input="test default", model="cognee-v1")
        response_body = await route.endpoint(request=request, user=mock_user)
        
        mock_responses_create.assert_called_once()
        called_kwargs = mock_responses_create.call_args.kwargs
        assert called_kwargs["model"] == "gpt-4o"
        assert called_kwargs["input"] == "test default"
        assert response_body.model == "cognee-v1"
        mock_responses_create.reset_mock()
        
        # Test case 2: Cognee model alias e.g. "cognee-v1-openai-gpt-4o-mini" (resolves to gpt-4o-mini)
        request = ResponseRequest(input="test alias", model="cognee-v1-openai-gpt-4o-mini")
        response_body = await route.endpoint(request=request, user=mock_user)
        
        mock_responses_create.assert_called_once()
        called_kwargs = mock_responses_create.call_args.kwargs
        assert called_kwargs["model"] == "gpt-4o-mini"
        assert called_kwargs["input"] == "test alias"
        assert response_body.model == "cognee-v1-openai-gpt-4o-mini"
        mock_responses_create.reset_mock()

        # Test case 3: Cognee general alias e.g. "cognee-v1-custom-model" (resolves to custom-model)
        request = ResponseRequest(input="test general alias", model="cognee-v1-custom-model")
        response_body = await route.endpoint(request=request, user=mock_user)
        
        mock_responses_create.assert_called_once()
        called_kwargs = mock_responses_create.call_args.kwargs
        assert called_kwargs["model"] == "custom-model"
        assert called_kwargs["input"] == "test general alias"
        assert response_body.model == "cognee-v1-custom-model"
        mock_responses_create.reset_mock()

        # Test case 4: Standard model name e.g. "gpt-3.5-turbo" (resolves to gpt-3.5-turbo directly)
        request = ResponseRequest(input="test standard", model="gpt-3.5-turbo")
        response_body = await route.endpoint(request=request, user=mock_user)
        
        mock_responses_create.assert_called_once()
        called_kwargs = mock_responses_create.call_args.kwargs
        assert called_kwargs["model"] == "gpt-3.5-turbo"
        assert called_kwargs["input"] == "test standard"
        assert response_body.model == "gpt-3.5-turbo"
        mock_responses_create.reset_mock()
