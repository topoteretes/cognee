"""Unit tests ensuring cognify() validates keyword arguments and raises TypeError on unexpected kwargs (#5773)."""

import importlib
import sys
import types
from unittest.mock import AsyncMock, patch
import pytest

# Ensure litellm shim exists for Windows environment if site-packages is minimal
if "litellm" not in sys.modules:
    import litellm
else:
    litellm = sys.modules["litellm"]

if not hasattr(litellm, "exceptions"):
    exc_mod = types.ModuleType("litellm.exceptions")
    exc_mod.NotFoundError = type("NotFoundError", (Exception,), {})
    litellm.exceptions = exc_mod
    sys.modules["litellm.exceptions"] = exc_mod

if not hasattr(litellm, "model_cost"):
    litellm.model_cost = {}

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


@pytest.mark.asyncio
async def test_cognify_raises_type_error_on_typo_kwargs():
    """Verify that a typo like extactor='gliner_demo' raises TypeError immediately."""
    with pytest.raises(TypeError) as exc_info:
        await cognify_module.cognify(extactor="gliner_demo")

    assert "unexpected keyword argument(s)" in str(exc_info.value)
    assert "'extactor'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_raises_type_error_on_multiple_unknown_kwargs():
    """Verify that multiple unknown kwargs are sorted and included in the TypeError message."""
    with pytest.raises(TypeError) as exc_info:
        await cognify_module.cognify(sumary_method="from_extraction", invalid_param=123)

    assert "unexpected keyword argument(s)" in str(exc_info.value)
    assert "'invalid_param'" in str(exc_info.value)
    assert "'sumary_method'" in str(exc_info.value)


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_cognify_permits_legacy_temporal_cognify(
    _resolver,
    _migrations,
    mock_get_default_tasks,
    mock_get_pipeline_executor,
    _remote_client,
):
    """Verify that legacy temporal_cognify is popped and does not trigger TypeError."""
    mock_get_default_tasks.return_value = []
    mock_get_pipeline_executor.return_value = AsyncMock(return_value={})

    # Should run cleanly without raising TypeError
    await cognify_module.cognify(extractor="llm", temporal_cognify=True)
    mock_get_default_tasks.assert_awaited_once()
