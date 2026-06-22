import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import the module directly so we can patch attributes via patch.object.
# Patching by the dotted string "cognee.api.v1.serve.state…" fails on
# Python 3.10's unittest.mock because `cognee.api.v1.serve` is shadowed by
# the `serve()` function re-exported in cognee/api/v1/__init__.py — mock's
# pre-3.11 dotted-path walk hits the function and raises AttributeError.
from cognee.api.v1.serve import state as serve_state
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _remember_module():
    return importlib.import_module("cognee.api.v1.remember.remember")


def _recall_module():
    return importlib.import_module("cognee.api.v1.recall.recall")


def _search_methods_module():
    return importlib.import_module("cognee.modules.search.methods.search")


@pytest.mark.asyncio
async def test_cognee_remember_public_api_completes():
    import cognee

    remember_mod = _remember_module()
    mock_user = MagicMock()
    mock_user.id = uuid4()
    mock_dataset = MagicMock()
    mock_dataset.id = uuid4()

    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=mock_user)),
        patch.object(
            remember_mod,
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, [mock_dataset])),
        ),
        patch("cognee.api.v1.add.add", AsyncMock()) as add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})) as cognify,
    ):
        result = await cognee.remember(
            "public memory smoke",
            dataset_name="public_api",
            self_improvement=False,
        )

    add.assert_awaited_once()
    cognify.assert_awaited_once()
    assert isinstance(result, cognee.RememberResult)
    assert result.status == "completed"
    assert result.dataset_name == "public_api"


@pytest.mark.asyncio
async def test_cognee_recall_public_api_resolves_default_user_for_graph_search():
    import cognee

    recall_mod = _recall_module()
    search_methods = _search_methods_module()
    mock_user = MagicMock()
    mock_user.id = uuid4()
    mock_payload = SearchResultPayload(
        result_object="graph result",
        search_type=SearchType.GRAPH_COMPLETION,
    )

    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch.object(recall_mod, "get_default_user", AsyncMock(return_value=mock_user)),
        patch.object(recall_mod, "set_session_user_context_variable", AsyncMock()),
        patch.object(
            search_methods,
            "authorized_search",
            AsyncMock(return_value=[mock_payload]),
        ) as authorized_search,
    ):
        results = await cognee.recall(
            "public recall smoke",
            query_type=SearchType.GRAPH_COMPLETION,
            auto_route=False,
        )

    authorized_search.assert_awaited_once()
    assert authorized_search.await_args.kwargs["user"] is mock_user
    assert len(results) == 1
    assert results[0].source == "graph"
    assert results[0].text == "graph result"
