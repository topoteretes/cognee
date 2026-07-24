import importlib
import sys
import types
from uuid import uuid4
import pytest
from cognee.modules.search.types import SearchType

def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)

@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")

@pytest.mark.asyncio
async def test_recall_normalization_and_weighting(monkeypatch, api_recall_mod):
    user = _make_user()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        return []

    # Graph search returns two results
    async def dummy_authorized_search(**kwargs):
        payload_mock1 = types.SimpleNamespace(
            search_type=SearchType.GRAPH_COMPLETION,
            only_context=False,
            completion=["Answer from Graph High"],
            context=None,
            result_object=None,
            dataset_id=uuid4(),
            dataset_name="dataset_1"
        )
        payload_mock2 = types.SimpleNamespace(
            search_type=SearchType.GRAPH_COMPLETION,
            only_context=False,
            completion=["Answer from Graph Low"],
            context=None,
            result_object=None,
            dataset_id=uuid4(),
            dataset_name="dataset_1"
        )
        return [payload_mock1, payload_mock2]

    # Session search returns two results
    async def dummy_search_session(**kwargs):
        from cognee.modules.recall.types.RecallResponse import ResponseQAEntry
        entry1 = ResponseQAEntry(
            time="2026-06-24T19:00:00Z",
            question="What is X?",
            context="X is Y.",
            answer="X is Y (High).",
            source="session"
        )
        entry2 = ResponseQAEntry(
            time="2026-06-24T19:00:00Z",
            question="What is Z?",
            context="Z is W.",
            answer="Z is W (Low).",
            source="session"
        )
        object.__setattr__(entry1, "score", 10.0)
        object.__setattr__(entry2, "score", 2.0)
        return [entry1, entry2]

    # Setup monkeypatches
    monkeypatch.setattr(api_recall_mod, "_search_session", dummy_search_session)
    monkeypatch.setattr(api_recall_mod, "set_session_user_context_variable", dummy_set_session_user_context_variable)
    monkeypatch.setattr(api_recall_mod, "get_authorized_existing_datasets", dummy_get_authorized_existing_datasets)

    # Mock authorized_search in the search.py module via sys.modules
    import cognee.modules.search.methods.search as dummy_import
    search_methods_module = sys.modules["cognee.modules.search.methods.search"]
    monkeypatch.setattr(search_methods_module, "authorized_search", dummy_authorized_search)

    # We also mock normalize_search_payload
    import cognee.modules.recall.methods.normalize_search_payload as normalize_mod
    def dummy_normalize_payload(payload):
        items = [
            types.SimpleNamespace(
                kind="graph_completion",
                search_type=payload.search_type,
                text=payload.completion[0],
                score=0.9 if "High" in payload.completion[0] else 0.5,
                dataset_id=str(payload.dataset_id),
                dataset_name=payload.dataset_name,
                model_dump=lambda **k: {"kind": "graph_completion", "search_type": payload.search_type, "text": payload.completion[0], "score": 0.9 if "High" in payload.completion[0] else 0.5}
            )
        ]
        return items

    monkeypatch.setattr(normalize_mod, "normalize_search_payload", dummy_normalize_payload)

    # Call recall with custom score_weights
    results = await api_recall_mod.recall(
        query_text="What is X?",
        scope=["session", "graph"],
        auto_route=False,
        user=user,
        session_id="session-1",
        score_weights={"session": 1.0, "graph": 0.5}
    )

    # Verify scores and sorting
    assert len(results) == 4
    # Expected ordering:
    # 1. session high -> score = 1.0
    # 2. graph high -> score = 0.5
    # 3. session low -> score = 0.0
    # 4. graph low -> score = 0.0
    assert results[0].source == "session"
    assert results[0].score == 1.0

    assert results[1].source == "graph"
    assert results[1].score == 0.5

    assert results[2].source == "session"
    assert results[2].score == 0.0

    assert results[3].source == "graph"
    assert results[3].score == 0.0
