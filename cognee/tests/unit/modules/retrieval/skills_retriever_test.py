import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from cognee.modules.retrieval.skills_retriever import SkillsRetriever
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

DATASET_ID = str(uuid4())
OTHER_DATASET_ID = str(uuid4())


def _make_unified_mock(vector_engine):
    """Create a mock unified engine that exposes the given vector engine."""
    unified = AsyncMock()
    unified.vector = vector_engine
    unified.graph = AsyncMock()
    return unified


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.search = AsyncMock()
    return engine


def _skill_payload(**overrides):
    """A full Skill vector payload, procedure body included."""
    payload = {
        "id": str(uuid4()),
        "name": "deploy-checklist",
        "description": "Steps to deploy to staging",
        "procedure": "SECRET PROCEDURE BODY",
        "skill_text": "deploy-checklist\n\nSECRET PROCEDURE BODY",
        "search_text": "deploy-checklist\n\nSECRET PROCEDURE BODY",
        "declared_tools": ["bash"],
        "maintainer": "ops-team",
        "maintainer_url": "https://example.com",
        "skill_version": "1.2",
        "tags": ["ops"],
        "license": "MIT",
        "source_repo_url": "",
        "source_dir": "skills/deploy-checklist",
        "dataset_scope": [DATASET_ID],
        "is_active": True,
    }
    payload.update(overrides)
    return payload


def _result(payload, score=0.1):
    return SimpleNamespace(id=payload.get("id"), payload=payload, score=score)


def _patch_engine(monkeypatch, vector_engine):
    import cognee.modules.retrieval.skills_retriever as mod

    async def _get_unified_engine():
        return _make_unified_mock(vector_engine)

    monkeypatch.setattr(mod, "get_unified_engine", _get_unified_engine)


def test_init_requires_dataset_id():
    """SKILLS search is single-dataset by invariant; no dataset is an error."""
    with pytest.raises(QueryValidationError, match="exactly one explicit dataset"):
        SkillsRetriever()


def test_init_defaults():
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    assert retriever.top_k == 5
    assert retriever.dataset_id == DATASET_ID


def test_init_custom_top_k_and_uuid_dataset():
    dataset_uuid = uuid4()
    retriever = SkillsRetriever(top_k=10, dataset_id=dataset_uuid)
    assert retriever.top_k == 10
    assert retriever.dataset_id == str(dataset_uuid)


def test_init_top_k_none_falls_back_to_default():
    retriever = SkillsRetriever(top_k=None, dataset_id=DATASET_ID)
    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_get_objects_filters_scope_and_active(monkeypatch, mock_vector_engine):
    """Only active skills scoped to this dataset survive; empty scope is excluded."""
    in_scope = _skill_payload(name="in-scope")
    inactive = _skill_payload(name="inactive", is_active=False)
    other_dataset = _skill_payload(name="other-dataset", dataset_scope=[OTHER_DATASET_ID])
    empty_scope = _skill_payload(name="empty-scope", dataset_scope=[])

    mock_vector_engine.search.return_value = [
        _result(in_scope),
        _result(inactive),
        _result(other_dataset),
        _result(empty_scope),
    ]
    _patch_engine(monkeypatch, mock_vector_engine)

    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    objects = await retriever.get_retrieved_objects("deploy")

    assert [obj.payload["name"] for obj in objects] == ["in-scope"]


@pytest.mark.asyncio
async def test_get_objects_overfetches_and_trims_to_top_k(monkeypatch, mock_vector_engine):
    """Fetch limit exceeds top_k (filtering shrinks results); output is trimmed."""
    payloads = [_skill_payload(name=f"skill-{i}") for i in range(5)]
    mock_vector_engine.search.return_value = [_result(p) for p in payloads]
    _patch_engine(monkeypatch, mock_vector_engine)

    retriever = SkillsRetriever(top_k=2, dataset_id=DATASET_ID)
    objects = await retriever.get_retrieved_objects("deploy")

    assert len(objects) == 2
    mock_vector_engine.search.assert_awaited_once_with(
        "Skill_search_text", "deploy", limit=20, include_payload=True
    )


@pytest.mark.asyncio
async def test_get_objects_dedupes_by_id(monkeypatch, mock_vector_engine):
    payload = _skill_payload(name="dup")
    mock_vector_engine.search.return_value = [_result(payload), _result(payload)]
    _patch_engine(monkeypatch, mock_vector_engine)

    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    objects = await retriever.get_retrieved_objects("deploy")

    assert len(objects) == 1


@pytest.mark.asyncio
async def test_get_objects_collection_not_found_returns_empty(monkeypatch, mock_vector_engine):
    """No skills ingested yet is a normal state — no NoDataError (unlike SUMMARIES)."""
    mock_vector_engine.search.side_effect = CollectionNotFoundError("Collection not found")
    _patch_engine(monkeypatch, mock_vector_engine)

    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    objects = await retriever.get_retrieved_objects("deploy")

    assert objects == []


@pytest.mark.asyncio
async def test_get_objects_empty_results(monkeypatch, mock_vector_engine):
    mock_vector_engine.search.return_value = []
    _patch_engine(monkeypatch, mock_vector_engine)

    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    objects = await retriever.get_retrieved_objects("deploy")

    assert objects == []


@pytest.mark.asyncio
async def test_context_lists_names_and_descriptions_only():
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    objects = [_result(_skill_payload(name="deploy-checklist"))]

    context = await retriever.get_context_from_objects("deploy", objects)

    assert context == "- `deploy-checklist`: Steps to deploy to staging"
    assert "SECRET PROCEDURE BODY" not in context


@pytest.mark.asyncio
async def test_context_empty_without_objects():
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    assert await retriever.get_context_from_objects("deploy", []) == ""


@pytest.mark.asyncio
async def test_completion_projection_is_metadata_only():
    """Results never expose the procedure body — progressive disclosure."""
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    payload = _skill_payload()
    objects = [_result(payload, score=0.42)]

    completion = await retriever.get_completion_from_context("deploy", objects, "")

    assert len(completion) == 1
    projected = completion[0]
    assert projected["name"] == "deploy-checklist"
    assert projected["description"] == "Steps to deploy to staging"
    assert projected["version"] == "1.2"
    assert projected["declared_tools"] == ["bash"]
    assert projected["dataset_scope"] == [DATASET_ID]
    assert projected["score"] == 0.42
    for stripped_field in ("procedure", "skill_text", "search_text"):
        assert stripped_field not in projected


@pytest.mark.asyncio
async def test_completion_tolerates_sparse_legacy_payload():
    """Skills ingested before newer fields existed must not crash the projection."""
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    sparse = {"id": str(uuid4()), "name": "old-skill", "dataset_scope": [DATASET_ID]}
    objects = [SimpleNamespace(id=sparse["id"], payload=sparse, score=None)]

    completion = await retriever.get_completion_from_context("deploy", objects, "")

    assert completion[0]["name"] == "old-skill"
    assert completion[0]["description"] == ""
    assert completion[0]["tags"] == []
    assert completion[0]["is_active"] is True
    assert "score" not in completion[0]


@pytest.mark.asyncio
async def test_completion_empty_without_objects():
    retriever = SkillsRetriever(dataset_id=DATASET_ID)
    assert await retriever.get_completion_from_context("deploy", [], "") == []
