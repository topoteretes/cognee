"""Discovery is provider-independent and authorization happens before metadata/LLM reads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.data import source_catalog as catalog


def dataset(name="workspace"):
    return SimpleNamespace(id=uuid4(), name=name)


def test_new_sources_and_node_sets_need_no_provider_registration():
    data = dataset()
    rows = [
        (
            data.id,
            [f"custom-source:{i}"],
            None,
            f"Lab {i} research",
            f"provider-{i}",
            f"Knowledge about experiment {i}",
        )
        for i in range(300)
    ]
    result = catalog.build_catalog([data], rows)
    assert len(result) == 301
    assert any("provider-299" in r["aliases"] for r in result)
    assert len({r["id"] for r in result}) == len(result)
    assert catalog.build_catalog([data], rows) == result


def test_same_node_set_in_two_datasets_has_distinct_identity():
    one, two = dataset(), dataset()
    rows = [(d.id, ["same"], None, None, None, None) for d in (one, two)]
    targets = catalog.build_catalog([one, two], rows)
    nodes = [t for t in targets if t["kind"] == "node_set"]
    assert len(nodes) == 2 and nodes[0]["id"] != nodes[1]["id"]


@pytest.mark.asyncio
async def test_unauthorized_dataset_fails_before_database_or_llm(monkeypatch):
    one, hidden = dataset(), dataset()
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", AsyncMock(return_value=[one]))
    monkeypatch.setattr(
        catalog, "get_relational_engine", lambda: pytest.fail("unauthorized DB read")
    )
    with pytest.raises(PermissionError):
        await catalog.source_catalog(SimpleNamespace(id=uuid4()), [hidden.id])


@pytest.mark.asyncio
async def test_hundreds_of_descriptors_are_routed_in_bounded_batches(monkeypatch):
    data = dataset()
    items = catalog.build_catalog(
        [data], [(data.id, [f"unknown:{i}"], None, f"Project {i}", None, None) for i in range(200)]
    )
    target = next(item for item in items if item["name"] == "unknown:177")
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    sizes = []

    async def rank(query, hint, descriptors):
        sizes.append(len(descriptors))
        return catalog.SourceChoices(
            choices=[
                catalog.SourceChoice(
                    source_id=target["id"], relevance=0.9, reason="Matches the requested project"
                )
            ]
            if any(d["id"] == target["id"] for d in descriptors)
            else []
        )

    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "project 177")
    assert [t["id"] for t in result["targets"]] == [target["id"]]
    assert max(sizes) <= 64
    assert result["catalog_entries"] == 201
    assert result["searched_content"] is False


@pytest.mark.asyncio
async def test_llm_cannot_invent_an_unauthorized_target(monkeypatch):
    items = catalog.build_catalog([dataset()], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    monkeypatch.setattr(
        catalog,
        "rank_descriptors",
        AsyncMock(
            return_value=catalog.SourceChoices(
                choices=[
                    catalog.SourceChoice(source_id=str(uuid4()), relevance=1, reason="injected")
                ]
            )
        ),
    )
    with pytest.raises(ValueError, match="unknown catalog ID"):
        await catalog.route_sources(None, "question")


@pytest.mark.asyncio
async def test_metadata_budget_is_not_silent_catalog_truncation(monkeypatch):
    items = catalog.build_catalog([dataset(str(i)) for i in range(20)], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    rank = AsyncMock()
    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "question", max_catalog_entries=10)
    assert result["status"] == "catalog_budget_exceeded"
    assert not result["targets"]
    rank.assert_not_awaited()


@pytest.mark.asyncio
async def test_inconclusive_routing_does_not_claim_no_content(monkeypatch):
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": [], "complete": True})
    )
    result = await catalog.route_sources(None, "question")
    assert result["status"] == "inconclusive"
    assert result["searched_content"] is False


@pytest.mark.asyncio
async def test_router_does_not_inherit_ambient_context(monkeypatch):
    from contextvars import ContextVar

    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    secret = ContextVar("unrelated_memory", default=None)
    token = secret.set("another session")

    async def operation(**kwargs):
        assert secret.get() is None
        return catalog.SourceChoices()

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", operation)
    try:
        await catalog.rank_descriptors("question", None, [])
    finally:
        secret.reset(token)


@pytest.mark.asyncio
async def test_large_catalog_reduces_winners_and_excludes_searched_targets(monkeypatch):
    items = catalog.build_catalog([dataset(str(i)) for i in range(900)], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    excluded = items[0]["id"]
    sizes = []

    async def rank(query, hint, descriptors):
        assert excluded not in [d["id"] for d in descriptors]
        sizes.append(len(descriptors))
        return catalog.SourceChoices(
            choices=[
                catalog.SourceChoice(source_id=d["id"], relevance=0.9, reason="relevant")
                for d in descriptors[:8]
            ]
        )

    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    from uuid import UUID

    result = await catalog.route_sources(
        None, "broad query", max_catalog_entries=1000, exclude_source_ids=[UUID(excluded)]
    )
    assert result["status"] == "selected"
    assert len(result["targets"]) == 6
    assert max(sizes) <= 64
    assert result["routing_llm_calls"] == len(sizes)


@pytest.mark.asyncio
async def test_native_metadata_projection_pages_aliases_and_revocation(monkeypatch):
    """Use a real isolated relational table; no graph, source fetch, or live state."""
    from uuid import UUID

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from cognee.modules.data.methods import resolve_data_id

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    engine_wrapper = SimpleNamespace(get_async_session=sessions)
    monkeypatch.setattr(catalog, "get_relational_engine", lambda: engine_wrapper)
    # The native alias resolver uses the same isolated relational engine.
    monkeypatch.setitem(
        resolve_data_id.__globals__, "get_relational_engine", lambda: engine_wrapper
    )
    one, hidden = dataset("research"), dataset("private")
    permission = AsyncMock(return_value=[one])
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", permission)
    monkeypatch.setattr(catalog, "get_authorized_existing_datasets", permission)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(catalog.Data.__table__.create)
        async with sessions() as db:
            for index in range(1, 6):
                db.add(
                    catalog.Data(
                        id=UUID(int=(10 << 124) + index),
                        dataset_id=one.id if index < 5 else hidden.id,
                        legacy_id=UUID(int=(10 << 124) + 99) if index == 1 else None,
                        node_set=["research"] if index != 2 else ["other"],
                        label=f"Document {index}",
                        raw_data_location="must-not-be-exposed",
                        external_metadata={"source": "new-provider", "secret": "never-in-catalog"},
                    )
                )
            await db.commit()
        result = await catalog.source_catalog(None)
        assert len(result["items"]) == 3
        assert all(item["dataset_id"] == str(one.id) for item in result["items"])
        assert "never-in-catalog" not in str(result)
        assert "must-not-be-exposed" not in str(result)
        assert any("new-provider" in item["aliases"] for item in result["items"])
        source_id = catalog.target_id(one.id, "research")
        page = await catalog.source_documents(None, source_id, limit=2)
        assert [item["id"] for item in page["items"]] == [
            str(UUID(int=(10 << 124) + 1)),
            str(UUID(int=(10 << 124) + 3)),
        ]
        page2 = await catalog.source_documents(
            None, source_id, after=UUID(page["next_cursor"]), limit=2
        )
        assert [item["id"] for item in page2["items"]] == [str(UUID(int=(10 << 124) + 4))]
        assert page2["next_cursor"] is None
        old = await catalog.source_document(None, one.id, UUID(int=(10 << 124) + 99))
        assert old["id"] == str(UUID(int=(10 << 124) + 1))
        with pytest.raises(PermissionError):
            await catalog.source_document(None, one.id, UUID(int=(10 << 124) + 5))
        monkeypatch.setattr(catalog, "MAX_METADATA_ROWS", 2)
        assert (await catalog.source_catalog(None))["complete"] is False
        permission.return_value = []
        with pytest.raises(PermissionError):
            await catalog.source_documents(None, source_id, after=page["next_cursor"])
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_named_source_cannot_expand_to_mixed_topic_group(monkeypatch):
    data = dataset()
    items = catalog.build_catalog(
        [data],
        [
            (data.id, ["meetings", "origin:custom-a"], None, "meeting A", "Custom A", None),
            (data.id, ["meetings", "origin:custom-b"], None, "meeting B", "Custom B", None),
        ],
    )
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )

    async def rank(query, hint, candidates):
        assert len(candidates) == 1
        assert candidates[0]["name"] == "origin:custom-a"
        return catalog.SourceChoices(
            choices=[
                catalog.SourceChoice(
                    source_id=candidates[0]["id"], relevance=1, reason="Exact provenance"
                )
            ]
        )

    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "commitments", source_hint="Custom A")
    assert result["source_resolution"] == "exact_metadata"
    assert result["targets"][0]["node_sets"] == ["origin:custom-a"]
    from uuid import UUID

    empty = await catalog.route_sources(
        None,
        "commitments",
        source_hint="Custom A",
        exclude_source_ids=[UUID(result["targets"][0]["id"])],
    )
    assert empty["status"] == "inconclusive" and empty["targets"] == []


@pytest.mark.asyncio
async def test_llm_indexes_resolve_only_to_supplied_targets(monkeypatch):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    items = catalog.build_catalog([dataset()], [])
    method = AsyncMock(
        return_value=catalog.RoutingChoices(
            choices=[catalog.RoutingChoice(index=0, relevance=1, reason="Relevant")]
        )
    )
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", method)
    result = await catalog.rank_descriptors("question", None, items)
    assert result.choices[0].source_id == items[0]["id"]
    assert items[0]["id"] not in method.await_args.kwargs["text_input"]
    method.return_value = catalog.RoutingChoices(
        choices=[catalog.RoutingChoice(index=999, relevance=1, reason="Invalid")]
    )
    with pytest.raises(ValueError, match="unknown catalog index"):
        await catalog.rank_descriptors("question", None, items)
