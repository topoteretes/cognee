import tempfile
from unittest.mock import MagicMock, patch

import pytest
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.tasks.memify.apply_feedback_weights import apply_feedback_weights
from cognee.tasks.memify.extract_feedback_qas import extract_feedback_qas
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)


class _InMemoryRedisList:
    def __init__(self):
        self.data: dict[str, list[str]] = {}

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        pass


class InMemoryGraphWithWeights:
    def __init__(self):
        self.node_weights = {"n1": 0.5}
        self.edge_weights = {"e1": 0.5}

    async def get_node_feedback_weights(self, node_ids):
        return {
            node_id: self.node_weights[node_id]
            for node_id in node_ids
            if node_id in self.node_weights
        }

    async def set_node_feedback_weights(self, node_feedback_weights):
        result = {}
        for node_id, weight in node_feedback_weights.items():
            if node_id in self.node_weights:
                self.node_weights[node_id] = float(weight)
                result[node_id] = True
            else:
                result[node_id] = False
        return result

    async def get_edge_feedback_weights(self, edge_object_ids):
        return {
            edge_object_id: self.edge_weights[edge_object_id]
            for edge_object_id in edge_object_ids
            if edge_object_id in self.edge_weights
        }

    async def set_edge_feedback_weights(self, edge_feedback_weights):
        result = {}
        for edge_object_id, weight in edge_feedback_weights.items():
            if edge_object_id in self.edge_weights:
                self.edge_weights[edge_object_id] = float(weight)
                result[edge_object_id] = True
            else:
                result[edge_object_id] = False
        return result


@pytest.fixture(params=["fs", "redis"])
def session_manager_with_backend(request):
    backend = request.param
    if backend == "fs":
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
                return_value={"data_root_directory": tmpdir},
            ):
                from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                    FSCacheAdapter,
                )

                adapter = FSCacheAdapter()
                sm = SessionManager(cache_engine=adapter)
                yield sm
                adapter.cache.close()
    else:
        store = _InMemoryRedisList()
        patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
        with (
            patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
            patch(f"{patch_mod}.aioredis.Redis", return_value=store),
        ):
            from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

            adapter = RedisAdapter(host="localhost", port=6379)
            sm = SessionManager(cache_engine=adapter)
            yield sm


def _make_user():
    user = MagicMock()
    user.id = "u1"
    return user


@pytest.mark.asyncio
async def test_feedback_weights_first_run_then_idempotent(session_manager_with_backend):
    sm = session_manager_with_backend
    user = _make_user()

    await sm.add_qa(
        user_id="u1",
        question="Q",
        context="C",
        answer="A",
        session_id="s1",
        feedback_score=5,
        used_graph_element_ids={"node_ids": ["n1"], "edge_ids": ["e1"]},
    )

    graph = InMemoryGraphWithWeights()

    with (
        patch("cognee.tasks.memify.extract_feedback_qas.session_user") as extract_user_ctx,
        patch("cognee.tasks.memify.apply_feedback_weights.session_user") as apply_user_ctx,
        patch("cognee.tasks.memify.extract_feedback_qas.get_session_manager", return_value=sm),
        patch("cognee.tasks.memify.apply_feedback_weights.get_session_manager", return_value=sm),
        patch("cognee.tasks.memify.apply_feedback_weights.get_graph_engine", return_value=graph),
    ):
        extract_user_ctx.get.return_value = user
        apply_user_ctx.get.return_value = user

        first_items = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            first_items.append(item)

        first_result = await apply_feedback_weights(first_items, alpha=0.1)

        second_items = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            second_items.append(item)

    assert len(first_items) == 1
    assert first_result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == pytest.approx(0.55)
    assert second_items == []


@pytest.mark.asyncio
async def test_feedback_weights_mixed_success_keeps_false(session_manager_with_backend):
    sm = session_manager_with_backend
    user = _make_user()

    await sm.add_qa(
        user_id="u1",
        question="Q",
        context="C",
        answer="A",
        session_id="s1",
        feedback_score=5,
        used_graph_element_ids={"node_ids": ["n1"], "edge_ids": ["missing-edge"]},
    )

    graph = InMemoryGraphWithWeights()

    with (
        patch("cognee.tasks.memify.extract_feedback_qas.session_user") as extract_user_ctx,
        patch("cognee.tasks.memify.apply_feedback_weights.session_user") as apply_user_ctx,
        patch("cognee.tasks.memify.extract_feedback_qas.get_session_manager", return_value=sm),
        patch("cognee.tasks.memify.apply_feedback_weights.get_session_manager", return_value=sm),
        patch("cognee.tasks.memify.apply_feedback_weights.get_graph_engine", return_value=graph),
    ):
        extract_user_ctx.get.return_value = user
        apply_user_ctx.get.return_value = user

        items = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            items.append(item)

        result = await apply_feedback_weights(items, alpha=0.1)

    assert len(items) == 1
    assert result["processed"] == 1
    assert result["applied"] == 0

    entries = await sm.get_session(user_id="u1", session_id="s1", formatted=False)
    assert entries[0]["memify_metadata"][MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY] is False
