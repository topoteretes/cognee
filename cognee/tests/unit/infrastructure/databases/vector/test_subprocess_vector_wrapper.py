import asyncio

import pytest

from cognee.infrastructure.memory_cleanup import stop_memory_cleanup_manager

from cognee.infrastructure.databases.vector.subprocess_vector_wrapper import (
    SubprocessVectorDBWrapper,
)


@pytest.fixture(autouse=True)
def _reset_memory_cleanup_manager():
    yield
    stop_memory_cleanup_manager(reset=True)


class FakeVectorAdapter:
    def __init__(self, prefix="value"):
        self.prefix = prefix
        self.closed = False

    async def has_collection(self, collection_name: str) -> bool:
        return collection_name == "known"

    async def create_collection(self, collection_name: str, payload_schema=None):
        return {"collection": collection_name, "payload_schema": payload_schema}

    async def create_data_points(self, collection_name: str, data_points):
        return len(data_points)

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        return [{"id": data_point_id} for data_point_id in data_point_ids]

    async def search(
        self,
        collection_name: str,
        query_text=None,
        query_vector=None,
        limit=None,
        with_vector=False,
        include_payload=False,
        node_name=None,
    ):
        return [{"collection": collection_name, "query_text": query_text, "limit": limit}]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit=None,
        with_vectors=False,
        include_payload=False,
        node_name=None,
    ):
        return [{"query_text": query_text} for query_text in query_texts]

    async def delete_data_points(self, collection_name: str, data_point_ids):
        return len(data_point_ids)

    async def prune(self):
        return "pruned"

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return [[float(len(item))] for item in data]

    async def get_connection(self):
        return None

    async def get_collection(self, collection_name: str):
        return None

    async def create_vector_index(self, index_name: str, index_property_name: str):
        return None

    async def index_data_points(self, index_name: str, index_property_name: str, data_points):
        return None

    def get_data_point_schema(self, model_type):
        return model_type

    @classmethod
    async def create_dataset(cls, dataset_id=None, user=None):
        return {}

    async def delete_dataset(self, dataset_id, user):
        return None

    async def close(self):
        self.closed = True

    async def custom_method(self, suffix: str):
        return f"{self.prefix}:{suffix}"


@pytest.mark.asyncio
async def test_subprocess_vector_wrapper_proxies_interface_calls():
    wrapper = SubprocessVectorDBWrapper(
        FakeVectorAdapter,
        prefix="wrapped",
        initialize_embedding_engine=False,
    )

    try:
        assert await wrapper.has_collection("known") is True
        assert await wrapper.search(
            "collection",
            query_text="hello",
            query_vector=None,
            limit=3,
        ) == [{"collection": "collection", "query_text": "hello", "limit": 3}]
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_subprocess_vector_wrapper_proxies_adapter_specific_methods():
    wrapper = SubprocessVectorDBWrapper(
        FakeVectorAdapter,
        prefix="wrapped",
        initialize_embedding_engine=False,
    )

    try:
        assert await wrapper.custom_method("ok") == "wrapped:ok"
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_subprocess_vector_wrapper_rejects_missing_methods():
    wrapper = SubprocessVectorDBWrapper(
        FakeVectorAdapter,
        initialize_embedding_engine=False,
    )

    try:
        with pytest.raises(AttributeError):
            await wrapper.missing_method()
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_subprocess_vector_wrapper_updates_last_access_timestamp():
    wrapper = SubprocessVectorDBWrapper(
        FakeVectorAdapter,
        initialize_embedding_engine=False,
    )

    try:
        before = wrapper.last_accessed_ts()
        await asyncio.sleep(0.01)
        await wrapper.has_collection("known")
        assert wrapper.last_accessed_ts() > before
    finally:
        await wrapper.close()


@pytest.mark.asyncio
async def test_subprocess_vector_wrapper_reports_memory_and_supports_clean(monkeypatch):
    wrapper = SubprocessVectorDBWrapper(
        FakeVectorAdapter,
        initialize_embedding_engine=False,
    )

    try:
        monkeypatch.setattr(
            "cognee.infrastructure.databases.vector.subprocess_vector_wrapper.get_process_rss",
            lambda pid: 321 if pid == wrapper._proc.pid else 0,
        )
        assert wrapper.memory_used() == 321
        wrapper.clean()
        assert wrapper.memory_used() == 0
    finally:
        await wrapper.close()
