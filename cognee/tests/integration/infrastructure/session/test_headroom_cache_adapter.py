"""HeadroomCacheAdapter: FS cache stays authoritative, QA turns are mirrored into a
Headroom memory store and kept in sync on update/delete/prune.

The Headroom backend is faked so the suite runs without `headroom-ai` installed; the
fake mimics the LocalBackend surface the adapter uses (save_memory / delete_memory /
close).
"""

import importlib
import tempfile
import types
import uuid
from unittest.mock import patch

import pytest

# The package __init__ re-exports the class under the module's name; import the module itself.
ADAPTER_MODULE = "cognee.infrastructure.databases.cache.headroom.HeadroomCacheAdapter"


class FakeHeadroomBackend:
    """Minimal stand-in for headroom.memory.backends.local.LocalBackend."""

    def __init__(self):
        self.memories: dict[str, types.SimpleNamespace] = {}
        self.save_calls: list[dict] = []
        self.delete_calls: list[str] = []
        self.closed = False
        self.fail_save = False

    async def save_memory(self, *, content, user_id, session_id=None, metadata=None, **_):
        if self.fail_save:
            raise RuntimeError("embedder unavailable")
        memory = types.SimpleNamespace(
            id=str(uuid.uuid4()),
            content=content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        self.memories[memory.id] = memory
        self.save_calls.append(
            {"content": content, "user_id": user_id, "session_id": session_id, "metadata": metadata}
        )
        return memory

    async def delete_memory(self, memory_id, reason=None, user_id=None):
        self.delete_calls.append(memory_id)
        return self.memories.pop(memory_id, None) is not None

    async def close(self):
        self.closed = True


@pytest.fixture
def headroom_adapter():
    """HeadroomCacheAdapter rooted in a temp cache directory with a fake Headroom backend."""
    module = importlib.import_module(ADAPTER_MODULE)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ),
        patch.object(module, "_ensure_headroom_installed"),
    ):
        inst = module.HeadroomCacheAdapter(headroom_agent_name="cognee-unit")
        inst._headroom_backend = FakeHeadroomBackend()
        yield inst
        inst.cache.close()


def _backend(adapter) -> FakeHeadroomBackend:
    return adapter._headroom_backend


@pytest.mark.asyncio
async def test_create_qa_entry_writes_fs_and_mirrors_to_headroom(headroom_adapter):
    await headroom_adapter.create_qa_entry(
        user_id="u1",
        session_id="s1",
        question="What is cognee?",
        context="Background context here.",
        answer="An AI memory platform.",
        qa_id="qa-1",
    )

    entries = await headroom_adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0].qa_id == "qa-1"
    assert entries[0].context == "Background context here."

    backend = _backend(headroom_adapter)
    assert len(backend.save_calls) == 1
    call = backend.save_calls[0]
    assert call["content"] == "Q: What is cognee?\nA: An AI memory platform."
    assert call["user_id"] == "u1"
    assert call["session_id"] == "s1"
    assert call["metadata"] == {
        "source": "cognee",
        "agent": "cognee-unit",
        "cognee_user_id": "u1",
        "cognee_session_id": "s1",
        "qa_id": "qa-1",
    }
    # Retrieval context stays local; only the turn itself is shared.
    assert "Background context" not in call["content"]

    memory_id = next(iter(backend.memories))
    assert headroom_adapter._load_memory_ids("u1", "s1") == {"qa-1": memory_id}


@pytest.mark.asyncio
async def test_generated_qa_id_is_tracked(headroom_adapter):
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q?", context="", answer="A."
    )

    entries = await headroom_adapter.get_all_qa_entries("u1", "s1")
    mapping = headroom_adapter._load_memory_ids("u1", "s1")
    assert list(mapping) == [entries[0].qa_id]


@pytest.mark.asyncio
async def test_mirror_failure_does_not_break_fs_write(headroom_adapter):
    _backend(headroom_adapter).fail_save = True

    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q?", context="", answer="A.", qa_id="qa-2"
    )

    entries = await headroom_adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0].qa_id == "qa-2"
    assert headroom_adapter._load_memory_ids("u1", "s1") == {}


@pytest.mark.asyncio
async def test_update_of_answer_replaces_memory(headroom_adapter):
    backend = _backend(headroom_adapter)
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q?", context="", answer="A.", qa_id="qa-3"
    )
    old_memory_id = headroom_adapter._load_memory_ids("u1", "s1")["qa-3"]

    assert await headroom_adapter.update_qa_entry("u1", "s1", "qa-3", answer="A, revised.")

    assert backend.delete_calls == [old_memory_id]
    assert len(backend.memories) == 1
    (memory,) = backend.memories.values()
    assert memory.content == "Q: Q?\nA: A, revised."
    assert headroom_adapter._load_memory_ids("u1", "s1") == {"qa-3": memory.id}


@pytest.mark.asyncio
async def test_feedback_only_update_is_not_mirrored(headroom_adapter):
    backend = _backend(headroom_adapter)
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q?", context="", answer="A.", qa_id="qa-4"
    )

    assert await headroom_adapter.update_qa_entry(
        "u1", "s1", "qa-4", feedback_score=5, feedback_text="great"
    )
    assert await headroom_adapter.update_qa_entry("u1", "s1", "qa-4", context="new ctx")

    assert len(backend.save_calls) == 1
    assert backend.delete_calls == []


@pytest.mark.asyncio
async def test_update_of_unknown_qa_id_touches_nothing(headroom_adapter):
    backend = _backend(headroom_adapter)

    assert not await headroom_adapter.update_qa_entry("u1", "s1", "missing", answer="x")

    assert backend.save_calls == []
    assert backend.delete_calls == []


@pytest.mark.asyncio
async def test_delete_qa_entry_removes_its_memory(headroom_adapter):
    backend = _backend(headroom_adapter)
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q1?", context="", answer="A1.", qa_id="qa-5"
    )
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="s1", question="Q2?", context="", answer="A2.", qa_id="qa-6"
    )
    mapping = headroom_adapter._load_memory_ids("u1", "s1")

    assert await headroom_adapter.delete_qa_entry("u1", "s1", "qa-5")

    assert backend.delete_calls == [mapping["qa-5"]]
    assert set(backend.memories) == {mapping["qa-6"]}
    assert headroom_adapter._load_memory_ids("u1", "s1") == {"qa-6": mapping["qa-6"]}
    assert not await headroom_adapter.delete_qa_entry("u1", "s1", "qa-5")
    assert len(backend.delete_calls) == 1


@pytest.mark.asyncio
async def test_delete_session_removes_all_session_memories(headroom_adapter):
    backend = _backend(headroom_adapter)
    for i in range(3):
        await headroom_adapter.create_qa_entry(
            user_id="u1",
            session_id="s1",
            question=f"Q{i}?",
            context="",
            answer="A.",
            qa_id=f"qa-{i}",
        )
    await headroom_adapter.create_qa_entry(
        user_id="u1", session_id="other", question="Keep?", context="", answer="Yes.", qa_id="keep"
    )
    kept_id = headroom_adapter._load_memory_ids("u1", "other")["keep"]

    assert await headroom_adapter.delete_session("u1", "s1")

    assert set(backend.memories) == {kept_id}
    assert headroom_adapter._load_memory_ids("u1", "s1") == {}
    assert await headroom_adapter.get_all_qa_entries("u1", "s1") == []
    assert not await headroom_adapter.delete_session("u1", "s1")


@pytest.mark.asyncio
async def test_prune_deletes_only_mirrored_memories(headroom_adapter):
    backend = _backend(headroom_adapter)
    # A memory another agent wrote into the shared Headroom store.
    foreign = await backend.save_memory(content="from claude code", user_id="u1")
    for session in ("s1", "s2"):
        await headroom_adapter.create_qa_entry(
            user_id="u1",
            session_id=session,
            question="Q?",
            context="",
            answer="A.",
            qa_id=f"qa-{session}",
        )
    assert len(backend.memories) == 3

    await headroom_adapter.prune()

    assert set(backend.memories) == {foreign.id}
    assert await headroom_adapter.get_all_qa_entries("u1", "s1") == []
    assert headroom_adapter._load_memory_ids("u1", "s1") == {}


@pytest.mark.asyncio
async def test_close_closes_headroom_backend(headroom_adapter):
    backend = _backend(headroom_adapter)

    await headroom_adapter.close()

    assert backend.closed is True
    assert headroom_adapter._headroom_backend is None


def test_backend_config_is_built_from_adapter_settings(tmp_path):
    """The real LocalBackend is constructed from the adapter's settings (headroom required)."""
    pytest.importorskip("headroom.memory.backends.local", reason="headroom-ai not installed")
    module = importlib.import_module(ADAPTER_MODULE)

    with patch(
        "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
        return_value={"data_root_directory": str(tmp_path)},
    ):
        adapter = module.HeadroomCacheAdapter(
            headroom_db_path=str(tmp_path / "hr" / "memory.db"),
            headroom_embedder="openai",
            headroom_embedder_model="text-embedding-3-small",
            headroom_embedder_api_key="sk-test",
            headroom_vector_dimension=1536,
        )
        try:
            backend = adapter._get_backend()
            config = backend._config
            assert config.db_path == str(tmp_path / "hr" / "memory.db")
            assert (tmp_path / "hr").is_dir()
            assert config.embedder_backend == "openai"
            assert config.embedder_model == "text-embedding-3-small"
            assert config.openai_api_key == "sk-test"
            assert config.vector_dimension == 1536
            assert adapter._get_backend() is backend
        finally:
            adapter.cache.close()


def test_missing_headroom_raises_configuration_error(tmp_path):
    module = importlib.import_module(ADAPTER_MODULE)
    from cognee.infrastructure.databases.exceptions import HeadroomNotInstalledError

    with (
        patch.object(module.importlib.util, "find_spec", return_value=None),
        pytest.raises(HeadroomNotInstalledError) as exc_info,
    ):
        module.HeadroomCacheAdapter()

    assert "pip install headroom-ai" in str(exc_info.value)
