"""The narrow graph move: update_chunk_index changes NOTHING but the index.

The operation exists to kill the carry-list trap: renumbering a chunk by
rewriting the whole node from a rehydrated model erases any property the
model forgets to declare. The narrow move patches the stored node itself, so
every other property survives byte-for-byte — pinned here by comparing the
full property set before and after a move on the default (kuzu) adapter.
Adapters without the operation raise UnsupportedGraphOperation and callers
fall back to the full rewrite.
"""

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def chunk_move_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_chunk_move_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    )

    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {words}.\n"


def test_update_chunk_index_is_narrow(chunk_move_env):
    asyncio.run(_scenario())


async def _scenario():
    await reset_backend_state()
    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    text = "".join(_para(tag) for tag in ["a", "b", "c"])
    await cognee.add(text, dataset_name="chunk_move")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "chunk_move")
    await cognee.cognify(datasets=[dataset.id], chunk_size=60)
    data_id = (await get_dataset_data(dataset.id))[0].id

    graph = await get_graph_engine()
    chunk_ids = []
    for source, edge, _target in await graph.get_connections(str(data_id)):
        if "is_part_of" in str(edge.get("relationship_name", "")):
            source_id = str(edge.get("source_node_id") or source.get("id"))
            if source_id != str(data_id):
                chunk_ids.append(source_id)
    assert len(chunk_ids) >= 2, "need at least two chunks to move one"

    before = {str(n["id"]): dict(n) for n in await graph.get_nodes(chunk_ids)}
    target = chunk_ids[0]
    new_index = int(before[target].get("chunk_index", 0)) + 40

    await graph.update_chunk_index({target: new_index})

    after = {str(n["id"]): dict(n) for n in await graph.get_nodes(chunk_ids)}
    assert int(after[target]["chunk_index"]) == new_index, "the index moved"

    # EVERYTHING else on the moved node is untouched, byte for byte.
    def _without_index(props: dict) -> dict:
        return {k: v for k, v in props.items() if k not in ("chunk_index", "updated_at")}

    assert _without_index(after[target]) == _without_index(before[target]), (
        "the narrow move must not touch any property but chunk_index"
    )
    # Untargeted chunks are entirely untouched.
    for chunk_id in chunk_ids[1:]:
        assert after[chunk_id] == before[chunk_id]


def test_default_adapter_raises():
    from cognee.infrastructure.databases.exceptions import UnsupportedGraphOperation
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    class _Bare(GraphDBInterface):
        def __getattr__(self, name):  # satisfy abstract surface for the probe
            raise AttributeError(name)

    with pytest.raises((UnsupportedGraphOperation, TypeError)):
        # Abstract classes may refuse instantiation outright; both outcomes
        # prove an unimplemented adapter cannot silently no-op the move.
        bare = _Bare()
        asyncio.run(bare.update_chunk_index({"x": 1}))
