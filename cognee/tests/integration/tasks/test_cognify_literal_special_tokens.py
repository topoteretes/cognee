"""add + cognify of a document that spells tiktoken special tokens (CLO-1384).

A document containing ``<|endoftext|>`` or ``<|fim_prefix|>`` used to fail cognify
for the whole dataset: chunking counts tokens with the embedding engine's
TikToken tokenizer, and tiktoken's default ``encode()`` raises on any special-token
spelling. The text must be tokenized as ordinary text and kept verbatim.

Runs on the default local stack (kuzu + lancedb + sqlite) with an OpenAI embedding
model (so the TikToken tokenizer is the one under test), mocked embeddings and a
mocked LLM — CI-safe, no API keys.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

DOCUMENT = (
    "Release notes for the tokenizer service.\n"
    "Prompts are terminated with the literal marker <|endoftext|> in the training data.\n"
    "Fill-in-the-middle requests start with <|fim_prefix|> before the code prefix.\n"
)


@pytest.fixture(scope="module")
def special_tokens_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_special_tokens_test_"))

    import cognee  # (cognee's import runs load_dotenv(override=True))

    def clear_config_caches():
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

    mp = pytest.MonkeyPatch()
    for key, value in {
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "CACHE_BACKEND": "sqlite",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "openai/text-embedding-3-large",
        "EMBEDDING_DIMENSIONS": "3072",
        "MOCK_EMBEDDING": "true",
        "GRAPH_EXTRACTOR": "llm",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "TELEMETRY_DISABLED": "1",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    }.items():
        mp.setenv(key, value)
    clear_config_caches()

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            return KnowledgeGraph(nodes=[], edges=[])
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


def test_cognify_keeps_literal_special_tokens(special_tokens_env):
    asyncio.run(_scenario())


async def _scenario():
    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    from cognee.infrastructure.llm.tokenizer.TikToken.adapter import TikTokenTokenizer
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
    from cognee.modules.users.methods import get_default_user

    assert isinstance(get_embedding_engine().tokenizer, TikTokenTokenizer), (
        "the scenario must chunk with the TikToken tokenizer"
    )

    await cognee.add(DOCUMENT, dataset_name="special_tokens")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "special_tokens")

    runs = await cognee.cognify(datasets=[dataset.id])

    assert all(isinstance(run, PipelineRunCompleted) for run in runs.values()), runs

    (row,) = await get_dataset_data(dataset.id)
    graph = await get_graph_engine()
    chunk_ids = []
    for source, edge, _target in await graph.get_connections(str(row.id)):
        if "is_part_of" in str(edge.get("relationship_name", "")):
            source_id = str(edge.get("source_node_id") or source.get("id"))
            if source_id != str(row.id):
                chunk_ids.append(source_id)
    assert chunk_ids, "chunks attached to the document node"
    chunk_text = "".join(str(node.get("text", "")) for node in await graph.get_nodes(chunk_ids))

    assert "<|endoftext|>" in chunk_text
    assert "<|fim_prefix|>" in chunk_text
