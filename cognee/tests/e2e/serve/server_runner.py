"""Boot a real cognee API server for the ``serve()`` e2e suite.

Run as a subprocess by ``test_serve_remote_routing.py``::

    python -m cognee.tests.e2e.serve.server_runner <port> <env-overrides.json>

A separate process is not a convenience, it is the point: the SDK under test
sets a process-global remote client, and the server's own handlers call the
same SDK functions. In one process every handler would see that client and
forward the request back to itself. Two processes are also what a real
deployment looks like — the SDK's local store and the server's store are
different directories, so the test can prove the SDK never touched its own.

LLM structured output and embeddings are mocked the same way the
incremental-update e2e suite mocks them: no API keys, deterministic graph.
"""

import json
import os
import sys
from pathlib import Path


def _clear_config_caches() -> None:
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


def _install_llm_mock() -> None:
    import re

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    marker = re.compile(r"ENT[A-Z0-9]+")

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(marker.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=f"marker {n}") for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    LLMGateway.acreate_structured_output = _mock_acreate


def main() -> None:
    port = int(sys.argv[1])
    overrides = json.loads(Path(sys.argv[2]).read_text())
    os.environ.update(overrides)

    # ``import cognee`` runs load_dotenv(override=True), which would let a repo
    # .env (say ENABLE_BACKEND_ACCESS_CONTROL=true) silently replace the
    # suite's explicit environment — and the auth posture is fixed at import
    # time. Keep .env as a fallback for anything unset, never an override.
    import dotenv

    original_load_dotenv = dotenv.load_dotenv

    def _load_dotenv_without_override(*args, **kwargs):
        kwargs["override"] = False
        return original_load_dotenv(*args, **kwargs)

    dotenv.load_dotenv = _load_dotenv_without_override

    import cognee  # noqa: F401

    os.environ.update(overrides)
    _clear_config_caches()
    _install_llm_mock()

    from cognee.api.client import start_api_server

    start_api_server(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
