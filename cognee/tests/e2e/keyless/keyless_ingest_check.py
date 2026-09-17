"""Keyless end to end: no LLM key, no embedding settings, real local models.

Proves the default routing a fresh install gets with nothing configured:
``GRAPH_EXTRACTOR=auto`` resolves to the GLiNER demo, embeddings resolve to fastembed,
and ``add -> cognify -> search(CHUNKS) -> visualize`` and ``remember`` all
complete without a single provider call. Needs ``cognee[gliner]``; models
download on first run (GLiNER ~750 MB, bge-small ~67 MB).

Isolation: every ``LLM_*`` / ``EMBEDDING_*`` / ``OPENAI_*`` variable is dropped
from the environment, the repo ``.env`` (which ``cognee/__init__.py`` loads)
is neutralised, and the preflight skip vars are cleared, because they also
disable the keyless rerouting (see ``keyless_local_defaults_apply``).

Run: ``python cognee/tests/e2e/keyless/keyless_ingest_check.py``

TODO: expand this suite — session-scoped ``remember(session_id=...)`` and the
``improve()`` stages it triggers (SDK-691), the CLI (``cognee-cli remember`` /
``recall``) and REST default search types once they are keyless-aware
(SDK-690), and the smaller ``fastino/gliner2.5-small-v1`` model if it becomes
the default.
"""

import asyncio
import os
import sys
from pathlib import Path

for key in list(os.environ):
    if key.startswith(("LLM_", "EMBEDDING_", "OPENAI_", "GRAPH_EXTRACTOR", "BAML_")):
        os.environ.pop(key)
for key in ("COGNEE_SKIP_PREFLIGHT", "COGNEE_SKIP_CONNECTION_TEST", "MOCK_EMBEDDING"):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parent / ".keyless_run"
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)  # pydantic-settings reads a cwd-relative .env; there is none here
os.environ["DATA_ROOT_DIRECTORY"] = str(ROOT / "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = str(ROOT / "system")
os.environ["AUTO_FEEDBACK"] = "false"  # the per-turn feedback analysis is an LLM call
os.environ["TELEMETRY_DISABLED"] = "1"

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False  # cognee/__init__.py would load the repo .env

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402

TEXT = (
    "Marie Curie was born in Warsaw and worked at the University of Paris. "
    "She won the Nobel Prize in Physics in 1903 together with Pierre Curie and Henri Becquerel. "
    "Pierre Curie was a professor at the Sorbonne."
)
STRUCTURAL_EDGES = {"contains", "is_part_of", "made_from"}


async def main() -> None:
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.modules.cognify.config import get_cognify_config, resolve_extractor

    assert not get_llm_config().llm_api_key, "an LLM key leaked into the environment"
    extractor = resolve_extractor(None, get_cognify_config())
    assert extractor == "gliner_demo", (
        f"expected the gliner_demo extractor without a key, got {extractor}"
    )
    engine = get_embedding_engine()
    assert type(engine).__name__ == "FastembedEmbeddingEngine", type(engine).__name__
    print(f"routing: extractor={extractor} embeddings={engine.model} ({engine.get_vector_size()}d)")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(TEXT, dataset_name="keyless")
    await cognee.cognify(["keyless"])

    chunks = await cognee.search(
        "Where was Marie Curie born?", query_type=SearchType.CHUNKS, datasets=["keyless"], top_k=3
    )
    assert chunks, "CHUNKS search returned nothing"

    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    entities = sorted(
        {n[1].get("name") for n in nodes if n[1].get("type") == "Entity" and n[1].get("name")}
    )
    relations = sorted({e[2] for e in edges})
    print(f"graph: {len(nodes)} nodes, {len(edges)} edges")
    print(f"entities: {entities}")
    print(f"relationship types: {relations}")
    assert len(entities) >= 3, f"too few entities extracted: {entities}"
    assert any(name in {"marie curie", "warsaw"} for name in entities), entities
    assert set(relations) - STRUCTURAL_EDGES, f"no extracted relationships: {relations}"

    html_path = ROOT / "graph.html"
    await cognee.visualize_graph(str(html_path), dataset="keyless", full=True)
    assert html_path.stat().st_size > 0, "visualization is empty"

    # remember() is the primary API: same routing, then improve() with no
    # session ids, which runs no LLM stage.
    await cognee.remember(
        "Henri Becquerel discovered radioactivity in 1896 in Paris.",
        dataset_name="keyless_remember",
    )
    remembered = await cognee.search(
        "Who discovered radioactivity?",
        query_type=SearchType.CHUNKS,
        datasets=["keyless_remember"],
        top_k=1,
    )
    assert remembered, "remember() data is not searchable"

    # With no usable LLM key, recall() without a query_type answers with CHUNKS.
    recalled = await cognee.recall("Where was Marie Curie born?", datasets=["keyless"], top_k=1)
    assert recalled and recalled[0].search_type == "CHUNKS", [
        getattr(r, "search_type", None) for r in recalled
    ]
    print("keyless e2e: PASS")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as error:
        print(f"keyless e2e: FAIL — {error}")
        sys.exit(1)
