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

Session memory is covered too: ``remember(session_id=...)`` and the
``improve()`` bridge it starts must run keyless, since none of its stages calls
an LLM for plain Q&A entries (SDK-753).

TODO: expand this suite — the CLI (``cognee-cli remember`` / ``recall``) and
REST default search types once they are keyless-aware (SDK-690), and the
smaller ``fastino/gliner2.5-small-v1`` model if it becomes the default.
"""

import asyncio
import logging
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
# AUTO_FEEDBACK stays at its default: the per-turn analysis and the LLM-only improve
# stages skip on their own when no usable LLM is configured (SDK-753).
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


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


LOG_CAPTURE = _LogCapture()
logging.getLogger().addHandler(LOG_CAPTURE)


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

    # Session memory: the entry lands in the session cache and the background
    # improve() bridges it into the graph. Its stages build their own pipelines,
    # which must declare no LLM need, or the setup layer's LLM probe fails the
    # bridge before any task runs (SDK-753).
    session_result = await cognee.remember(
        "Niels Bohr proposed the Bohr model of the atom in 1913 in Copenhagen.",
        dataset_name="keyless_remember",
        session_id="keyless_session",
    )
    await session_result  # wait for the background bridge
    bridge = session_result.improve
    assert bridge is not None and bridge.status != "errored", bridge
    by_stage = {stage.stage: stage for stage in bridge.stages}
    assert by_stage["persist_session_qa"].status in ("completed", "already_completed"), by_stage[
        "persist_session_qa"
    ]
    assert not [s for s in bridge.stages if s.status == "errored"], bridge.stages
    bridged = await cognee.search(
        "Who proposed the Bohr model?",
        query_type=SearchType.CHUNKS,
        datasets=["keyless_remember"],
        top_k=3,
    )
    assert any("Bohr" in str(getattr(r, "result", r)) for r in bridged), bridged
    # An explicit improve over the same session must not raise either.
    explicit = await cognee.improve(dataset="keyless_remember", session_ids=["keyless_session"])
    assert explicit.status != "errored", explicit

    # Cross the periodic agent-context extraction threshold: every trace gets a
    # deterministic summary, and the due batch declines before building an LLM
    # client. Stage 3 then persists the traces while the LLM-only stages skip.
    from cognee.infrastructure.session.agent_context_extraction import TRACE_EXTRACTION_INTERVAL
    from cognee.memory.entries import TraceEntry

    for step in range(TRACE_EXTRACTION_INTERVAL):
        trace_result = await cognee.remember(
            TraceEntry(
                origin_function="search_docs",
                status="success",
                method_params={"query": "Bohr model"},
                method_return_value={"hits": step + 1},
                generate_feedback_with_llm=True,
            ),
            dataset_name="keyless_remember",
            session_id="keyless_session",
        )
        await trace_result
    traced = await cognee.improve(dataset="keyless_remember", session_ids=["keyless_session"])
    traced_stages = {stage.stage: stage for stage in traced.stages}
    assert traced.status != "errored", traced
    assert not [s for s in traced.stages if s.status == "errored"], traced.stages
    assert traced_stages["persist_agent_traces"].status in ("completed", "already_completed"), (
        traced_stages["persist_agent_traces"]
    )
    for llm_stage in ("extract_agent_context", "distill_sessions"):
        assert (traced_stages[llm_stage].status, traced_stages[llm_stage].reason) == (
            "skipped",
            "no_llm_configured",
        ), traced_stages[llm_stage]

    # With no usable LLM key, recall() without a query_type answers with CHUNKS.
    recalled = await cognee.recall("Where was Marie Curie born?", datasets=["keyless"], top_k=1)
    assert recalled and recalled[0].search_type == "CHUNKS", [
        getattr(r, "search_type", None) for r in recalled
    ]
    missing_key_logs = [
        message for message in LOG_CAPTURE.messages if "LLMAPIKeyNotSetError" in message
    ]
    assert not missing_key_logs, missing_key_logs
    print("keyless e2e: PASS")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as error:
        print(f"keyless e2e: FAIL — {error}")
        sys.exit(1)
