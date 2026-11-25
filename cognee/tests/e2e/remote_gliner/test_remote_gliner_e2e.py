"""Remote GLiNER end to end (SDK-980): the whole GLiNER cognify pipeline against a real worker.

Skipped unless a worker is configured, e.g.::

    COGNEE_GLINER_TRANSPORT=http COGNEE_GLINER_ENDPOINT=http://localhost:8080 \\
        pytest cognee/tests/e2e/remote_gliner

``grpc`` (``http://host:50051``) and ``amqp`` (``amqp://user:pass@host:5672``) work
the same way. The worker must support windowing: gliner_worker main at or after
92aba3c (topoteretes/gliner_worker#4).
Embeddings are mocked and the databases live in a scratch directory, so no API
key is needed. A run that reaches the end prints ``REMOTE-GLINER-E2E-RAN``.

Everything runs in one coroutine: cognee's cached engines bind to the running loop.
"""

from __future__ import annotations

import os
import sys

import pytest

TRANSPORT = os.environ.get("COGNEE_GLINER_TRANSPORT", "local").strip().lower()
pytestmark = pytest.mark.skipif(
    TRANSPORT in ("", "local") or not os.environ.get("COGNEE_GLINER_ENDPOINT"),
    reason="COGNEE_GLINER_TRANSPORT / COGNEE_GLINER_ENDPOINT do not name a remote worker",
)

NAMES = [
    f"{first} {last}"
    for first, last in zip(
        ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farid", "Greta", "Hiro", "Ines"]
        + ["Jonas", "Kira", "Luca", "Mira", "Nadia", "Omar", "Priya", "Quinn", "Rosa"],
        ["Novak", "Okafor", "Silva", "Tanaka", "Muller", "Rossi"] * 3,
    )
]
# Far past the encoder's window: a pass without windowing loses most of these.
LONG_TEXT = " ".join(
    f"{name} moved to Lisbon last spring and joined a small design studio there. "
    "The team met every Tuesday to review sketches, budgets, and client feedback "
    "before lunch. Colleagues described the work as slow but rewarding."
    for name in NAMES
)
SHORT_TEXT = "Tim Cook is the chief executive of Apple Inc., which is based in Cupertino."


@pytest.mark.asyncio
async def test_remote_gliner_builds_the_graph_without_a_local_model(tmp_path):
    import cognee

    os.environ.update(MOCK_EMBEDDING="true", TELEMETRY_DISABLED="1")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))

    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.tasks.graph.gliner_demo import GlinerRunStats, get_gliner_demo_tasks
    from cognee.tasks.graph.gliner_demo.remote import close_shared_workers

    await cognee.add([LONG_TEXT, SHORT_TEXT], dataset_name="remote_gliner_e2e")

    # Label-bank schema: the probe runs on the worker too.
    stats = GlinerRunStats()
    tasks = await get_gliner_demo_tasks(chunk_size=8000, stats=stats)
    await cognee.run_custom_pipeline(
        tasks=tasks, dataset="remote_gliner_e2e", pipeline_name="cognify_pipeline"
    )

    assert stats.model, "the worker's model was not recorded"
    assert stats.chunks >= 2 and stats.nodes > 0 and stats.kept_edges > 0
    assert all(schema.source == "label_bank" for schema in stats.schemas_by_document.values())

    graph_nodes, _ = await (await get_graph_engine()).get_graph_data()
    node_names = {str(properties.get("name", "")).lower() for _, properties in graph_nodes}
    missing = [name for name in NAMES if name.lower() not in node_names]
    # Windowing on the worker reaches the end of the long chunk.
    assert not missing, f"names not extracted: {missing}"

    # Caller labels: one more run over the same data.
    caller_stats = GlinerRunStats()
    tasks = await get_gliner_demo_tasks(
        {"person": "", "organization": "", "location": ""},
        {"works_for": "", "located_in": ""},
        chunk_size=8000,
        stats=caller_stats,
    )
    await cognee.run_custom_pipeline(
        tasks=tasks, dataset="remote_gliner_e2e", pipeline_name="cognify_pipeline"
    )
    assert caller_stats.kept_edges > 0

    await close_shared_workers()
    assert "gliner2" not in sys.modules and "torch" not in sys.modules
    print(f"REMOTE-GLINER-E2E-RAN transport={TRANSPORT} model={stats.model}")
