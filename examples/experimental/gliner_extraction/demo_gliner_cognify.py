"""Demo: cognee cognify with GLiNER2 doing ALL intelligence — zero LLM calls.

The LLM API key is deliberately replaced with a fake value before cognee
loads: if any pipeline stage attempted an LLM completion it would fail
loudly. Only embeddings receive the real key (an embedding model, not an
LLM). Requires: pip install cognee[gliner]  (or: pip install gliner2[local])
"""

import asyncio
import os
import pathlib
import sys

DEMO_DIR = pathlib.Path(__file__).parent / ".cognee_gliner"
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(DEMO_DIR / "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(DEMO_DIR / "system"))
os.environ.setdefault("TELEMETRY_DISABLED", "1")
# Skip the first-run LLM connectivity ping (the LLM key is intentionally broken).
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
# Per-query feedback analysis is an LLM call — keep searches LLM-free too.
os.environ["AUTO_FEEDBACK"] = "false"

_real_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
os.environ["EMBEDDING_API_KEY"] = _real_key
os.environ["LLM_API_KEY"] = "sk-FAKE-proof-that-no-llm-is-called"
os.environ["OPENAI_API_KEY"] = "sk-FAKE-proof-that-no-llm-is-called"

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from gliner2 import GLiNER2  # noqa: E402

from gliner_cognify import gliner_cognify  # noqa: E402

TEXT = """
Elon Musk founded SpaceX in 2002 in Hawthorne, California. SpaceX created the
Starship rocket and the Falcon 9 rocket. Gwynne Shotwell works for SpaceX as
President. SpaceX acquired Swarm Technologies in 2021.

Sam Altman works for OpenAI, which is located in San Francisco. OpenAI created
ChatGPT in 2022. Microsoft, led by Satya Nadella, invested in OpenAI.
"""


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    await cognee.add(TEXT, dataset_name="gliner_demo")
    # auto_schema=False: this demo deliberately runs with a broken LLM key,
    # so LLM ontology discovery is off; the generic default labels are used.
    # With a working LLM config, drop the flag to get the default pipeline
    # (cached per-dataset ontology, discovered by one LLM call).
    await gliner_cognify(datasets=["gliner_demo"], extractor=extractor, auto_schema=False)

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    print(f"graph: {len(nodes)} nodes, {len(edges)} edges")
    for _, props in nodes:
        if props.get("type") == "TextSummary":
            print(f"summary: {props.get('text')}")

    results = await cognee.search(
        query_text="Who acquired what?", query_type=SearchType.SUMMARIES, top_k=2
    )
    print(f"LLM-free search: {results}")


if __name__ == "__main__":
    asyncio.run(main())
