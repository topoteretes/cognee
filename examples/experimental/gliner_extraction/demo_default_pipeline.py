"""Showcase: the DEFAULT gliner_cognify pipeline, fully local intelligence.

No schema is configured anywhere. Run 1: a local LLM (Ollama gemma3 4B)
invents the dataset ontology in ONE call, it is cached and versioned, and
GLiNER2 extracts the graph. Run 2 (more data, same dataset): the cached
ontology loads with ZERO LLM calls. Search at the end is embeddings-only.

Requires: ollama running with gemma3 pulled (ollama pull gemma3).
"""

import asyncio
import json
import os
import pathlib
import sys
import time

DEMO_DIR = pathlib.Path(__file__).parent / ".cognee_gliner_default"
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(DEMO_DIR / "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(DEMO_DIR / "system"))
os.environ.setdefault("TELEMETRY_DISABLED", "1")
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["AUTO_FEEDBACK"] = "false"

# The ONLY LLM in the pipeline: a small local model for ontology discovery.
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "gemma3:latest")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from gliner2 import GLiNER2  # noqa: E402

from gliner_cognify import gliner_cognify  # noqa: E402

DATASET = "space_missions"
SCHEMA_CACHE = pathlib.Path(__file__).parent / ".gliner_schema_cache"

RUN_1_TEXT = """
The Artemis II mission will carry astronauts Reid Wiseman, Victor Glover,
Christina Koch, and Jeremy Hansen around the Moon aboard the Orion spacecraft,
launched by the Space Launch System rocket from Kennedy Space Center. NASA
partnered with Lockheed Martin, which manufactured the Orion capsule, while
Aerojet Rocketdyne supplied the RS-25 engines. The mission is scheduled for
2026 and will test the life support systems during a ten-day lunar flyby.
"""

RUN_2_TEXT = """
SpaceX's Starship completed its orbital test from Starbase in Texas, powered
by 33 Raptor engines burning liquid methane. Elon Musk stated the vehicle will
support the Artemis III lunar landing, for which NASA awarded SpaceX a $2.9
billion contract in 2021. Gwynne Shotwell oversees the Human Landing System
program together with mission commander Charlie Duke as advisor.
"""


def section(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}", flush=True)


def show_ontology():
    cache_file = SCHEMA_CACHE / f"{DATASET}.json"
    cache = json.loads(cache_file.read_text())
    print(f"ontology v{cache['version']}  (source: {cache['schema_source']})")
    print(f"  entity types:   {', '.join(sorted(cache['entity_types']))}")
    print(f"  relation types: {', '.join(sorted(cache['relation_types']))}")


async def show_graph():
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    entities = [p.get("name") for _, p in nodes if p.get("type") == "Entity"]
    print(f"graph: {len(nodes)} nodes / {len(edges)} edges, entities: {sorted(entities)}")


async def main():
    (SCHEMA_CACHE / f"{DATASET}.json").unlink(missing_ok=True)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    section("RUN 1 — no schema configured anywhere")
    t0 = time.time()
    await cognee.add(RUN_1_TEXT, dataset_name=DATASET)
    await gliner_cognify(datasets=[DATASET], extractor=extractor)
    print(f"[{time.time() - t0:.0f}s] first ingestion (includes ONE local gemma3 discovery call)")
    show_ontology()
    await show_graph()

    section("RUN 2 — more data, same dataset: cached ontology, ZERO LLM calls")
    t0 = time.time()
    await cognee.add(RUN_2_TEXT, dataset_name=DATASET)
    await gliner_cognify(datasets=[DATASET], extractor=extractor)
    print(f"[{time.time() - t0:.0f}s] second ingestion")
    show_ontology()
    await show_graph()

    section("LLM-free search over the discovered-ontology graph")
    results = await cognee.search(
        query_text="Who is flying around the Moon?", query_type=SearchType.CHUNKS, top_k=1
    )
    for r in results:
        for item in r.get("search_result", []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            print(f"  {text[:180]}")


if __name__ == "__main__":
    asyncio.run(main())
