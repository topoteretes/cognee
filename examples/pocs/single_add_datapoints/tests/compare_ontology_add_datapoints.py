"""
Run non-POC and POC add_data_points with the same KG, query graph DB for nodes/edges, compare.
"""

import asyncio
from pathlib import Path

from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from kg_from_text import get_demo_kg, run_with_kg
from utils import _get_graph_snapshot, _compare

ONTOLOGY_PATH = Path(__file__).resolve().parent / "data" / "simple_ontology.owl"


async def run_with_kg_and_snapshot(
    kg,
    use_poc,
    ontology_path: Path = ONTOLOGY_PATH,
) -> tuple[set, set]:
    """Prune, run_with_kg(use_poc=False), return normalized nodes and edges."""
    await run_with_kg(
        kg,
        use_poc=use_poc,
        ontology_resolver=RDFLibOntologyResolver(str(ontology_path)),
    )
    return await _get_graph_snapshot()


async def compare_kg_from_text_runs():
    kg = await get_demo_kg(
        use_cached=False,
        sample_text="Qubits use superposition. Qubits can have either values 0, 1 or be in superposition.",
    )

    print("Running non-POC...")
    non_poc_nodes, non_poc_edges = await run_with_kg_and_snapshot(kg, use_poc=False)
    print(f"  non-POC: {len(non_poc_nodes)} nodes, {len(non_poc_edges)} edges")

    print("Running POC...")
    poc_nodes, poc_edges = await run_with_kg_and_snapshot(kg, use_poc=True)
    print(f"  POC: {len(poc_nodes)} nodes, {len(poc_edges)} edges")

    _compare("non-POC nodes", non_poc_nodes, "POC nodes", poc_nodes)
    _compare("non-POC edges", non_poc_edges, "POC edges", poc_edges)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(compare_kg_from_text_runs())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
