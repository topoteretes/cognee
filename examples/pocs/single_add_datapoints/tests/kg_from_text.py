"""
Create a KnowledgeGraph from text via LLM (as in cognify), save/load to JSON, run expand with empty chunk.
"""

import asyncio
import json
from pathlib import Path

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.modules.data.processing.document_types import Document
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.utils.expand_with_nodes_and_edges import expand_with_nodes_and_edges
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points, index_graph_edges

from examples.pocs.single_add_datapoints.poc_expand_with_nodes_and_edges import (
    poc_expand_with_nodes_and_edges,
)

DEMO_JSON = Path(__file__).resolve().parent / "data" / "demo_kg.json"
USE_CACHED_KG = False  # Set to False to call LLM and overwrite JSON
USE_POC = False  # Set to False for production expand

SAMPLE_TEXT = """
Alice is from New York. Alice knows Bob. Bob is from Berlin.
Bob is a Data Scientist. Alice is a Data Scientist, too.
"""


async def create_kg_from_text(
    text: str,
    json_path: str | Path | None = None,
    custom_prompt: str | None = None,
) -> KnowledgeGraph:
    """Extract knowledge graph from text using LLM (same as cognify), optionally save to JSON, return kg."""
    kg = await extract_content_graph(text, KnowledgeGraph, custom_prompt=custom_prompt)
    valid_node_ids = {node.id for node in kg.nodes}
    kg.edges = [
        e
        for e in kg.edges
        if e.source_node_id in valid_node_ids and e.target_node_id in valid_node_ids
    ]
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kg.model_dump_json(indent=2), encoding="utf-8")
    return kg


def load_kg_from_json(json_path: str | Path) -> KnowledgeGraph:
    """Load a KnowledgeGraph from a JSON file (same shape as created by create_kg_from_text)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return KnowledgeGraph.model_validate(data)


def make_empty_chunk(text: str = "") -> DocumentChunk:
    """Chunk with no contains; expand will populate it from the graph."""
    doc = Document(
        name="doc",
        id=generate_node_id("doc"),
        raw_data_location="tmp/doc.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    return DocumentChunk(
        id=generate_node_id("doc_chunk"),
        text=text or SAMPLE_TEXT,
        chunk_size=100,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=doc,
        contains=[],
    )


async def add_datapoints_with_poc(
    chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> None:
    """POC: expand chunks with graph, then add_data_points on the chunks."""
    poc_expand_with_nodes_and_edges(chunks, chunk_graphs, ontology_resolver)
    await add_data_points(chunks)


async def add_datapoints_without_poc(
    chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> None:
    """Production: expand, add_data_points on graph_nodes, then add and index graph_edges."""
    graph_nodes, graph_edges = expand_with_nodes_and_edges(chunks, chunk_graphs, ontology_resolver)
    if graph_nodes:
        await add_data_points(graph_nodes)
    if graph_edges:
        graph_engine = await get_graph_engine()
        await graph_engine.add_edges(graph_edges)
        await index_graph_edges(graph_edges)


async def run_with_kg(
    kg: KnowledgeGraph,
    use_poc: bool,
    ontology_resolver: BaseOntologyResolver = None,
    prune_first: bool = True,
) -> None:
    """Run expand (poc or original) with an empty chunk and the given graph, then add_data_points."""
    if prune_first:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    chunk = make_empty_chunk()
    chunk_graphs = [kg]

    if use_poc:
        await add_datapoints_with_poc([chunk], chunk_graphs, ontology_resolver)
    else:
        await add_datapoints_without_poc([chunk], chunk_graphs, ontology_resolver)


async def get_demo_kg(
    use_cached: bool = USE_CACHED_KG,
    demo_json: Path = DEMO_JSON,
    sample_text: str = SAMPLE_TEXT,
):
    """Load KG from JSON if use_cached and file exists, else create via LLM and save. Returns kg."""
    if use_cached and demo_json.exists():
        kg = load_kg_from_json(demo_json)
        print(f"Loaded KG from {demo_json} ({len(kg.nodes)} nodes, {len(kg.edges)} edges)")
    else:
        kg = await create_kg_from_text(sample_text, json_path=demo_json)
        print(f"Created KG and saved to {demo_json} ({len(kg.nodes)} nodes, {len(kg.edges)} edges)")
    return kg


async def main():
    kg = await get_demo_kg()
    await run_with_kg(kg, use_poc=USE_POC)


if __name__ == "__main__":
    asyncio.run(main())
