import asyncio
from typing import List, Union, Optional
from rdflib import Graph
from uuid import uuid4
from datetime import datetime, timezone

from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points
from cognee.tasks.graph import extract_graph_from_data
from cognee.modules.data.methods import create_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document


# ---------- Step 1: Load Ontology (from file object) ----------
async def load_ontology_data(ontology_file) -> list[dict]:
    """
    Loads OWL/RDF ontology directly from a file-like object and extracts RDF triples.
    """
    g = Graph()
    g.parse(ontology_file, format="xml")  # Works for standard OWL/RDF/XML

    triples = []
    for s, p, o in g:
        triples.append({"subject": str(s), "predicate": str(p), "object": str(o)})
    return triples


# ---------- Step 2: Convert Triples into Chunks ----------
def convert_triples_to_chunks(triples: list[dict]) -> list[DocumentChunk]:
    """
    Convert ontology triples into Cognee-compatible DocumentChunk objects.
    """
    chunks = []

    # Minimal valid Document (from your class)
    ontology_doc = Document(
        id=uuid4(),
        name="in_memory_ontology.owl",
        raw_data_location="in_memory_source",
        external_metadata=None,
        mime_type="application/rdf+xml"
    )

    for i, t in enumerate(triples):
        text = f"{t['subject']} {t['predicate']} {t['object']}"
        chunk = DocumentChunk(
            id=uuid4(),
            text=text,
            chunk_size=len(text.split()),
            chunk_index=i,
            cut_type="triple",
            is_part_of=ontology_doc,
            metadata={"triple": t, "index_fields": ["text"]}
        )
        chunks.append(chunk)
    return chunks


# ---------- Step 3: Run Ontology Pipeline ----------
async def run_ontology_pipeline(ontology_file, dataset_name: str = "ontology_dataset"):
    """
    Run the ontology ingestion pipeline directly from a file object (no file path).
    """
    from cognee.low_level import setup
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = await create_dataset(dataset_name, user, session)

    # ✅ Process ontology file directly
    triples = await load_ontology_data(ontology_file)
    chunks = convert_triples_to_chunks(triples)

    # Define pipeline tasks
    tasks = [
        Task(
            extract_graph_from_data,
            graph_model=KnowledgeGraph,
            task_config={"batch_size": 20},
        ),
        Task(add_data_points, task_config={"batch_size": 20}),
    ]

    # Run tasks with chunks
    async for run_status in run_tasks(tasks, dataset.id, chunks, user, "ontology_pipeline"):
        yield run_status
