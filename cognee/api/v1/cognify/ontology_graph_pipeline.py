import asyncio
from typing import Union, Dict, List, Tuple
from rdflib import Graph
from uuid import uuid4
from io import IOBase, BytesIO
from pydantic import Field

from cognee.low_level import DataPoint
from cognee.infrastructure.engine import Edge
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import create_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.task import Task
import hashlib
from uuid import UUID


# -----------------------------
# STEP 1: Load ontology triples
# -----------------------------
async def load_ontology_data(ontology_file: Union[str, bytes, IOBase], format: str) -> list[dict]:
    """Parses RDF/OWL ontology into subject-predicate-object triples."""
    g = Graph()
    if isinstance(ontology_file, bytes):
        ontology_file = BytesIO(ontology_file)
    if isinstance(ontology_file, IOBase):
        try:
            ontology_file.seek(0)
        except (OSError, AttributeError):
            # Some streams may not be seekable; continue with current position.
            pass
    try:
        g.parse(ontology_file, format=format)
    except Exception as e:
        raise ValueError(f"Failed to parse ontology file: {e}")

    triples = []
    for s, p, o in g:
        triples.append({
            "subject": str(s),
            "predicate": str(p),
            "object": str(o)
        })
    return triples


# -------------------------------------
# STEP 2: Convert RDF triples to DataPoints
# -------------------------------------
class OntologyEntity(DataPoint):
    """
    Represents an ontology resource as a Cognee DataPoint.

    `related_to` stores outgoing relationships as tuples of (Edge metadata, target entity).
    """

    name: str
    uri: str
    related_to: List[Tuple[Edge, "OntologyEntity"]] = Field(default_factory=list)
    metadata: dict = {"index_fields": ["name"]}


OntologyEntity.model_rebuild()


def _extract_label(uri: str) -> str:
    """Return the local name for a URI (last fragment or path component)."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1] or uri
    if "/" in uri:
        return uri.rstrip("/").rsplit("/", 1)[-1] or uri
    return uri


async def ontology_to_datapoints(triples: list[dict]) -> list[DataPoint]:
    """
    Converts parsed triples into Cognee DataPoints (entities + relations).
    This preserves the ontology's structure as a graph.
    """
    entities: Dict[str, OntologyEntity] = {}

    for t in triples:
        subj = t["subject"]
        pred = t["predicate"]
        obj = t["object"]

        # Create or reuse entities
        if subj not in entities:
            entities[subj] = OntologyEntity(
                 id=UUID(hashlib.md5(subj.encode()).hexdigest()),
                name=_extract_label(subj),
                uri=subj,
            )

        if obj not in entities:
            # Only create entities for URI references, not literals
            if t.get("object_type") == "URIRef":
                 entities[obj] = OntologyEntity(
                     id=uuid4(),
                     name=_extract_label(obj),
                     uri=obj,
                 )
            else:
                # Handle literals as edge properties or skip creating entity
                continue

        predicate_label = _extract_label(pred)
        edge = Edge(
            relationship_type=predicate_label,
            properties={"uri": pred},
        )
        if not any(
            existing_edge.relationship_type == predicate_label and target.uri == obj
            for existing_edge, target in entities[subj].related_to
        ):
            entities[subj].related_to.append((edge, entities[obj]))

    return list(entities.values())


# -------------------------------------
# STEP 3: Define the custom task function
# -------------------------------------
async def ontology_ingestion_task(inputs: list, format: str):
    """
    Custom Cognee Task: Ingest OWL/RDF ontology and store as structured DataPoints.
    """
    ontology_file = inputs[0]
    triples = await load_ontology_data(ontology_file, format)
    datapoints = await ontology_to_datapoints(triples)
    await add_data_points(datapoints)
    return datapoints


# -------------------------------------
# STEP 4: Build and run the pipeline
# -------------------------------------
async def run_ontology_pipeline(ontology_file: Union[str, bytes, IOBase], format: str = "xml"):
    import cognee
    from cognee.low_level import setup

    # Reset state for clean runs
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = await create_dataset("ontology_dataset", user, session)

    # Define your pipeline with the new custom task
    tasks = [
        Task(ontology_ingestion_task,format=format, task_config={"batch_size": 50}),
    ]

    async for status in run_tasks(tasks, dataset.id, ontology_file, user, "ontology_ingestion_pipeline"):
        yield status

