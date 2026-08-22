"""Build GLiNER extraction schemas from an OWL ontology.

GLiNER must be told what to extract (it is schema-driven, not open-ended).
Instead of hardcoding label lists, derive them from the ontology cognee
already supports:

  - OWL classes           -> GLiNER entity types (rdfs:comment as description)
  - OWL object properties -> GLiNER relation types (rdfs:comment as description)

For entities the ontology does NOT model, `discover_additional_types` makes
ONE LLM call per dataset (not per chunk) over a text sample and proposes
extra labels to merge into the GLiNER schema — open-ended discovery at
per-dataset cost, deterministic extraction at per-chunk scale.

Note: when ONTOLOGY_FILE_PATH is set, cognee's `extract_graph_from_data`
additionally canonicalizes and enriches the extracted graph against the same
ontology (matching extracted entities to OWL individuals/classes) — that
mechanism applies to GLiNER output unchanged.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, Graph

from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver


def _label_and_comment(graph: Graph, uri) -> Tuple[str, Optional[str]]:
    label = graph.value(uri, RDFS.label)
    name = str(label) if label else str(uri).split("#")[-1].split("/")[-1]
    comment = graph.value(uri, RDFS.comment)
    return name.strip().lower().replace(" ", "_"), (str(comment) if comment else None)


def gliner_schema_from_ontology(ontology_file: str) -> tuple[dict, dict]:
    """Derive (entity_types, relation_types) for GLiNER from an OWL file."""
    resolver = RDFLibOntologyResolver(ontology_file=ontology_file)
    graph = resolver.graph
    if graph is None:
        raise ValueError(f"Ontology could not be loaded from {ontology_file!r}")

    entity_types = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        name, comment = _label_and_comment(graph, cls)
        entity_types[name] = comment or f"Entities of type {name}"

    relation_types = {}
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        name, comment = _label_and_comment(graph, prop)
        relation_types[name] = comment or f"Relationship: {name}"

    return entity_types, relation_types


class ProposedType(BaseModel):
    """A proposed extraction label not covered by the ontology."""

    name: str = Field(description="snake_case label name")
    description: str = Field(description="One sentence: what text spans match this label")


class ProposedSchema(BaseModel):
    """Extra extraction labels proposed from a sample of the dataset."""

    entity_types: List[ProposedType]
    relation_types: List[ProposedType]


async def discover_additional_types(
    sample_texts: list[str],
    known_entity_types: dict,
    known_relation_types: dict,
    max_new_types: int = 8,
) -> tuple[dict, dict]:
    """ONE LLM call proposing labels the ontology does not cover.

    Returns (extra_entity_types, extra_relation_types) to merge into the
    GLiNER schema. Requires a working LLM configuration.
    """
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    sample = "\n---\n".join(sample_texts)[:8000]
    proposed = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Known entity types: {sorted(known_entity_types)}\n"
            f"Known relation types: {sorted(known_relation_types)}\n\n"
            f"Sample of the dataset:\n{sample}"
        ),
        system_prompt=(
            "You configure a span-extraction model. From the sample, propose up to "
            f"{max_new_types} ADDITIONAL entity types and relation types that appear "
            "in the data but are NOT in the known lists. Only propose types that "
            "occur repeatedly. snake_case names, one-sentence descriptions."
        ),
        response_model=ProposedSchema,
    )

    extra_entities = {
        t.name: t.description for t in proposed.entity_types if t.name not in known_entity_types
    }
    extra_relations = {
        t.name: t.description for t in proposed.relation_types if t.name not in known_relation_types
    }
    return extra_entities, extra_relations
