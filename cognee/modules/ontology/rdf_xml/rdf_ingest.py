"""Ingest an RDF T-Box + A-Box into the cognee memory graph, preserving URIs.

This is the "complementary layer" bridge: point cognee at an RDF knowledge base
(the caller's system of record) and get memory-graph nodes that keep the stable
external IRIs, rather than the default LLM-extraction path that canonicalizes
entities into a closed local vocabulary.

Design choices, aligned with an open-world setup:
- Node identity is derived from the IRI (not the label), so two individuals
  that happen to share a local name stay distinct, and re-ingesting the same
  RDF is idempotent.
- The full external IRI is preserved verbatim on ``DataPoint.ontology_uri`` so
  it can be linked out to other domains and round-tripped back to RDF (see
  ``cognee.modules.graph.rdf.export``).
- Nothing is fuzzy-matched or renamed into a local vocabulary here.

The pure builder ``build_datapoints_from_rdf`` takes a parsed ``rdflib.Graph``
and returns cognee ``DataPoint``s; ``load_rdf_graph`` handles file/format
parsing; ``ingest_rdf`` is a thin convenience that persists via the standard
storage task.
"""

from typing import IO, Any, Dict, List, Optional, Union

from rdflib import Graph, OWL, RDF, RDFS, URIRef
from rdflib.term import Node as RDFNode

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Entity, EntityType
from cognee.shared.logging_utils import get_logger

logger = get_logger("RDFIngest")


def _local_name(uri: RDFNode) -> str:
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.rstrip("/").split("/")[-1]


def _label_or_local_name(graph: Graph, subject: RDFNode) -> str:
    label = graph.value(subject, RDFS.label)
    if label is not None:
        text = str(label).strip()
        if text:
            return text
    return _local_name(subject)


def load_rdf_graph(source: Union[str, List[str], IO, List[IO], Graph]) -> Graph:
    """Parse ``source`` into an ``rdflib.Graph``.

    Accepts an already-parsed ``Graph`` (returned as-is) or any input the
    ``RDFLibOntologyResolver`` accepts (a file path, list of paths, or
    file-like object), reusing its robust multi-format parsing.
    """
    if isinstance(source, Graph):
        return source

    # Reuse the resolver's format detection / fallback parsing for files.
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

    resolver = RDFLibOntologyResolver(ontology_file=source)
    if resolver.graph is None:
        raise ValueError("Could not parse any RDF triples from the provided source.")
    return resolver.graph


def build_datapoints_from_rdf(rdf_graph: Graph) -> List[DataPoint]:
    """Turn a parsed RDF graph (T-Box + A-Box) into cognee ``DataPoint``s.

    Classes (``owl:Class`` and anything used as an ``rdfs:subClassOf`` target)
    become ``EntityType`` nodes; individuals (subjects typed by a known class)
    become ``Entity`` nodes linked to their class via ``is_a``. ``rdfs:subClassOf``
    becomes ``is_a`` between classes, and object-property assertions between
    individuals become relations named by the property's local name. Every node
    keeps its source IRI on ``ontology_uri``.

    Returns the ``EntityType`` and ``Entity`` nodes with ``is_a``/``relations``
    populated — ready to hand to ``add_data_points``.
    """
    # --- collect classes (T-Box) ---
    class_uris: set = set(rdf_graph.subjects(RDF.type, OWL.Class))
    class_uris.update(rdf_graph.objects(None, RDFS.subClassOf))
    class_uris.update(rdf_graph.subjects(RDFS.subClassOf, None))
    # Only keep IRIs (skip blank nodes / literals).
    class_uris = {uri for uri in class_uris if isinstance(uri, URIRef)}

    type_nodes: Dict[RDFNode, EntityType] = {}
    for class_uri in class_uris:
        name = _label_or_local_name(rdf_graph, class_uri)
        type_nodes[class_uri] = EntityType(
            id=EntityType.id_for(str(class_uri)),
            name=name,
            description=name,
            ontology_valid=True,
            ontology_uri=str(class_uri),
        )

    # class -> superclass (is_a between EntityTypes)
    for sub_uri, super_uri in rdf_graph.subject_objects(RDFS.subClassOf):
        if sub_uri in type_nodes and super_uri in type_nodes:
            type_nodes[sub_uri].relations.append((_is_a_edge(), type_nodes[super_uri]))

    # --- collect individuals (A-Box): subjects typed by a known class ---
    entity_nodes: Dict[RDFNode, Entity] = {}
    individual_types: Dict[RDFNode, List[RDFNode]] = {}
    for subj, obj in rdf_graph.subject_objects(RDF.type):
        if isinstance(subj, URIRef) and obj in type_nodes:
            individual_types.setdefault(subj, []).append(obj)

    for ind_uri, types in individual_types.items():
        name = _label_or_local_name(rdf_graph, ind_uri)
        primary_type = type_nodes[types[0]]
        entity_nodes[ind_uri] = Entity(
            id=Entity.id_for(str(ind_uri)),
            name=name,
            is_a=primary_type,
            description=name,
            ontology_valid=True,
            ontology_uri=str(ind_uri),
        )
        # Additional class memberships beyond the primary become is_a relations.
        for extra_type in types[1:]:
            entity_nodes[ind_uri].relations.append((_is_a_edge(), type_nodes[extra_type]))

    # --- object-property assertions between individuals ---
    for subj, pred, obj in rdf_graph:
        if pred in (RDF.type, RDFS.subClassOf) or pred == RDFS.label:
            continue
        if subj in entity_nodes and obj in entity_nodes:
            relationship_name = _local_name(pred)
            entity_nodes[subj].relations.append((_edge(relationship_name), entity_nodes[obj]))

    data_points: List[DataPoint] = []
    data_points.extend(type_nodes.values())
    data_points.extend(entity_nodes.values())
    logger.info(
        "Built %d datapoints from RDF (%d classes, %d individuals)",
        len(data_points),
        len(type_nodes),
        len(entity_nodes),
    )
    return data_points


def _is_a_edge():
    from cognee.infrastructure.engine.models.Edge import Edge

    return Edge(relationship_type="is_a")


def _edge(relationship_name: str):
    from cognee.infrastructure.engine.models.Edge import Edge

    return Edge(relationship_type=relationship_name)


async def ingest_rdf(
    source: Union[str, List[str], IO, List[IO], Graph],
    ctx: Optional[Any] = None,
) -> List[DataPoint]:
    """Parse ``source`` and persist the resulting nodes/edges into the graph.

    Thin convenience over ``build_datapoints_from_rdf`` + the standard
    ``add_data_points`` storage task. Returns the ingested data points.
    """
    from cognee.tasks.storage.add_data_points import add_data_points

    rdf_graph = load_rdf_graph(source)
    data_points = build_datapoints_from_rdf(rdf_graph)
    if not data_points:
        logger.warning("No datapoints produced from RDF source; nothing to ingest.")
        return []
    return await add_data_points(data_points, ctx=ctx)
