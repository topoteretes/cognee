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
- RDF object-property predicate IRIs are preserved on explicit custom edge
  properties as ``predicate_uri``. The internal relationship name stays a
  graph-safe storage label; ``predicate_uri`` is the source of truth for RDF
  export.
- ``is_a`` relationships map to ``rdf:type`` / ``rdfs:subClassOf`` during RDF
  export.
- Nothing is fuzzy-matched or renamed into a local vocabulary here.

Scope:
- RDF ingest preserves object-property assertions only when both endpoints are
  ingested individuals.
- Class declarations, blank nodes, reasoning, and arbitrary literal/data
  property round-trip are out of scope for this bridge.

The pure builder ``build_graph_from_rdf`` takes a parsed ``rdflib.Graph`` and
returns cognee ``DataPoint``s plus explicit RDF custom edges.
``build_datapoints_from_rdf`` is a compatibility wrapper for callers that only
need nodes. ``load_rdf_graph`` handles file/format parsing; ``ingest_rdf`` is a
thin convenience that persists via the standard storage task.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import IO, Any, Dict, List, Optional, Union

from rdflib import Graph, OWL, RDF, RDFS, URIRef
from rdflib.term import Node as RDFNode

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.engine.models import Entity, EntityType
from cognee.shared.logging_utils import get_logger

logger = get_logger("RDFIngest")

CustomEdge = tuple[str, str, str, dict[str, Any]]


@dataclass
class RDFIngestGraph:
    data_points: list[DataPoint]
    custom_edges: list[CustomEdge]


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


def _relationship_name_for_predicate(predicate: URIRef) -> str:
    local_name = generate_edge_name(_local_name(predicate) or "predicate")
    digest = sha256(str(predicate).encode("utf-8")).hexdigest()[:12]
    return f"rdf_{local_name}_{digest}"


def _predicate_edge_properties(predicate: URIRef) -> dict[str, Any]:
    return {
        "predicate_uri": str(predicate),
        "edge_text": _local_name(predicate),
    }


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


def _collect_class_uris(rdf_graph: Graph) -> set[URIRef]:
    class_uris: set = set(rdf_graph.subjects(RDF.type, OWL.Class))
    class_uris.update(rdf_graph.objects(None, RDFS.subClassOf))
    class_uris.update(rdf_graph.subjects(RDFS.subClassOf, None))
    return {uri for uri in class_uris if isinstance(uri, URIRef)}


def _build_class_nodes(rdf_graph: Graph) -> Dict[RDFNode, EntityType]:
    type_nodes: Dict[RDFNode, EntityType] = {}
    class_uris = _collect_class_uris(rdf_graph)
    for class_uri in class_uris:
        name = _label_or_local_name(rdf_graph, class_uri)
        type_nodes[class_uri] = EntityType(
            id=EntityType.id_for(str(class_uri)),
            name=name,
            description=name,
            ontology_valid=True,
            ontology_uri=str(class_uri),
        )
    return type_nodes


def _attach_subclass_relations(rdf_graph: Graph, type_nodes: Dict[RDFNode, EntityType]) -> None:
    for sub_uri, super_uri in rdf_graph.subject_objects(RDFS.subClassOf):
        if sub_uri in type_nodes and super_uri in type_nodes:
            type_nodes[sub_uri].relations.append((_is_a_edge(), type_nodes[super_uri]))


def _collect_individual_types(
    rdf_graph: Graph, type_nodes: Dict[RDFNode, EntityType]
) -> Dict[RDFNode, List[RDFNode]]:
    individual_types: Dict[RDFNode, List[RDFNode]] = {}
    for subj, obj in rdf_graph.subject_objects(RDF.type):
        if isinstance(subj, URIRef) and obj in type_nodes:
            individual_types.setdefault(subj, []).append(obj)
    return individual_types


def _build_individual_nodes(
    rdf_graph: Graph,
    individual_types: Dict[RDFNode, List[RDFNode]],
    type_nodes: Dict[RDFNode, EntityType],
) -> Dict[RDFNode, Entity]:
    entity_nodes: Dict[RDFNode, Entity] = {}
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
    return entity_nodes


def _attach_extra_type_relations(
    entity_nodes: Dict[RDFNode, Entity],
    individual_types: Dict[RDFNode, List[RDFNode]],
    type_nodes: Dict[RDFNode, EntityType],
) -> None:
    for ind_uri, types in individual_types.items():
        for extra_type in types[1:]:
            entity_nodes[ind_uri].relations.append((_is_a_edge(), type_nodes[extra_type]))


def _build_object_property_edge(
    source_node: Entity,
    target_node: Entity,
    predicate: URIRef,
) -> CustomEdge:
    relationship_name = _relationship_name_for_predicate(predicate)
    return (
        str(source_node.id),
        str(target_node.id),
        relationship_name,
        _predicate_edge_properties(predicate),
    )


def _build_object_property_edges(
    rdf_graph: Graph, entity_nodes: Dict[RDFNode, Entity]
) -> list[CustomEdge]:
    custom_edges: list[CustomEdge] = []
    for subj, pred, obj in rdf_graph:
        if pred in (RDF.type, RDFS.subClassOf) or pred == RDFS.label:
            continue
        if not isinstance(pred, URIRef):
            continue
        if subj not in entity_nodes or obj not in entity_nodes:
            continue
        custom_edges.append(
            _build_object_property_edge(entity_nodes[subj], entity_nodes[obj], pred)
        )
    return custom_edges


def build_graph_from_rdf(rdf_graph: Graph) -> RDFIngestGraph:
    """Turn a parsed RDF graph into cognee nodes plus RDF object-property edges.

    Classes become ``EntityType`` nodes. Individuals typed by known classes
    become ``Entity`` nodes. ``rdf:type`` and ``rdfs:subClassOf`` stay in the
    ``DataPoint`` structure as ``is_a`` relationships. RDF object-property
    assertions between ingested individuals become explicit custom graph edges
    with ``predicate_uri`` properties.
    """
    type_nodes = _build_class_nodes(rdf_graph)
    _attach_subclass_relations(rdf_graph, type_nodes)

    individual_types = _collect_individual_types(rdf_graph, type_nodes)
    entity_nodes = _build_individual_nodes(rdf_graph, individual_types, type_nodes)
    _attach_extra_type_relations(entity_nodes, individual_types, type_nodes)

    custom_edges = _build_object_property_edges(rdf_graph, entity_nodes)
    data_points: list[DataPoint] = [*type_nodes.values(), *entity_nodes.values()]

    logger.info(
        "Built %d datapoints and %d custom edges from RDF (%d classes, %d individuals)",
        len(data_points),
        len(custom_edges),
        len(type_nodes),
        len(entity_nodes),
    )
    return RDFIngestGraph(data_points=data_points, custom_edges=custom_edges)


def build_datapoints_from_rdf(rdf_graph: Graph) -> List[DataPoint]:
    """Compatibility wrapper returning only RDF-ingested ``DataPoint`` nodes."""
    return build_graph_from_rdf(rdf_graph).data_points


def _is_a_edge():
    from cognee.infrastructure.engine.models.Edge import Edge

    return Edge(relationship_type="is_a")


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
    ingest_graph = build_graph_from_rdf(rdf_graph)
    if not ingest_graph.data_points:
        logger.warning("No datapoints produced from RDF source; nothing to ingest.")
        return []
    return await add_data_points(
        ingest_graph.data_points,
        custom_edges=ingest_graph.custom_edges,
        ctx=ctx,
    )
