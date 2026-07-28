"""Export the cognee memory graph as RDF and query it with SPARQL.

cognee stores knowledge in a labeled property graph (Kuzu/Neo4j/etc.). This
module builds an RDF view over that graph so downstream consumers can query it
with SPARQL and link its identifiers out to other ontologies/datasets, without
being tied to one graph engine's query language.

Node identifiers are chosen open-world-first: a node that carries an
``ontology_uri`` (preserved end-to-end from ontology matching or RDF ingestion,
see ``DataPoint.ontology_uri``) is emitted under that stable external IRI;
un-grounded nodes get a minted cognee IRI under ``DEFAULT_BASE_IRI`` so the RDF
stays well-formed. RDF-ingested object-property edges that carry
``predicate_uri`` are emitted with that original RDF predicate IRI. Cognee-native
edges without ``predicate_uri`` keep minted cognee predicate IRIs. Nothing is
collapsed into a closed local vocabulary.
"""

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

from cognee.shared.logging_utils import get_logger

logger = get_logger("RDFExport")

# Base IRI for minted (non-ontology-grounded) cognee identifiers.
DEFAULT_BASE_IRI = "https://cognee.ai/graph/"

NodeData = Dict[str, Any]
Node = Tuple[str, NodeData]
EdgeData = Tuple[str, str, str, Dict[str, Any]]

# Relationship names that carry native RDF semantics rather than a minted
# cognee predicate. ``is_a`` links an individual to its class (rdf:type) or a
# class to a superclass (rdfs:subClassOf) — resolved per-edge from node type.
_IS_A_RELATIONSHIPS = {"is_a"}


def _minted_node_iri(node_id: str, base_iri: str) -> URIRef:
    return URIRef(f"{base_iri}node/{quote(str(node_id), safe='')}")


def _node_iri(node_id: str, props: NodeData, base_iri: str) -> URIRef:
    """Resolve a node's IRI: its preserved ontology IRI, else a minted one."""
    ontology_uri = props.get("ontology_uri") if props else None
    if ontology_uri:
        return URIRef(str(ontology_uri))
    return _minted_node_iri(node_id, base_iri)


def _predicate_iri(relationship_name: str, base_iri: str) -> URIRef:
    return URIRef(f"{base_iri}prop/{quote(str(relationship_name), safe='')}")


def _edge_predicate_iri(relationship_name: str, props: NodeData, base_iri: str) -> URIRef:
    predicate_uri = props.get("predicate_uri") if props else None
    if predicate_uri:
        return URIRef(str(predicate_uri))
    return _predicate_iri(relationship_name, base_iri)


def _is_entity_type(props: NodeData) -> bool:
    # DataPoint.__init__ pins ``type`` to the class name, so a class node is
    # stored with type == "EntityType".
    return bool(props) and props.get("type") == "EntityType"


def graph_data_to_rdf(
    nodes: List[Node],
    edges: List[EdgeData],
    base_iri: str = DEFAULT_BASE_IRI,
) -> Graph:
    """Build an RDF graph from property-graph ``(nodes, edges)``.

    Pure and side-effect free — takes the tuples returned by
    ``GraphDBInterface.get_graph_data`` and returns an ``rdflib.Graph``. Kept
    separate from the async DB wrappers so it is trivially testable.

    Args:
        nodes: ``(node_id, properties)`` tuples. ``properties`` may carry
            ``name`` (→ ``rdfs:label``), ``type``, and ``ontology_uri``.
        edges: ``(source_id, target_id, relationship_name, properties)`` tuples.
        base_iri: prefix for minted IRIs of un-grounded nodes and predicates.

    Returns:
        An ``rdflib.Graph`` view of the memory graph.
    """
    g = Graph()
    g.bind("cognee", Namespace(base_iri))
    g.bind("cprop", Namespace(f"{base_iri}prop/"))

    node_props: Dict[str, NodeData] = {str(nid): (props or {}) for nid, props in nodes}

    for node_id, props in nodes:
        props = props or {}
        subject = _node_iri(node_id, props, base_iri)

        name = props.get("name")
        if name:
            g.add((subject, RDFS.label, Literal(name)))

    for source_id, target_id, relationship_name, props in edges:
        source_props = node_props.get(str(source_id), {})
        target_props = node_props.get(str(target_id), {})

        subject = _node_iri(source_id, source_props, base_iri)
        obj = _node_iri(target_id, target_props, base_iri)

        if relationship_name in _IS_A_RELATIONSHIPS:
            # class -> superclass is subClassOf; individual -> class is rdf:type.
            predicate = RDFS.subClassOf if _is_entity_type(source_props) else RDF.type
        else:
            predicate = _edge_predicate_iri(relationship_name, props or {}, base_iri)

        g.add((subject, predicate, obj))

    return g


async def export_memory_graph_to_rdf(
    base_iri: str = DEFAULT_BASE_IRI,
    graph_engine: Optional[Any] = None,
) -> Graph:
    """Read the whole memory graph and return it as an ``rdflib.Graph``.

    Args:
        base_iri: prefix for minted IRIs of un-grounded nodes and predicates.
        graph_engine: an initialized graph adapter; resolved via
            ``get_graph_engine`` when omitted.
    """
    if graph_engine is None:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()

    nodes, edges = await graph_engine.get_graph_data()
    logger.info("Exporting memory graph to RDF: %d nodes, %d edges", len(nodes), len(edges))
    return graph_data_to_rdf(nodes, edges, base_iri=base_iri)


async def serialize_memory_graph(
    rdf_format: str = "turtle",
    base_iri: str = DEFAULT_BASE_IRI,
    graph_engine: Optional[Any] = None,
) -> str:
    """Serialize the memory graph to an RDF string (turtle by default)."""
    graph = await export_memory_graph_to_rdf(base_iri=base_iri, graph_engine=graph_engine)
    return graph.serialize(format=rdf_format)


async def query_memory_graph_sparql(
    query: str,
    base_iri: str = DEFAULT_BASE_IRI,
    graph_engine: Optional[Any] = None,
) -> List[Any]:
    """Run a SPARQL query against an RDF view of the memory graph.

    Materializes the graph into an in-memory rdflib store and executes the
    query there. Returns rows as rdflib result rows (SELECT) or the appropriate
    rdflib result for other query forms coerced to a list.
    """
    graph = await export_memory_graph_to_rdf(base_iri=base_iri, graph_engine=graph_engine)
    return list(graph.query(query))
