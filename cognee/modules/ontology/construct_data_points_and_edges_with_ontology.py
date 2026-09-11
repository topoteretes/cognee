from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import construct_data_points_and_edges
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.exceptions import EmptyOntologyInStrictModeError
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.ontology_env_config import (
    get_ontology_env_config,
    normalize_ontology_mode,
)
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_ONTOLOGY_CLASS_CATEGORY = "classes"
_ONTOLOGY_INDIVIDUAL_CATEGORY = "individuals"


@dataclass(frozen=True)
class OntologyMatch:
    node_category: str
    canonical_name: str
    canonical_uri: Optional[str]
    ontology_nodes: list[AttachedOntologyNode]
    ontology_edges: list[tuple[str, str, str]]
    first_source_chunk: DocumentChunk


OntologyMatchLookup = dict[tuple[str, str], Optional[OntologyMatch]]


@dataclass
class _StrictModeDropCounts:
    total_nodes: int = 0
    dropped_nodes: int = 0
    total_edges: int = 0
    dropped_edges: int = 0

    def add(self, other: "_StrictModeDropCounts") -> None:
        self.total_nodes += other.total_nodes
        self.dropped_nodes += other.dropped_nodes
        self.total_edges += other.total_edges
        self.dropped_edges += other.dropped_edges


def ensure_ontology_usable_in_strict_mode(ontology_resolver: BaseOntologyResolver) -> None:
    """Refuse strict mode when the resolver holds no ontology entries at all.

    An empty lookup — typically a mistyped ONTOLOGY_FILE_PATH, which the resolver
    only warns about — is indistinguishable from "the ontology rejected everything"
    once matching starts, so strict mode would silently drop the entire graph.
    Resolvers without a ``lookup`` dict (custom implementations) are not checked.
    """
    lookup = getattr(ontology_resolver, "lookup", None)
    if isinstance(lookup, dict) and not any(lookup.values()):
        raise EmptyOntologyInStrictModeError()


def _find_ontology_match(
    ontology_resolver: BaseOntologyResolver,
    normalized_extracted_name: str,
    node_category: str,
    first_source_chunk: DocumentChunk,
) -> Optional[OntologyMatch]:
    """Find the ontology node and subgraph matching one extracted name."""
    ontology_nodes, ontology_edges, matched_ontology_node = ontology_resolver.get_subgraph(
        node_name=normalized_extracted_name,
        node_type=node_category,
    )
    if not matched_ontology_node:
        return None

    return OntologyMatch(
        node_category=node_category,
        canonical_name=matched_ontology_node.name,
        canonical_uri=(
            str(matched_ontology_node.uri) if matched_ontology_node.uri is not None else None
        ),
        ontology_nodes=ontology_nodes,
        ontology_edges=ontology_edges,
        first_source_chunk=first_source_chunk,
    )


def _find_ontology_matches_for_extracted_graphs(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
) -> OntologyMatchLookup:
    """Find one ontology match for each distinct extracted name and type."""
    ontology_match_lookup: OntologyMatchLookup = {}
    for data_chunk, extracted_graph in zip(data_chunks, extracted_graphs):
        if not extracted_graph:
            continue

        for node in extracted_graph.nodes:
            for node_category, extracted_name in (
                (_ONTOLOGY_CLASS_CATEGORY, node.type),
                (_ONTOLOGY_INDIVIDUAL_CATEGORY, node.name),
            ):
                normalized_extracted_name = generate_node_name(extracted_name)
                lookup_key = (node_category, normalized_extracted_name)
                if lookup_key in ontology_match_lookup:
                    continue

                ontology_match_lookup[lookup_key] = _find_ontology_match(
                    ontology_resolver,
                    normalized_extracted_name,
                    node_category,
                    data_chunk,
                )

    return ontology_match_lookup


def _get_ontology_match(
    ontology_match_lookup: OntologyMatchLookup,
    node_category: str,
    extracted_name: str,
) -> Optional[OntologyMatch]:
    """Return the ontology match already looked up for an extracted name."""
    return ontology_match_lookup.get((node_category, generate_node_name(extracted_name)))


def _canonicalize_extracted_graph(
    extracted_graph: KnowledgeGraph,
    ontology_match_lookup: OntologyMatchLookup,
    strict: bool = False,
) -> _StrictModeDropCounts:
    """Canonicalize one graph and collapse nodes matched to the same ontology entity.

    Strict mode is entity-grounding only: a node is retained when the ontology matched
    EITHER its type against a class OR its name against an individual — an unknown
    entity with a recognized type survives. Nodes with neither match are dropped along
    with their edges. There is no domain/range/cardinality/disjointness reasoning, and
    relationship names are not checked against the ontology.
    """
    extracted_node_ids: set[str] = set()
    for node in extracted_graph.nodes:
        if node.id in extracted_node_ids:
            raise ValueError(f"Duplicate node id in extracted graph: {node.id}")
        extracted_node_ids.add(node.id)

    retained_nodes: list[Node] = []
    dropped_node_ids: set[str] = set()
    surviving_node_id_by_entity_id: dict[UUID, str] = {}
    surviving_node_id_by_collapsed_node_id: dict[str, str] = {}

    for node in extracted_graph.nodes:
        entity_type_match = _get_ontology_match(
            ontology_match_lookup,
            _ONTOLOGY_CLASS_CATEGORY,
            node.type,
        )
        if entity_type_match is not None:
            node.type = entity_type_match.canonical_name

        entity_match = _get_ontology_match(
            ontology_match_lookup,
            _ONTOLOGY_INDIVIDUAL_CATEGORY,
            node.name,
        )
        if entity_match is None:
            if strict and entity_type_match is None:
                dropped_node_ids.add(node.id)
            else:
                retained_nodes.append(node)
            continue

        node.name = entity_match.canonical_name
        canonical_entity_id = Entity.id_for(entity_match.canonical_name)
        surviving_node_id = surviving_node_id_by_entity_id.get(canonical_entity_id)
        if surviving_node_id is None:
            surviving_node_id_by_entity_id[canonical_entity_id] = node.id
            retained_nodes.append(node)
            continue

        surviving_node_id_by_collapsed_node_id[node.id] = surviving_node_id

    drop_counts = _StrictModeDropCounts(
        total_nodes=len(extracted_node_ids),
        dropped_nodes=len(dropped_node_ids),
        total_edges=len(extracted_graph.edges),
    )

    extracted_graph.nodes = retained_nodes
    if not surviving_node_id_by_collapsed_node_id and not dropped_node_ids:
        return drop_counts

    retained_edges = []
    for edge in extracted_graph.edges:
        edge.source_node_id = surviving_node_id_by_collapsed_node_id.get(
            edge.source_node_id,
            edge.source_node_id,
        )
        edge.target_node_id = surviving_node_id_by_collapsed_node_id.get(
            edge.target_node_id,
            edge.target_node_id,
        )
        if edge.source_node_id in dropped_node_ids or edge.target_node_id in dropped_node_ids:
            continue
        retained_edges.append(edge)

    drop_counts.dropped_edges = len(extracted_graph.edges) - len(retained_edges)
    extracted_graph.edges = retained_edges
    return drop_counts


def canonicalize_extracted_graphs(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
    strict: bool = False,
) -> tuple[list[KnowledgeGraph], OntologyMatchLookup]:
    """Canonicalize ontology matches in place before ordinary graph construction.

    Strict mode prunes only the extracted graph: chunks were already stored and
    embedded earlier in the pipeline, so chunk-based search types (CHUNKS,
    CHUNKS_LEXICAL, RAG_COMPLETION) still retrieve text mentioning dropped entities.
    Only graph-based search types see the pruned view.
    """
    if strict:
        ensure_ontology_usable_in_strict_mode(ontology_resolver)

    ontology_match_lookup = _find_ontology_matches_for_extracted_graphs(
        data_chunks,
        extracted_graphs,
        ontology_resolver,
    )

    run_drop_counts = _StrictModeDropCounts()
    for extracted_graph in extracted_graphs:
        if extracted_graph:
            run_drop_counts.add(
                _canonicalize_extracted_graph(extracted_graph, ontology_match_lookup, strict=strict)
            )

    if strict and (run_drop_counts.dropped_nodes or run_drop_counts.dropped_edges):
        retained_nodes = run_drop_counts.total_nodes - run_drop_counts.dropped_nodes
        logger.warning(
            "Strict ontology mode dropped %s of %s extracted node(s) (%.0f%% retained) "
            "and %s of %s edge(s) across %s graph(s). A high drop ratio usually means "
            "the ontology does not cover the corpus's vocabulary.",
            run_drop_counts.dropped_nodes,
            run_drop_counts.total_nodes,
            100.0 * retained_nodes / run_drop_counts.total_nodes
            if run_drop_counts.total_nodes
            else 0.0,
            run_drop_counts.dropped_edges,
            run_drop_counts.total_edges,
            len(extracted_graphs),
        )

    return extracted_graphs, ontology_match_lookup


def _get_data_point_class_for_ontology_category(
    node_category: str,
) -> type[Entity] | type[EntityType] | None:
    if node_category == _ONTOLOGY_CLASS_CATEGORY:
        return EntityType
    if node_category == _ONTOLOGY_INDIVIDUAL_CATEGORY:
        return Entity
    return None


def _mark_existing_ontology_data_point(
    data_point_class: type[Entity] | type[EntityType],
    ontology_name: str,
    ontology_uri: Optional[str],
    data_points_by_id: dict[str, Entity | EntityType],
) -> bool:
    data_point_id = data_point_class.id_for(ontology_name)
    data_point = data_points_by_id.get(str(data_point_id))
    if data_point is None:
        return False

    data_point.ontology_valid = True
    if ontology_uri is not None:
        data_point.ontology_uri = ontology_uri
    return True


def _add_ontology_data_points(
    ontology_nodes: list[AttachedOntologyNode],
    source_chunk: DocumentChunk,
    data_points_by_id: dict[str, Entity | EntityType],
) -> None:
    for ontology_node in ontology_nodes:
        data_point_class = _get_data_point_class_for_ontology_category(ontology_node.category)
        if data_point_class is None:
            continue

        ontology_uri = str(ontology_node.uri) if ontology_node.uri is not None else None
        if _mark_existing_ontology_data_point(
            data_point_class,
            ontology_node.name,
            ontology_uri,
            data_points_by_id,
        ):
            continue

        data_point_id = data_point_class.id_for(ontology_node.name)
        data_point_key = str(data_point_id)
        normalized_name = generate_node_name(ontology_node.name)
        if data_point_class is EntityType:
            data_points_by_id[data_point_key] = EntityType(
                id=data_point_id,
                name=normalized_name,
                description=normalized_name,
                ontology_valid=True,
                ontology_uri=ontology_uri,
                importance_weight=source_chunk.importance_weight,
            )
        else:
            data_points_by_id[data_point_key] = Entity(
                id=data_point_id,
                name=normalized_name,
                description=normalized_name,
                ontology_valid=True,
                ontology_uri=ontology_uri,
                belongs_to_set=source_chunk.belongs_to_set,
                importance_weight=source_chunk.importance_weight,
            )


def _add_ontology_edges(
    ontology_nodes: list[AttachedOntologyNode],
    ontology_edges: list[tuple[str, str, str]],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    ontology_nodes_by_normalized_name = {
        generate_edge_name(node.name): node for node in ontology_nodes
    }
    for source_name, ontology_relationship_name, target_name in ontology_edges:
        source_node = ontology_nodes_by_normalized_name.get(generate_edge_name(source_name))
        target_node = ontology_nodes_by_normalized_name.get(generate_edge_name(target_name))
        if source_node is None or target_node is None:
            continue

        source_data_point_class = _get_data_point_class_for_ontology_category(source_node.category)
        target_data_point_class = _get_data_point_class_for_ontology_category(target_node.category)
        if source_data_point_class is None or target_data_point_class is None:
            continue

        relationship_name = generate_edge_name(ontology_relationship_name)
        edge_identity = EdgeIdentity(
            source_id=str(source_data_point_class.id_for(source_node.name)),
            target_id=str(target_data_point_class.id_for(target_node.name)),
            relationship_name=relationship_name,
        )
        edges_by_identity.setdefault(
            edge_identity,
            Edge(relationship_type=relationship_name),
        )


def add_ontology_data_points_and_edges(
    ontology_match_lookup: OntologyMatchLookup,
    data_points_by_id: dict[str, Entity | EntityType],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    """Add the nodes and edges from each uniquely matched ontology subgraph."""
    added_ontology_root_keys: set[tuple[str, str]] = set()
    for ontology_match in ontology_match_lookup.values():
        if ontology_match is None:
            continue

        ontology_root_key = (
            ontology_match.node_category,
            ontology_match.canonical_uri or ontology_match.canonical_name,
        )
        if ontology_root_key in added_ontology_root_keys:
            continue
        added_ontology_root_keys.add(ontology_root_key)

        canonical_data_point_class = _get_data_point_class_for_ontology_category(
            ontology_match.node_category
        )
        if canonical_data_point_class is not None:
            _mark_existing_ontology_data_point(
                canonical_data_point_class,
                ontology_match.canonical_name,
                ontology_match.canonical_uri,
                data_points_by_id,
            )

        _add_ontology_data_points(
            ontology_match.ontology_nodes,
            ontology_match.first_source_chunk,
            data_points_by_id,
        )
        _add_ontology_edges(
            ontology_match.ontology_nodes,
            ontology_match.ontology_edges,
            edges_by_identity,
        )


def construct_data_points_and_edges_with_ontology(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver,
    ontology_mode: Optional[str] = None,
) -> tuple[dict[str, Entity | EntityType], dict[EdgeIdentity, Edge]]:
    """Canonicalize, construct, and enrich extracted graphs with ontology data.

    ``ontology_mode`` overrides the ONTOLOGY_MODE environment value for this call;
    when None, the environment value applies.
    """
    if ontology_mode is None:
        ontology_mode = get_ontology_env_config().ontology_mode
    strict = normalize_ontology_mode(ontology_mode) == "strict"
    canonicalized_graphs, ontology_match_lookup = canonicalize_extracted_graphs(
        data_chunks,
        extracted_graphs,
        ontology_resolver,
        strict=strict,
    )
    data_points_by_id, edges_by_identity = construct_data_points_and_edges(
        data_chunks,
        canonicalized_graphs,
    )
    add_ontology_data_points_and_edges(
        ontology_match_lookup,
        data_points_by_id,
        edges_by_identity,
    )
    return data_points_by_id, edges_by_identity
