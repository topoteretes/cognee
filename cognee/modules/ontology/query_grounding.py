"""Ground a natural-language query in the configured ontology at recall time.

Cognify applies the ontology at write time: extracted names are canonicalized and the
matched ontology subgraph (classes, individuals, ``is_a`` chains) is written into the
graph under deterministic ids — ``EntityType.id_for(name)`` for classes and
``Entity.id_for(name)`` for individuals (see
``construct_data_points_and_edges_with_ontology``). This module is the read-time
counterpart. It resolves the words of a query against the same resolver and returns

* the graph node ids of the matched ontology nodes, so a retriever can pin them as
  traversal seeds next to its vector hits, and
* a short text block telling the LLM what each matched term means in the business
  model (canonical name, ``is_a`` chain, object-property relations).

No LLM call and no embedding call are made. Everything fails open: a resolver error, a
missing file or an unmatched query yields an empty grounding and the retriever runs
exactly as it did before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology import term_matching
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_query_grounding")

ONTOLOGY_CLASS_CATEGORY = "classes"
ONTOLOGY_INDIVIDUAL_CATEGORY = "individuals"

# Vector collections the pinned node ids belong to, by ontology category. Both are
# already part of every graph-completion search, so a pinned id lands in a
# collection the scorer knows how to project.
SEED_COLLECTION_BY_CATEGORY = {
    ONTOLOGY_CLASS_CATEGORY: "EntityType_name",
    ONTOLOGY_INDIVIDUAL_CATEGORY: "Entity_name",
}

DEFAULT_MAX_CONCEPTS = 5
DEFAULT_MAX_NGRAM = 3
_MAX_CANDIDATE_TERMS = 60
_MAX_PARENTS = 4
_MAX_RELATIONS = 5
_MIN_TOKEN_LENGTH = term_matching._MIN_TOKEN_LENGTH
_STOPWORDS = term_matching.STOPWORDS
_TOKEN_PATTERN = term_matching.TOKEN_PATTERN


@dataclass(frozen=True)
class GroundedConcept:
    """One query term resolved to an ontology node that also lives in the graph."""

    term: str
    canonical_name: str
    category: str
    node_id: str
    uri: str | None = None
    parents: tuple[str, ...] = ()
    relations: tuple[tuple[str, str], ...] = ()

    @property
    def seed_collection(self) -> str:
        return SEED_COLLECTION_BY_CATEGORY[self.category]

    def describe(self) -> str:
        kind = "class" if self.category == ONTOLOGY_CLASS_CATEGORY else "individual"
        parts = [f'"{self.term}" refers to {self.canonical_name} ({kind})']
        if self.parents:
            parts.append("is a " + " > ".join(self.parents))
        for relation_name, target_name in self.relations:
            parts.append(f"{relation_name.replace('_', ' ')} {target_name}")
        return "; ".join(parts)


@dataclass(frozen=True)
class QueryGrounding:
    """Ontology matches for one query, plus the seeds and context they translate to."""

    query: str
    concepts: tuple[GroundedConcept, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.concepts)

    def seed_node_ids_by_collection(self) -> dict[str, list[str]]:
        """Graph node ids to pin as exact hits, keyed by vector collection name."""
        seeds: dict[str, list[str]] = {}
        for concept in self.concepts:
            ids = seeds.setdefault(concept.seed_collection, [])
            if concept.node_id not in ids:
                ids.append(concept.node_id)
        return seeds

    def seed_node_ids(self) -> list[str]:
        return [
            node_id
            for node_ids in self.seed_node_ids_by_collection().values()
            for node_id in node_ids
        ]

    def to_context_block(self) -> str:
        """Render the grounding as a context section, or "" when nothing matched."""
        if not self.concepts:
            return ""
        lines = ["## Ontology grounding"]
        lines.extend(f"- {concept.describe()}" for concept in self.concepts)
        return "\n".join(lines)


def extract_candidate_terms(query: str, max_ngram: int = DEFAULT_MAX_NGRAM) -> list[str]:
    """Return lookup terms for a query: longest n-grams first, then shorter ones.

    Terms are lowercased, joined by ``_`` (the resolver's key form), stripped of
    stopwords at n-gram boundaries, and capped so a long query cannot turn into
    hundreds of fuzzy lookups.
    """
    tokens = _TOKEN_PATTERN.findall(query.lower())
    if not tokens:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for size in range(min(max_ngram, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            window = tokens[start : start + size]
            if window[0] in _STOPWORDS or window[-1] in _STOPWORDS:
                continue
            if size == 1 and len(window[0]) < _MIN_TOKEN_LENGTH:
                continue
            term = "_".join(window)
            if term in seen:
                continue
            seen.add(term)
            candidates.append(term)
            if len(candidates) >= _MAX_CANDIDATE_TERMS:
                return candidates
    return candidates


_normalize_key = term_matching.normalize_key
_multiword_match_is_sound = term_matching.multiword_match_is_sound


# ``is_a`` targets that say nothing about the business: the OWL meta-classes the
# resolver reports for individuals, and the universal root.
_TRIVIAL_PARENTS = frozenset({"class", "thing", "namedindividual", "resource"})


def _concept_from_subgraph(
    term: str,
    category: str,
    nodes: list,
    root,
    edges: list[tuple[str, str, str]],
) -> GroundedConcept:
    """Build a GroundedConcept from what ``get_subgraph`` returned for one term."""
    root_key = _normalize_key(root.name)
    display_name_by_key = {_normalize_key(node.name): node.name for node in nodes}

    def display(label: str) -> str:
        return display_name_by_key.get(label, label)

    is_a_by_source: dict[str, list[str]] = {}
    relations: list[tuple[str, str]] = []
    for source_label, relationship_name, target_label in edges:
        if relationship_name == "is_a":
            if target_label not in _TRIVIAL_PARENTS:
                is_a_by_source.setdefault(source_label, []).append(target_label)
        elif source_label == root_key and len(relations) < _MAX_RELATIONS:
            relations.append((relationship_name, display(target_label)))

    parents: list[str] = []
    current = root_key
    visited = {root_key}
    while len(parents) < _MAX_PARENTS:
        next_parents = [p for p in is_a_by_source.get(current, []) if p not in visited]
        if not next_parents:
            break
        current = next_parents[0]
        visited.add(current)
        parents.append(display(current))

    data_point_class = EntityType if category == ONTOLOGY_CLASS_CATEGORY else Entity
    return GroundedConcept(
        term=term.replace("_", " "),
        canonical_name=root.name,
        category=category,
        node_id=str(data_point_class.id_for(root.name)),
        uri=str(root.uri) if root.uri is not None else None,
        parents=tuple(parents),
        relations=tuple(relations),
    )


def ground_query(
    query: str,
    resolver: BaseOntologyResolver | None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
) -> QueryGrounding:
    """Resolve the terms of ``query`` against ``resolver``.

    Longer n-grams win over the unigrams they contain ("credit exposure" suppresses a
    separate match on "exposure"); each ontology node is reported once even when
    several terms match it. Resolver failures are logged and treated as no match.
    """
    if not query or resolver is None:
        return QueryGrounding(query=query or "")

    concepts: list[GroundedConcept] = []
    matched_nodes: set[tuple[str, str]] = set()
    consumed_tokens: set[str] = set()

    for term in extract_candidate_terms(query):
        if len(concepts) >= max_concepts:
            break
        term_tokens = set(term.split("_"))
        if term_tokens & consumed_tokens:
            continue

        for category in (ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY):
            try:
                nodes, edges, root = resolver.get_subgraph(node_name=term, node_type=category)
            except Exception as error:  # fail open: the retriever must not break
                logger.debug(
                    "Ontology lookup failed for %r (%s): %s", term, category, error, exc_info=True
                )
                continue
            if root is None or not _multiword_match_is_sound(term, root.name):
                continue
            node_key = (category, root.name)
            if node_key in matched_nodes:
                continue
            matched_nodes.add(node_key)
            consumed_tokens |= term_tokens
            concepts.append(_concept_from_subgraph(term, category, nodes, root, edges))
            break

    if concepts:
        logger.info(
            "Ontology grounding matched %d concept(s) for query: %s",
            len(concepts),
            ", ".join(concept.canonical_name for concept in concepts),
        )
    return QueryGrounding(query=query, concepts=tuple(concepts))


@lru_cache(maxsize=8)
def _load_resolver(ontology_file_path: str) -> BaseOntologyResolver | None:
    """Parse the configured ontology once per process (RDF parsing is not free)."""
    from cognee.modules.ontology.get_default_ontology_resolver import (
        get_ontology_resolver_from_env,
    )

    env_config = get_ontology_env_config()
    try:
        return get_ontology_resolver_from_env(
            ontology_resolver=env_config.ontology_resolver,
            matching_strategy=env_config.matching_strategy,
            ontology_file_path=ontology_file_path,
        )
    except Exception as error:
        logger.warning(
            "Ontology query grounding disabled: could not load ontology from %r: %s",
            ontology_file_path,
            error,
            exc_info=True,
        )
        return None


def get_query_grounding_resolver() -> BaseOntologyResolver | None:
    """The resolver recall should ground against, or None when grounding is off.

    Grounding is on when ``ONTOLOGY_FILE_PATH`` names an ontology and
    ``ONTOLOGY_QUERY_GROUNDING`` (default true) has not turned it off. The resolver
    is cached per file path for the life of the process.
    """
    env_config = get_ontology_env_config()
    if not env_config.ontology_query_grounding or not env_config.ontology_file_path:
        return None
    return _load_resolver(env_config.ontology_file_path)


def ground_query_with_configured_ontology(
    query: str | None,
    resolver: BaseOntologyResolver | None = None,
    enabled: bool | None = None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
) -> QueryGrounding:
    """Retriever entry point: resolve the flag and the resolver, then ground.

    ``enabled=None`` follows ``ONTOLOGY_QUERY_GROUNDING``; an explicit resolver is
    used as-is (tests, custom resolvers), otherwise the configured one is loaded.
    """
    if not query:
        return QueryGrounding(query="")
    if enabled is False:
        return QueryGrounding(query=query)
    if resolver is None:
        if enabled is None:
            resolver = get_query_grounding_resolver()
        else:
            env_config = get_ontology_env_config()
            resolver = (
                _load_resolver(env_config.ontology_file_path)
                if env_config.ontology_file_path
                else None
            )
    if resolver is None:
        return QueryGrounding(query=query)
    return _ground_cached(resolver, query, max_concepts)


@lru_cache(maxsize=256)
def _ground_cached(resolver: BaseOntologyResolver, query: str, max_concepts: int) -> QueryGrounding:
    # Concurrent-mode turns ground the same query from two lanes and again when the
    # context block is rendered; the lookup is deterministic, so memoize per resolver.
    return ground_query(query, resolver, max_concepts=max_concepts)
