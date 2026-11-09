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

No LLM call and no embedding call are made. An unmatched query yields an empty
grounding, and the retriever runs exactly as it did before.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env
from cognee.modules.ontology.models import ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.ontology.term_matching import (
    MIN_TOKEN_LENGTH,
    STOPWORDS,
    TOKEN_PATTERN,
    compact,
    compact_key_index,
    singular,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_query_grounding")

# Vector collections the pinned node ids belong to, by ontology category. Both are
# already part of every graph-completion search, so a pinned id lands in a
# collection the scorer knows how to project.
SEED_COLLECTION_BY_CATEGORY = {
    ONTOLOGY_CLASS_CATEGORY: "EntityType_name",
    ONTOLOGY_INDIVIDUAL_CATEGORY: "Entity_name",
}


@dataclass(frozen=True)
class GroundedConcept:
    """One query term resolved to an ontology node that also lives in the graph."""

    term: str
    canonical_name: str
    category: str
    node_id: str
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

    def to_context_block(self) -> str:
        """Render the grounding as a context section, or "" when nothing matched."""
        if not self.concepts:
            return ""
        lines = ["## Ontology grounding"]
        lines.extend(f"- {concept.describe()}" for concept in self.concepts)
        return "\n".join(lines)


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
    root_key = generate_edge_name(root.name)
    display_name_by_key = {generate_edge_name(node.name): node.name for node in nodes}

    def display(label: str) -> str:
        return display_name_by_key.get(label, label)

    is_a_by_source: dict[str, list[str]] = {}
    relations: list[tuple[str, str]] = []
    for source_label, relationship_name, target_label in edges:
        if relationship_name == "is_a":
            if target_label not in _TRIVIAL_PARENTS:
                is_a_by_source.setdefault(source_label, []).append(target_label)
        elif source_label == root_key:
            relations.append((relationship_name, display(target_label)))

    parents: list[str] = []
    current = root_key
    visited = {root_key}
    while True:
        next_parents = [p for p in is_a_by_source.get(current, []) if p not in visited]
        if not next_parents:
            break
        current = next_parents[0]
        visited.add(current)
        parents.append(display(current))

    data_point_class = EntityType if category == ONTOLOGY_CLASS_CATEGORY else Entity
    return GroundedConcept(
        term=term,
        canonical_name=root.name,
        category=category,
        node_id=str(data_point_class.id_for(root.name)),
        parents=tuple(parents),
        relations=tuple(relations),
    )


def _match_run(run: list[str], indexes: list[tuple[str, dict[str, str]]]) -> tuple[str, str] | None:
    """``(category, resolver key)`` of the first index holding the run, or None.

    The run is compared in compact form as written and with its last word singular,
    so "credit exposures" finds ``creditexposure``.
    """
    forms = (compact("".join(run)), compact("".join(run[:-1]) + singular(run[-1])))
    for category, index in indexes:
        for form in forms:
            key = index.get(form)
            if key is not None:
                return category, key
    return None


def ground_query(
    query: str,
    resolver: BaseOntologyResolver | None,
) -> QueryGrounding:
    """Resolve the terms of ``query`` against ``resolver``.

    Every run of query words, longest first, whose first and last words are not
    stopwords is looked up by exact compact key, in the classes and then the
    individuals. A matched run uses up its words, so "credit exposure" suppresses a
    separate match on "exposure"; each ontology node is reported once.
    """
    if not query or resolver is None:
        return QueryGrounding()

    indexes = [
        (category, compact_key_index(resolver, category))
        for category in (ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY)
    ]
    # A run of more words than the longest key has characters cannot match.
    longest_key = max((len(key) for _, index in indexes for key in index), default=0)
    tokens = TOKEN_PATTERN.findall(query.lower())

    concepts: list[GroundedConcept] = []
    matched_nodes: set[tuple[str, str]] = set()
    used_positions: set[int] = set()

    for size in range(min(len(tokens), longest_key), 0, -1):
        for start in range(len(tokens) - size + 1):
            positions = range(start, start + size)
            run = tokens[start : start + size]
            if used_positions.intersection(positions):
                continue
            if run[0] in STOPWORDS or run[-1] in STOPWORDS:
                continue
            if size == 1 and len(run[0]) < MIN_TOKEN_LENGTH:
                continue
            match = _match_run(run, indexes)
            if match is None:
                continue
            category, key = match
            used_positions.update(positions)
            nodes, edges, root = resolver.get_subgraph(node_name=key, node_type=category)
            if (category, root.name) in matched_nodes:
                continue
            matched_nodes.add((category, root.name))
            concepts.append(_concept_from_subgraph(" ".join(run), category, nodes, root, edges))

    if concepts:
        logger.info(
            "Ontology grounding matched %d concept(s) for query: %s",
            len(concepts),
            ", ".join(concept.canonical_name for concept in concepts),
        )
    return QueryGrounding(concepts=tuple(concepts))


@lru_cache(maxsize=1)
def _load_resolver(file_mtimes: tuple[float | None, ...], **settings: str) -> BaseOntologyResolver:
    """Parse the configured ontology, exactly as cognify builds it.

    ``file_mtimes`` is part of the cache key only: an edited file gets a new key and is
    parsed again, and one entry keeps just the current version in memory. An error is
    raised, never cached, so the first call after the file is fixed loads it.
    """
    return get_ontology_resolver_from_env(**settings)


def _file_mtimes(ontology_file_path: str) -> tuple[float | None, ...]:
    """Modification time of each comma-separated path, ``None`` for a missing file."""
    paths = [path.strip() for path in ontology_file_path.split(",")]
    return tuple(os.path.getmtime(path) if os.path.exists(path) else None for path in paths)


def get_query_grounding_resolver() -> BaseOntologyResolver | None:
    """The resolver recall should ground against, or None when grounding is off.

    Grounding is on when ``ONTOLOGY_FILE_PATH`` names an ontology and
    ``ONTOLOGY_QUERY_GROUNDING`` (default true) has not turned it off. The resolver
    is reloaded when a configured file changes.
    """
    env_config = get_ontology_env_config()
    if not env_config.ontology_query_grounding or not env_config.ontology_file_path:
        return None
    return _load_resolver(_file_mtimes(env_config.ontology_file_path), **env_config.to_dict())
