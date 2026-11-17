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

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import accumulate

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env
from cognee.modules.ontology.models import ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.ontology.term_matching import (
    compact,
    compact_key_index,
    lookup_words,
    singular,
    tokens_of,
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

    def describe(self) -> str:
        kind = "class" if self.category == ONTOLOGY_CLASS_CATEGORY else "individual"
        parts = [f'"{self.term}" refers to {self.canonical_name} ({kind})']
        if self.parents:
            parts.append("is a " + " > ".join(self.parents))
        for relation_name, target_name in self.relations:
            parts.append(f"{relation_name.replace('_', ' ')} {target_name}")
        return "; ".join(parts)


def seed_node_ids_by_collection(concepts: list[GroundedConcept]) -> dict[str, list[str]]:
    """Graph node ids to pin as exact hits, keyed by vector collection name."""
    seeds: dict[str, list[str]] = {}
    for concept in concepts:
        seeds.setdefault(SEED_COLLECTION_BY_CATEGORY[concept.category], []).append(concept.node_id)
    return seeds


def grounding_block(concepts: list[GroundedConcept]) -> str:
    """The ``## Ontology grounding`` context section, or "" when nothing matched."""
    if not concepts:
        return ""
    return "\n".join(
        ["## Ontology grounding", *(f"- {concept.describe()}" for concept in concepts)]
    )


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
    """Build a GroundedConcept from what ``get_subgraph`` returned for one term.

    Only edges to classes and individuals count, the nodes cognify writes
    (``_add_ontology_edges``): a blank node, such as an ``owl:Restriction``
    superclass, or an undeclared URI is not a parent or a relation target.
    """
    root_key = generate_edge_name(root.name)
    display_name_by_key = {generate_edge_name(node.name): node.name for node in nodes}
    written_keys = {
        generate_edge_name(node.name)
        for node in nodes
        if node.category in (ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY)
    }

    def display(label: str) -> str:
        return display_name_by_key.get(label, label)

    is_a_by_source: dict[str, list[str]] = {}
    relations: list[tuple[str, str]] = []
    for source_label, relationship_name, target_label in edges:
        if target_label not in written_keys:
            continue
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


@dataclass
class _ResolverIndex:
    """What grounding reads from one resolver, kept between queries."""

    indexes: list[tuple[str, dict[str, str]]]
    longest_key: int
    # ``(category, resolver key)`` -> its concept, filled the first time a query names it.
    concepts: dict[tuple[str, str], GroundedConcept] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _resolver_index(resolver: BaseOntologyResolver) -> _ResolverIndex:
    """The compact key indexes of ``resolver``, built on its first query.

    One entry, like ``_load_resolver``: recall grounds against one resolver at a time.
    """
    indexes = [
        (category, compact_key_index(resolver, category))
        for category in (ONTOLOGY_CLASS_CATEGORY, ONTOLOGY_INDIVIDUAL_CATEGORY)
    ]
    longest_key = max((len(key) for _, index in indexes for key in index), default=0)
    return _ResolverIndex(indexes=indexes, longest_key=longest_key)


def _match_run(run: list[str], indexes: list[tuple[str, dict[str, str]]]) -> tuple[str, str] | None:
    """``(category, resolver key)`` of the first index that holds ``run``, or None."""
    for category, index in indexes:
        key = lookup_words(run, index)
        if key is not None:
            return category, key
    return None


def _concept_for_key(
    resolver: BaseOntologyResolver,
    resolver_index: _ResolverIndex,
    category: str,
    key: str,
    term: str,
) -> GroundedConcept:
    """The concept ``key`` names, reported for ``term``.

    The key's ontology subgraph is walked the first time a query names it; later
    queries against the same resolver reuse the result.
    """
    concept = resolver_index.concepts.get((category, key))
    if concept is None:
        nodes, edges, root = resolver.get_subgraph(node_name=key, node_type=category)
        concept = _concept_from_subgraph(term, category, nodes, root, edges)
        resolver_index.concepts[(category, key)] = concept
    return replace(concept, term=term)


def ground_query(
    query: str | None,
    resolver: BaseOntologyResolver | None,
) -> list[GroundedConcept]:
    """The ontology nodes the words of ``query`` name in ``resolver``.

    Every run of query words, longest first, is looked up by exact compact key, in the
    classes and then the individuals; a lone stopword ("will", "may") is not. A
    matched run uses up its words, so "credit exposure" suppresses a separate match on
    "exposure"; each ontology node is reported once. The resolver is treated as
    read-only: its indexes and concepts are kept for the next query.
    """
    if not query or resolver is None:
        return []

    resolver_index = _resolver_index(resolver)
    tokens = tokens_of(query)
    # Compact lengths: ``ends[i]`` of the first ``i`` tokens together, and of each token
    # in its singular form, which ``lookup_words`` also tries for a run's last word.
    ends = [0, *accumulate(len(compact(token)) for token in tokens)]
    singular_lengths = [len(compact(singular(token))) for token in tokens]

    concepts: list[GroundedConcept] = []
    # A resolver key names exactly one ontology node, so matched keys are matched nodes.
    matched_keys: set[tuple[str, str]] = set()
    used_positions: set[int] = set()

    # A run of more words than the longest key has characters cannot match.
    for size in range(min(len(tokens), resolver_index.longest_key), 0, -1):
        for start in range(len(tokens) - size + 1):
            end = start + size
            # Nor can a run whose shortest form is longer than every key.
            if ends[end - 1] - ends[start] + singular_lengths[end - 1] > resolver_index.longest_key:
                continue
            positions = range(start, end)
            run = tokens[start:end]
            if used_positions.intersection(positions):
                continue
            match = _match_run(run, resolver_index.indexes)
            if match is None:
                continue
            used_positions.update(positions)
            if match in matched_keys:
                continue
            matched_keys.add(match)
            category, key = match
            concepts.append(
                _concept_for_key(resolver, resolver_index, category, key, " ".join(run))
            )

    if concepts:
        logger.info(
            "Ontology grounding matched %d concept(s) for query: %s",
            len(concepts),
            ", ".join(concept.canonical_name for concept in concepts),
        )
    return concepts


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


def _ground(query: str, ontology_file_path: str, settings: dict[str, str]) -> list[GroundedConcept]:
    """``ground_query`` against the files as they are now. Runs on ``_grounding_thread``."""
    return ground_query(query, _load_resolver(_file_mtimes(ontology_file_path), **settings))


# Every grounding step that can be slow runs on this one thread: the file check, a parse
# (with ``ONTOLOGY_REASONING=owl_rl``, the closure too), the scan and the subgraph walks.
# Searches that wait for a parse queue here instead of holding threads of the shared pool,
# and the resolver's caches are only ever used from this thread. A concurrent.futures
# executor, unlike an asyncio lock, works on any event loop.
_grounding_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ontology-grounding")


async def ground(query: str | None) -> list[GroundedConcept]:
    """The ontology concepts ``query`` names, or [] when grounding is off.

    Grounding is on when ``ONTOLOGY_FILE_PATH`` names an ontology and
    ``ONTOLOGY_QUERY_GROUNDING`` (default true) has not turned it off. The resolver
    is parsed again when a configured file changes.
    """
    env_config = get_ontology_env_config()
    if not query or not env_config.ontology_query_grounding or not env_config.ontology_file_path:
        return []
    return await asyncio.get_running_loop().run_in_executor(
        _grounding_thread, _ground, query, env_config.ontology_file_path, env_config.to_dict()
    )


async def with_grounding_block(query: str | None, context: str) -> str:
    """``context`` with the ``## Ontology grounding`` block for ``query`` in front.

    Only in front of real context: on its own the block names concepts without any
    facts about them, and an empty context must stay empty so "nothing found" remains
    detectable.
    """
    if not context:
        return context
    block = grounding_block(await ground(query))
    return f"{block}\n\n{context}" if block else context
