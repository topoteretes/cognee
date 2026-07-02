"""Retriever for analytical/aggregate questions over the knowledge graph.

GRAPH_AGGREGATION answers "how many", "count by type" and "most connected"
questions structurally, without raw Cypher and without counting the vector-retrieved
context (which would let unrelated entity types pollute the result).

The key invariant: an extracted entity is *not* differentiated by its ``type``
property. Every ``Entity`` carries ``type == "Entity"`` (set from the class name in
``DataPoint``), so ``get_filtered_graph_data([{"type": ["Issue"]}])`` returns nothing.
The semantic category ("Issue", "Pull Request", ...) lives one hop away on an
``EntityType`` node reached through the entity's ``is_a`` edge. Counting therefore
resolves the question's noun against the real ``EntityType`` vocabulary and counts the
distinct ``Entity`` nodes linked to it by ``is_a`` -- so a "Pull Request" is excluded
from an "issues" count by graph structure, not by hoping it stayed out of context.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.truth_subspace.align import cosine

logger = get_logger("GraphAggregationRetriever")

# Property value of the ``type`` field on the category nodes (set from the class name).
ENTITY_TYPE_NODE = "EntityType"
# Property value of the ``type`` field on extracted entity nodes.
ENTITY_NODE = "Entity"
# Relationship name connecting an Entity to its EntityType.
ENTITY_TYPE_RELATION = "is_a"


class AggregationOperation(str, Enum):
    """Closed set of supported aggregation operations."""

    COUNT = "count"
    GROUP_BY_COUNT = "group_by_count"
    TOP_BY_DEGREE = "top_by_degree"


class AggregationSpec(BaseModel):
    """Structured, safe representation of an aggregation question.

    The LLM is constrained to this schema so the question never reaches the graph as
    free text: ``operation`` is a closed enum and ``target_type`` is later validated
    against the entity types that actually exist in the graph.
    """

    operation: AggregationOperation = Field(
        description="The kind of aggregation requested.",
    )
    target_type: Optional[str] = Field(
        default=None,
        description="Singular noun naming the entity type asked about (e.g. 'issue').",
    )
    filters: List[str] = Field(
        default_factory=list,
        description="Qualifier words that narrow the set (e.g. 'open').",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="For top_by_degree, how many results were requested.",
    )


def _match_key(text: str) -> str:
    """Normalize a type name for tolerant matching (case, spacing, simple plural)."""
    key = " ".join(text.strip().lower().replace("_", " ").split())
    if len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


class GraphAggregationRetriever(BaseRetriever):
    """Retriever for analytical/aggregate graph queries (count / group / rank).

    Public methods:

    - get_retrieved_objects: parse the question, load the type vocabulary and pull the
      structural data needed for the requested operation.
    - get_context_from_objects: compute the aggregation and render it as a context string.
    - get_completion_from_context: return the rendered one-line answer.
    """

    def __init__(
        self,
        system_prompt_path: str = "graph_aggregation_spec.txt",
        top_k: int = 10,
        resolution_similarity_threshold: float = 0.8,
        resolution_margin: float = 0.05,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            system_prompt_path: Prompt used to parse a question into an AggregationSpec.
            top_k: Default number of results for top_by_degree.
            resolution_similarity_threshold: Minimum cosine similarity for the embedding
                fallback to accept an entity-type match. Kept high because sentence
                embeddings report a large baseline similarity even between unrelated
                words, so a low threshold would resolve any noun to some type.
            resolution_margin: Minimum gap the best embedding match must have over the
                runner-up. A near-tie at the top is the signature of "no real match"
                (every type is roughly equidistant), so the retriever refuses instead.
            session_id: Optional session identifier (unused; kept for interface parity).
        """
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k
        self.resolution_similarity_threshold = resolution_similarity_threshold
        self.resolution_margin = resolution_margin
        self.session_id = session_id

    async def _parse_spec(self, query: str) -> AggregationSpec:
        """Parse a natural-language question into a guarded aggregation spec."""
        system_prompt = read_query_prompt(self.system_prompt_path)
        if system_prompt is None:
            raise ValueError(f"Aggregation spec prompt not found: {self.system_prompt_path}")
        return await LLMGateway.acreate_structured_output(
            text_input=query,
            system_prompt=system_prompt,
            response_model=AggregationSpec,
        )

    async def _load_entity_type_vocabulary(
        self, graph_engine: GraphDBInterface
    ) -> List[Tuple[str, str]]:
        """Return the (id, name) of every EntityType node actually present in the graph."""
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": [ENTITY_TYPE_NODE]}])
        vocabulary: List[Tuple[str, str]] = []
        for node_id, properties in nodes:
            name = (properties or {}).get("name")
            if name:
                vocabulary.append((node_id, name))
        return vocabulary

    def _resolve_types_exact(
        self, target_type: Optional[str], vocabulary: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Resolve a noun to entity types by normalized exact match (case/plural tolerant)."""
        if not target_type:
            return []
        target_key = _match_key(target_type)
        return [(tid, name) for tid, name in vocabulary if _match_key(name) == target_key]

    async def _resolve_types_embedding(
        self, target_type: str, vocabulary: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Resolve a noun to its nearest entity type by embedding similarity.

        Used only when exact matching fails. Failures (no embedding backend, etc.) are
        swallowed and treated as "no match" so the caller can refuse cleanly rather than
        guess.
        """
        if not vocabulary:
            return []
        names = [name for _, name in vocabulary]
        try:
            from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

            embedding_engine = get_embedding_engine()
            vectors = await embedding_engine.embed_text([target_type] + names)
        except Exception as error:  # pragma: no cover - depends on embedding backend
            logger.warning(f"Embedding resolution unavailable for '{target_type}': {error}")
            return []

        # A backend that returns a misaligned (None/empty/short) result must refuse
        # cleanly rather than let an error escape and crash the whole search. Compute the
        # count defensively so even a None return does not blow up the guard itself.
        vector_count = len(vectors) if vectors else 0
        if vector_count != len(names) + 1:
            logger.warning(
                f"Embedding backend returned {vector_count} vectors for "
                f"{len(names) + 1} inputs; skipping embedding resolution."
            )
            return []

        query_vector = vectors[0]
        scored = sorted(
            (
                (cosine(query_vector, vector), vocabulary[index])
                for index, vector in enumerate(vectors[1:])
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_match = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        # Require both a high absolute score and a clear margin over the runner-up.
        # Sentence-embedding models report a high baseline similarity even between
        # unrelated words, so a top match that is only marginally ahead of the rest
        # signals no real match -- refuse rather than guess.
        if (
            best_score >= self.resolution_similarity_threshold
            and best_score - runner_up >= self.resolution_margin
        ):
            return [best_match]
        return []

    async def _resolve_types(
        self, target_type: Optional[str], vocabulary: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Resolve a noun to entity types: exact first, then embedding fallback."""
        resolved = self._resolve_types_exact(target_type, vocabulary)
        if resolved or not target_type:
            return resolved
        return await self._resolve_types_embedding(target_type, vocabulary)

    async def get_retrieved_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse the question and pull the structural data the operation needs."""
        graph_engine = await get_graph_engine()

        if await graph_engine.is_empty():
            logger.warning("Aggregation attempted on an empty knowledge graph.")
            return {"empty": True, "spec": None, "vocabulary": []}

        spec = await self._parse_spec(query or "")
        vocabulary = await self._load_entity_type_vocabulary(graph_engine)

        resolved_types: List[Tuple[str, str]] = []
        nodes: List[Tuple[str, Dict[str, Any]]] = []
        edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

        if spec.operation is AggregationOperation.COUNT:
            resolved_types = await self._resolve_types(spec.target_type, vocabulary)
            # Only materialize the graph once a type resolves; an unresolved noun
            # short-circuits to a clean refusal in _compute_count. We read the full
            # structure (not get_neighborhood with an edge-type filter, which trips a
            # parser assertion on the Kuzu/Ladybug backend) and count the is_a edges
            # whose target is the resolved EntityType.
            if resolved_types:
                nodes, edges = await graph_engine.get_graph_data()
        else:
            # group_by_count and top_by_degree both need the full structure.
            nodes, edges = await graph_engine.get_graph_data()

        return {
            "empty": False,
            "spec": spec,
            "vocabulary": vocabulary,
            "resolved_types": resolved_types,
            "nodes": nodes,
            "edges": edges,
        }

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> str:
        """Render the aggregation result as the natural-language context string.

        The structured result is carried in ``retrieved_objects``; the context returned
        here is a plain sentence so it satisfies the ``str | List[str]`` contract that
        ``SearchResultPayload.context`` enforces (a dict would fail validation).
        """
        return self._render_answer(self._compute_result(retrieved_objects))

    def _compute_result(self, retrieved_objects: Any) -> Dict[str, Any]:
        """Compute the exact aggregation result from the retrieved structure."""
        if not retrieved_objects:
            return {"status": "no_data"}
        if retrieved_objects.get("empty"):
            return {"status": "empty_graph"}

        spec: AggregationSpec = retrieved_objects["spec"]
        vocabulary: List[Tuple[str, str]] = retrieved_objects["vocabulary"]

        if spec.operation is AggregationOperation.COUNT:
            # Counting relies on the type vocabulary to differentiate entities, so refuse
            # cleanly when the graph has no EntityType nodes at all.
            if not vocabulary:
                return {"status": "no_entity_types", "operation": "count"}
            return self._compute_count(spec, retrieved_objects)
        if spec.operation is AggregationOperation.GROUP_BY_COUNT:
            return self._compute_group_by_count(vocabulary, retrieved_objects["edges"])
        # top_by_degree ranks connectivity and needs no type vocabulary. Guard against a
        # non-positive top_k from the LLM: a negative value would slice the ranking
        # wrongly (e.g. ranking[:-1] drops the most connected entity).
        top_k = spec.top_k if spec.top_k and spec.top_k > 0 else self.top_k
        return self._compute_top_by_degree(
            retrieved_objects["nodes"],
            retrieved_objects["edges"],
            top_k,
        )

    def _compute_count(
        self, spec: AggregationSpec, retrieved_objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Count distinct entities of the resolved type, refusing unknown nouns."""
        resolved_types: List[Tuple[str, str]] = retrieved_objects["resolved_types"]
        vocabulary: List[Tuple[str, str]] = retrieved_objects["vocabulary"]
        if not resolved_types:
            return {
                "status": "unknown_type",
                "operation": "count",
                "requested": spec.target_type,
                "available_types": sorted(name for _, name in vocabulary),
            }

        resolved_ids = {type_id for type_id, _ in resolved_types}
        resolved_names = sorted({name for _, name in resolved_types})

        entity_ids = {
            source
            for source, target, relation, _ in retrieved_objects["edges"]
            if relation == ENTITY_TYPE_RELATION and target in resolved_ids
        }

        result: Dict[str, Any] = {
            "status": "ok",
            "operation": "count",
            "target_types": resolved_names,
            "count": len(entity_ids),
        }

        filters = [term.strip().lower() for term in spec.filters if term.strip()]
        if filters:
            # Only index node properties when a qualifier filter is actually present.
            node_properties = {node_id: props for node_id, props in retrieved_objects["nodes"]}
            filtered_ids = {
                entity_id
                for entity_id in entity_ids
                if self._entity_matches_filters(node_properties.get(entity_id, {}), filters)
            }
            result["filters"] = filters
            result["filtered_count"] = len(filtered_ids)
            result["filters_best_effort"] = True
        return result

    @staticmethod
    def _entity_matches_filters(properties: Dict[str, Any], filters: List[str]) -> bool:
        """Best-effort substring match of qualifier terms over an entity's text fields."""
        haystack = " ".join(
            str(properties.get(field, "")) for field in ("name", "description")
        ).lower()
        return all(term in haystack for term in filters)

    def _compute_group_by_count(
        self,
        vocabulary: List[Tuple[str, str]],
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Count distinct entities grouped by their EntityType."""
        id_to_name = {type_id: name for type_id, name in vocabulary}
        groups: Dict[str, set] = {}
        for source, target, relation, _ in edges:
            if relation == ENTITY_TYPE_RELATION and target in id_to_name:
                groups.setdefault(id_to_name[target], set()).add(source)

        counts = {name: len(ids) for name, ids in groups.items()}
        ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        # Count distinct entities for the total: an entity typed under two EntityTypes
        # lands in two groups, so summing group sizes would double-count it.
        distinct_entities = set().union(*groups.values()) if groups else set()
        return {
            "status": "ok",
            "operation": "group_by_count",
            "groups": ordered,
            "total": len(distinct_entities),
        }

    def _compute_top_by_degree(
        self,
        nodes: List[Tuple[str, Dict[str, Any]]],
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        top_k: int,
    ) -> Dict[str, Any]:
        """Rank entities by number of distinct neighbors (most connected first)."""
        entity_properties = {
            node_id: (props or {})
            for node_id, props in nodes
            if (props or {}).get("type") == ENTITY_NODE
        }
        neighbors: Dict[str, set] = {node_id: set() for node_id in entity_properties}
        for source, target, _, _ in edges:
            if source == target:
                continue  # skip self-loops and the SELF fallback edge
            # Count only entity-to-entity links. Edges to an EntityType (is_a) or to a
            # DocumentChunk are ingestion structure, not semantic connectivity, and would
            # otherwise dominate the "most connected" ranking.
            if source in neighbors and target in neighbors:
                neighbors[source].add(target)
                neighbors[target].add(source)

        ranking = sorted(
            neighbors.items(),
            key=lambda item: (-len(item[1]), str(entity_properties[item[0]].get("name") or "")),
        )
        top = [
            {
                "id": node_id,
                "name": entity_properties[node_id].get("name"),
                "degree": len(neighbor_ids),
            }
            for node_id, neighbor_ids in ranking[:top_k]
        ]
        return {"status": "ok", "operation": "top_by_degree", "ranking": top}

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> List[str]:
        """Return the rendered answer (already produced as the context string)."""
        if isinstance(context, str):
            return [context]
        # Defensive: recompute if invoked without a prepared context string.
        return [self._render_answer(self._compute_result(retrieved_objects))]

    def _render_answer(self, result: Dict[str, Any]) -> str:
        """Turn a computed result dict into a short human-readable sentence."""
        status = result.get("status")
        if status == "empty_graph":
            return "The knowledge graph is empty, so there is nothing to aggregate."
        if status == "no_entity_types":
            return (
                "No entity types were found in the graph, so entities cannot be "
                "differentiated for aggregation."
            )
        if status == "unknown_type":
            available = ", ".join(result.get("available_types", [])) or "none"
            return (
                f"Could not find an entity type matching '{result.get('requested')}'. "
                f"Available types: {available}."
            )
        if status != "ok":
            return "This query could not be answered as a graph aggregation."

        operation = result.get("operation")
        if operation == "count":
            return self._render_count(result)
        if operation == "group_by_count":
            return self._render_group_by_count(result)
        return self._render_top_by_degree(result)

    @staticmethod
    def _render_count(result: Dict[str, Any]) -> str:
        types = " / ".join(result.get("target_types", [])) or "matching entities"
        if result.get("filters"):
            qualifiers = " ".join(result["filters"])
            return (
                f"{result.get('filtered_count')} of the {result.get('count')} {types} "
                f"entities match '{qualifiers}' (best-effort text filter)."
            )
        return f"There are {result.get('count')} {types} entities."

    @staticmethod
    def _render_group_by_count(result: Dict[str, Any]) -> str:
        groups = result.get("groups", {})
        if not groups:
            return "No typed entities were found to group."
        breakdown = ", ".join(f"{name}: {count}" for name, count in groups.items())
        return f"Entity counts by type ({result.get('total')} total): {breakdown}."

    @staticmethod
    def _render_top_by_degree(result: Dict[str, Any]) -> str:
        ranking = result.get("ranking", [])
        if not ranking:
            return "No entities were found to rank by connectivity."
        rendered = ", ".join(
            f"{entry.get('name') or entry.get('id')} ({entry.get('degree')} connections)"
            for entry in ranking
        )
        return f"Most connected entities: {rendered}."
