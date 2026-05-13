"""AUTO_RESTRICTED ontology generation.

Per batch: N parallel per-chunk discovery LLM calls draft candidate types and
relations; a serialized 2-call resolve step (entity-type clustering + relation
clustering with narrative-verb rejection, running in parallel) folds those
drafts plus the prior canonical into the new canonical; N parallel extraction
calls then run with the restricted KnowledgeGraph allowlist.
"""

import asyncio
import os
import re
from typing import Any, Literal, Type

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.data_models import Edge, KnowledgeGraph, Node
from cognee.shared.logging_utils import get_logger


logger = get_logger("auto_restricted_ontology")


class GeneratedOntologyRestriction(BaseModel):
    entity_types: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)


class _ResolvedTypeClusters(BaseModel):
    clusters: list[list[str]] = Field(default_factory=list)


class _ResolvedRelationClusters(BaseModel):
    clusters: list[list[str]] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


def _snake_case(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def _unique(values: list[str]) -> list[str]:
    unique_values = {}
    for value in values:
        key = re.sub(r"[^a-z0-9]+", "", value.lower())
        if key and key not in unique_values:
            unique_values[key] = value
    return list(unique_values.values())


_DISCOVERY_PROMPT = """
You are designing a compact canonical ontology allowlist for knowledge graph
extraction from arbitrary text: articles, books, emails, notes, technical
docs, code, tickets, contracts, tables, logs, scientific text, business text,
fiction, and mixed-domain documents.

Return only two allowlists:
1. entity_types: allowed node categories
2. relations: allowed edge predicates

The goal is not to extract every entity or every fact. The goal is to choose a
small, reusable schema that lets a later extractor represent the important
source-grounded facts with consistent names. Small clean allowlists are better
than large noisy ones. When unsure, omit.

General rules:
- Infer only from the supplied text. Do not add world knowledge.
- Prefer names useful across the whole document or dataset, not names tailored
  to one sentence.
- Collapse synonyms and near-duplicates inline. Return one canonical name per
  concept. Do not return both a relation and its inverse.
- Use lowercase snake_case for every value.

entity_types:
- Each value must be a singular common-noun category that many instances could
  belong to, not a specific named entity.
- Prefer broad but informative categories over hyper-specific labels.
- Use roles only when the role is central and reusable in the text. Otherwise
  use the broader type.

relations:
- Each value must be a snake_case predicate for a directed edge between two
  entities. It should read naturally as "source relation target".
- Prefer durable, source-grounded facts over sentence-level action verbs.
- Prefer short predicates, usually 1-3 words.
- Rewrite one-off or past-tense actions into stable predicates when possible;
  if no stable rewrite fits, omit the relation.
- Keep predicates atomic.
- Do not encode source and target types into the relation name.
- Avoid vague catch-all predicates unless the text truly requires generality.
"""


async def _generate_per_chunk_restriction(
    chunk: DocumentChunk, **kwargs: Any
) -> GeneratedOntologyRestriction:
    sample = chunk.text or ""
    restriction = await LLMGateway.acreate_structured_output(
        text_input=f"Generate entity type and relation allowlists for this text:\n\n{sample}",
        system_prompt=_DISCOVERY_PROMPT,
        response_model=GeneratedOntologyRestriction,
        **kwargs,
    )
    return GeneratedOntologyRestriction(
        entity_types=_unique([_snake_case(t) for t in restriction.entity_types]),
        relations=_unique([_snake_case(r) for r in restriction.relations]),
    )


async def _resolve_entity_types(raw_types: list[str], **kwargs: Any) -> list[list[str]]:
    if not raw_types:
        return []
    refined = await LLMGateway.acreate_structured_output(
        text_input=f"Surface forms to cluster:\n{raw_types}",
        system_prompt="""
You curate canonical entity TYPES for a knowledge graph.

Group semantic synonyms into clusters.
Example: ["city", "town", "municipality"] -> one cluster.

For each cluster, put the canonical name FIRST: snake_case, broadest and
most reusable form. Drop instance-like names (specific named entities are
not types; e.g., "Eiffel Tower" is an instance, "monument" is a type).

Return all clusters.
""",
        response_model=_ResolvedTypeClusters,
        **kwargs,
    )
    return refined.clusters


async def _resolve_relations(
    raw_relations: list[str], **kwargs: Any
) -> tuple[list[list[str]], list[str]]:
    if not raw_relations:
        return [], []
    refined = await LLMGateway.acreate_structured_output(
        text_input=f"Surface forms to cluster:\n{raw_relations}",
        system_prompt="""
You curate canonical RELATIONS (predicates) for a knowledge graph.

Rules:
- Group synonyms into clusters.
  Example: ["located_in", "is_in", "situated_in"] -> one cluster.
- First item of each cluster is the canonical: snake_case, present-tense PREDICATE.
- REJECT past-tense narrative verbs and one-off actions
  (e.g., "painted", "flew", "broke", "climbed") -> put these in `rejected`.
- REJECT instance-like names.
- Prefer atomic predicates over composite ones; split the composite into
  atomic then group.
  Example: prefer ["has_width", "has_height"] over ["has_dimensions"].

Return clusters + rejected.
""",
        response_model=_ResolvedRelationClusters,
        **kwargs,
    )
    return refined.clusters, refined.rejected


async def _resolve_restrictions(
    restrictions: list[GeneratedOntologyRestriction], **kwargs: Any
) -> GeneratedOntologyRestriction:
    raw_types = _unique([t.lower() for r in restrictions for t in r.entity_types])
    raw_relations = _unique(
        [_snake_case(r) for restriction in restrictions for r in restriction.relations]
    )

    type_clusters, (relation_clusters, rejected) = await asyncio.gather(
        _resolve_entity_types(raw_types, **kwargs),
        _resolve_relations(raw_relations, **kwargs),
    )

    rejected_set = {_snake_case(r) for r in rejected}
    canonical_relations = [
        _snake_case(c[0])
        for c in relation_clusters
        if c and not any(_snake_case(m) in rejected_set for m in c)
    ]
    canonical_types = [c[0].lower() for c in type_clusters if c]

    if rejected:
        logger.info("AUTO_RESTRICTED rejected relations: %s", rejected)

    return GeneratedOntologyRestriction(
        entity_types=canonical_types,
        relations=canonical_relations,
    )


def _build_restricted_prompt(prompt: str, restriction: GeneratedOntologyRestriction) -> str:
    return f"""{prompt}

AUTO-RESTRICTED ONTOLOGY RULES
- node.type must be one of: {", ".join(restriction.entity_types)}
- edge.relationship_name must be one of: {", ".join(restriction.relations)}
- omit facts that do not fit these allowlists
"""


def _build_restricted_knowledge_graph_model(
    restriction: GeneratedOntologyRestriction,
) -> type[KnowledgeGraph]:
    if not restriction.entity_types or not restriction.relations:
        return KnowledgeGraph

    entity_types = Literal.__getitem__(tuple(restriction.entity_types))
    relations = Literal.__getitem__(tuple(restriction.relations))

    class RestrictedNode(Node):
        type: entity_types

    class RestrictedEdge(Edge):
        relationship_name: relations

    class RestrictedKnowledgeGraph(KnowledgeGraph):
        nodes: list[RestrictedNode] = Field(default_factory=list)
        edges: list[RestrictedEdge] = Field(default_factory=list)

    return RestrictedKnowledgeGraph


def _base_prompt(custom_prompt: str | None = None) -> str:
    if custom_prompt:
        return custom_prompt
    prompt_path = get_llm_config().graph_prompt_path
    base_directory = os.path.dirname(prompt_path) if os.path.isabs(prompt_path) else None
    return render_prompt(os.path.basename(prompt_path), {}, base_directory=base_directory)


class AutoRestrictedOntology:
    """Per-chunk discovery (N parallel calls) + 2-call resolve (clustering +
    narrative-verb rejection, in parallel under a lock) + N parallel
    extraction calls. Cost per batch: 2N + 2 LLM calls.
    """

    def __init__(self) -> None:
        # TODO: productionization
        # 1. in-process canonical won't scale — it's bounded to a single
        #    cognify run on a single worker, lives only in memory, and is
        #    lost when the run ends. Treat `self.canonical` as a stand-in
        #    for an external canonical store (e.g. a DB/cache keyed by
        #    tenant/dataset) so canonicals persist across runs and are
        #    shared across workers.
        # 2. once the canonical grows, we can't pass the whole allowlist
        #    into every discovery/extraction prompt. Need a retrieval step
        #    that surfaces only the canonicals relevant to the current
        #    batch (e.g. embedding/keyword lookup over the canonical store)
        #    before injecting them as context.
        self.canonical = GeneratedOntologyRestriction()
        self._lock = asyncio.Lock()

    async def _update_canonical(
        self, chunks: list[DocumentChunk], **llm_kwargs: Any
    ) -> GeneratedOntologyRestriction:
        per_chunk = await asyncio.gather(
            *[_generate_per_chunk_restriction(chunk, **llm_kwargs) for chunk in chunks]
        )
        async with self._lock:
            self.canonical = await _resolve_restrictions(
                [self.canonical, *per_chunk], **llm_kwargs
            )
            return self.canonical

    async def calculate_chunk_graphs(
        self,
        chunks: list[DocumentChunk],
        graph_model: Type[BaseModel],
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[KnowledgeGraph]:
        if graph_model is not KnowledgeGraph:
            raise ValueError("AUTO_RESTRICTED ontology generation only supports KnowledgeGraph.")
        if not chunks:
            return []

        llm_kwargs = dict(kwargs)
        llm_kwargs.pop("calculate_chunk_graphs", None)

        canonical = await self._update_canonical(chunks, **llm_kwargs)

        logger.info(
            "AUTO_RESTRICTED canonical: %d types, %d relations: %s",
            len(canonical.entity_types),
            len(canonical.relations),
            canonical.model_dump_json(),
        )

        base = _base_prompt(custom_prompt)
        if not canonical.entity_types or not canonical.relations:
            logger.warning(
                "AUTO_RESTRICTED resolved to empty canonical; falling back to unrestricted extraction."
            )
            restricted_model, prompt = KnowledgeGraph, base
        else:
            restricted_model = _build_restricted_knowledge_graph_model(canonical)
            prompt = _build_restricted_prompt(base, canonical)

        return await asyncio.gather(
            *[
                extract_content_graph(
                    chunk.text, restricted_model, custom_prompt=prompt, **llm_kwargs
                )
                for chunk in chunks
            ]
        )
