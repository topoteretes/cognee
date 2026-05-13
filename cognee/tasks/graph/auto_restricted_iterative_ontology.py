"""AUTO_RESTRICTED_ITERATIVE ontology generation.

Per batch: ONE discovery LLM call that sees all the batch's chunks plus the
prior canonical as context, then N parallel extract calls with the restricted
KnowledgeGraph allowlist. No separate resolve step — the discovery prompt is
asked to reuse existing canonicals. The full read-modify-write of
`self.canonical` is serialized so batches observe one another's contributions.
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


logger = get_logger("auto_restricted_iterative_ontology")


class GeneratedOntologyRestriction(BaseModel):
    entity_types: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)


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
- Reuse EXISTING CANONICAL ONTOLOGY values verbatim whenever they cover the
  same concept. Add a new value only when the existing list has no good match.
- Collapse synonyms and near-duplicates inline. Return one canonical name per
  concept. Do not return both a relation and its inverse.
- Use lowercase snake_case for every value.

entity_types:
- Each value must be a singular common-noun category that many instances could
  belong to, not a specific named entity.
- Prefer broad but informative categories over hyper-specific labels.
- Use roles only when the role is central and reusable in the text. Otherwise
  use the broader type.
- Do not use literal values or properties as types unless the text treats them
  as entities that have relationships.

relations:
- Each value must be a snake_case predicate for a directed edge between two
  entities. It should read naturally as "source relation target".
- Prefer durable, source-grounded facts over sentence-level action verbs.
- Prefer short predicates, usually 1-3 words.
- Rewrite one-off or past-tense actions into stable predicates when possible;
  if no stable rewrite fits, omit the relation.
- Keep predicates atomic. Prefer atomic fields over composite ones.
- Do not encode source and target types into the relation name.
- Avoid vague catch-all predicates unless the text truly requires that generality.

Size guidance:
- Include enough coverage for the central recurring concepts, but keep the
  ontology compact and canonical.
"""


async def _generate_iterative_restriction(
    chunks: list[DocumentChunk],
    existing: GeneratedOntologyRestriction,
    **kwargs: Any,
) -> GeneratedOntologyRestriction:
    sample = "\n\n".join(chunk.text for chunk in chunks if getattr(chunk, "text", None))

    existing_context = ""
    if existing.entity_types or existing.relations:
        existing_context = (
            "EXISTING CANONICAL ONTOLOGY (reuse these names when applicable; "
            "only add new ones for concepts the existing list doesn't cover):\n"
            f"- entity_types: {existing.entity_types}\n"
            f"- relations: {existing.relations}\n\n"
        )

    restriction = await LLMGateway.acreate_structured_output(
        text_input=(
            f"{existing_context}"
            f"Generate entity type and relation allowlists for this text:\n\n{sample}"
        ),
        system_prompt=_DISCOVERY_PROMPT,
        response_model=GeneratedOntologyRestriction,
        **kwargs,
    )
    return GeneratedOntologyRestriction(
        entity_types=_unique(
            [
                *[_snake_case(t) for t in existing.entity_types],
                *[_snake_case(t) for t in restriction.entity_types],
            ]
        ),
        relations=_unique(
            [
                *[_snake_case(r) for r in existing.relations],
                *[_snake_case(r) for r in restriction.relations],
            ]
        ),
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


class AutoRestrictedOntologyIterative:
    """One LLM discovery call per batch with the prior canonical injected as
    context; N parallel extraction calls with the resulting allowlist.

    Cost per batch: 1 + N LLM calls. Lock serializes the entire discovery so
    later batches reliably see earlier batches' canonicals.
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

    async def calculate_chunk_graphs(
        self,
        chunks: list[DocumentChunk],
        graph_model: Type[BaseModel],
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[KnowledgeGraph]:
        if graph_model is not KnowledgeGraph:
            raise ValueError(
                "AUTO_RESTRICTED_ITERATIVE ontology generation only supports KnowledgeGraph."
            )
        if not chunks:
            return []

        llm_kwargs = dict(kwargs)
        llm_kwargs.pop("calculate_chunk_graphs", None)

        async with self._lock:
            self.canonical = await _generate_iterative_restriction(
                chunks, existing=self.canonical, **llm_kwargs
            )
            canonical = self.canonical

        logger.info(
            "AUTO_RESTRICTED_ITERATIVE canonical: %d types, %d relations: %s",
            len(canonical.entity_types),
            len(canonical.relations),
            canonical.model_dump_json(),
        )

        base = _base_prompt(custom_prompt)
        if not canonical.entity_types or not canonical.relations:
            logger.warning(
                "AUTO_RESTRICTED_ITERATIVE resolved to empty canonical; "
                "falling back to unrestricted extraction."
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
