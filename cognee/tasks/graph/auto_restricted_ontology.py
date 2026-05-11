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
ONTOLOGY_CHUNKS_PER_GROUP = 5


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


async def generate_restriction_from_chunks(
    chunks: list[DocumentChunk], **kwargs: Any
) -> GeneratedOntologyRestriction:
    sample = "\n\n".join(chunk.text for chunk in chunks if getattr(chunk, "text", None))
    restriction = await LLMGateway.acreate_structured_output(
        text_input=f"Generate entity type and relation allowlists for this text batch:\n\n{sample}",
        system_prompt="""
Generate a small automatic ontology for KnowledgeGraph extraction.

Return:
- entity_types: allowed entity types for node.type, not entity names
- relations: allowed edge types for edge.relationship_name in lowercase snake_case

Infer only from the text. Do not include domain/range triples.
""",
        response_model=GeneratedOntologyRestriction,
        **kwargs,
    )
    restriction = GeneratedOntologyRestriction(
        entity_types=_unique(
            [" ".join(entity_type.strip().split()) for entity_type in restriction.entity_types]
        ),
        relations=_unique([_snake_case(relation) for relation in restriction.relations]),
    )
    logger.info(f"AUTO_RESTRICTED generated ontology restriction: {restriction.model_dump_json()}")
    return restriction


def build_restricted_prompt(prompt: str, restriction: GeneratedOntologyRestriction) -> str:
    return f"""{prompt}

AUTO-RESTRICTED ONTOLOGY RULES
- node.type must be one of: {", ".join(restriction.entity_types)}
- edge.relationship_name must be one of: {", ".join(restriction.relations)}
- omit facts that do not fit these allowlists
"""


def build_restricted_knowledge_graph_model(
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


async def auto_restricted_calculate_chunk_graphs(
    chunks: list[DocumentChunk],
    graph_model: Type[BaseModel],
    custom_prompt: str | None = None,
    **kwargs: Any,
) -> list[KnowledgeGraph]:
    if graph_model is not KnowledgeGraph:
        raise ValueError("AUTO_RESTRICTED ontology generation only supports KnowledgeGraph.")

    llm_kwargs = dict(kwargs)
    llm_kwargs.pop("calculate_chunk_graphs", None)

    base_prompt = _base_prompt(custom_prompt)

    async def extract_group(group: list[DocumentChunk]) -> list[KnowledgeGraph]:
        restriction = await generate_restriction_from_chunks(group, **llm_kwargs)
        restricted_graph_model = build_restricted_knowledge_graph_model(restriction)
        prompt = build_restricted_prompt(base_prompt, restriction)
        return await asyncio.gather(
            *[
                extract_content_graph(
                    chunk.text,
                    restricted_graph_model,
                    custom_prompt=prompt,
                    **llm_kwargs,
                )
                for chunk in group
            ]
        )

    grouped_graphs = await asyncio.gather(
        *[
            extract_group(chunks[index : index + ONTOLOGY_CHUNKS_PER_GROUP])
            for index in range(0, len(chunks), ONTOLOGY_CHUNKS_PER_GROUP)
        ]
    )

    return [graph for group_graphs in grouped_graphs for graph in group_graphs]
