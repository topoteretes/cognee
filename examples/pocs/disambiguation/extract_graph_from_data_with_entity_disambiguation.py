import asyncio
from typing import Type, List, Optional, Dict
from pydantic import BaseModel

from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.tasks.graph.exceptions import (
    InvalidGraphModelError,
    InvalidDataChunksError,
)
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs

from cognee.infrastructure.databases.vector import get_vector_engine_async


async def build_prompt(chunk, vector_search_limit, custom_prompt) -> Optional[str]:
    vector_engine = await get_vector_engine_async()
    exists = await vector_engine.has_collection(collection_name="Entity_name")

    if not (exists and custom_prompt):
        return custom_prompt

    results_per_chunk = await vector_engine.search(
        collection_name="Entity_name",
        query_text=chunk.text,
        limit=vector_search_limit,
        include_payload=True,
    )
    prompt = custom_prompt
    for result in results_per_chunk:
        prompt = prompt + "\n  -" + result.payload["text"]
    return prompt


async def extract_graph_from_data_with_entity_disambiguation_task(
    data_chunks: List[DocumentChunk],
    context: Dict,
    graph_model: Type[BaseModel],
    config: Config = None,
    custom_prompt: Optional[str] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """
    Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model.
    """
    vector_search_limit = kwargs.get("vector_search_limit") or 5

    if not isinstance(data_chunks, list) or not data_chunks:
        raise InvalidDataChunksError("must be a non-empty list of DocumentChunk.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidDataChunksError("each chunk must have a 'text' attribute")
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)

    chunk_prompts = await asyncio.gather(
        *[build_prompt(chunk, vector_search_limit, custom_prompt) for chunk in data_chunks]
    )

    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=prompt)
            for chunk, prompt in zip(data_chunks, chunk_prompts)
        ]
    )

    # Extract resolver from config if provided, otherwise get default
    if config is None:
        ontology_config = get_ontology_env_config()
        if (
            ontology_config.ontology_file_path
            and ontology_config.ontology_resolver
            and ontology_config.matching_strategy
        ):
            config: Config = {
                "ontology_config": {
                    "ontology_resolver": get_ontology_resolver_from_env(**ontology_config.to_dict())
                }
            }
        else:
            config: Config = {
                "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
            }

    ontology_resolver = config["ontology_config"]["ontology_resolver"]

    pipeline_name = context.get("pipeline_name") if isinstance(context, dict) else None
    task_name = "extract_graph_from_data"

    return await integrate_chunk_graphs(
        data_chunks,
        chunk_graphs,
        graph_model,
        ontology_resolver,
        pipeline_name=pipeline_name,
        task_name=task_name,
    )
