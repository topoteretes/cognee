"""GLiNER demo: LLM-free knowledge-graph extraction and summaries (SDK-537).

Selected with ``GRAPH_EXTRACTOR=gliner_demo`` or ``cognify(extractor="gliner_demo")``,
and by default (``GRAPH_EXTRACTOR=auto``) whenever no usable LLM key is configured.

DEMO: The open-source GLiNER extractor is a demo of cognee's enterprise GLiNER
extraction. It is free to use and needs no LLM key; the production-grade version
(higher accuracy, broader label coverage) is available with a cognee enterprise
licence. The first run with it logs that notice once.

Interested in the production-grade GLiNER extraction? Write to us at
social@cognee.ai to explore the options.

Usage::

    from cognee.tasks.graph.gliner_demo import get_gliner_demo_tasks

    tasks = await get_gliner_demo_tasks(entity_types=[...], relation_types=[...])
    await cognee.run_custom_pipeline(
        tasks=tasks, user=user, dataset="...", pipeline_name="cognify_pipeline"
    )
    # Every task declares needs_llm=False, so the first-run check probes only
    # the embeddings this pipeline actually uses.

Requires the ``gliner`` extra (``pip install "cognee[gliner]"``).
"""

from .banks import LABEL_BANK, RELATION_BANK
from .extractor import (
    DEFAULT_MODEL,
    GlinerNotInstalledError,
    build_gliner_schema,
    extract_batch,
    get_extractor,
    load_extractor,
)
from .mapping import MappedChunk, knowledge_graph_from_gliner_result, map_gliner_result
from .schema import (
    MAX_TYPES,
    GlinerSchema,
    resolve_schema,
    schema_from_label_bank,
    schema_from_ontology,
    to_snake_case,
)
from .summary import build_text_summary, format_chunk_summary
from .tasks import (
    GlinerOptions,
    GlinerRunStats,
    build_gliner_extraction_task,
    extract_graph_and_summarize_with_gliner,
    get_gliner_demo_tasks,
)

__all__ = [
    "DEFAULT_MODEL",
    "LABEL_BANK",
    "MAX_TYPES",
    "RELATION_BANK",
    "GlinerNotInstalledError",
    "GlinerOptions",
    "GlinerRunStats",
    "GlinerSchema",
    "MappedChunk",
    "build_gliner_extraction_task",
    "build_gliner_schema",
    "build_text_summary",
    "extract_batch",
    "extract_graph_and_summarize_with_gliner",
    "format_chunk_summary",
    "get_extractor",
    "get_gliner_demo_tasks",
    "knowledge_graph_from_gliner_result",
    "load_extractor",
    "map_gliner_result",
    "resolve_schema",
    "schema_from_label_bank",
    "schema_from_ontology",
    "to_snake_case",
]
