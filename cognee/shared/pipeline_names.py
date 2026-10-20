"""Names of the pipelines cognee runs.

A pipeline's name is what ``pipeline_runs.pipeline_name`` records for each run
and the key of the per-item completion stamp in ``Data.pipeline_status``
(``pipeline_status[pipeline_name][str(dataset_id)]``). Status readers — the
dataset status endpoints, MCP ``cognify_status``, recall's readiness probe —
select runs by these names, so they live in one leaf module that imports
nothing from cognee.

``cognify()`` builds memory in three runs per dataset, in this order
(``cognee.modules.cognify.memory_pipelines``): caching, indexing, cognify.
"""

# add(): stages data; writes no graph datapoints.
ADD_PIPELINE_NAME = "add_pipeline"
# 1. Stores each document's text so keyword (BM25) searches answer at once.
CACHING_PIPELINE_NAME = "caching_pipeline"
# 2. Chunks the documents and stores + embeds the chunks: vector search.
INDEXING_PIPELINE_NAME = "indexing_pipeline"
# 3. Entity/relationship extraction, summaries, the final write: graph search.
COGNIFY_PIPELINE_NAME = "cognify_pipeline"
# remember(content_type="code"): the deterministic enola code graph.
CODE_GRAPH_PIPELINE_NAME = "code_graph_pipeline"

# The two runs that make a dataset searchable before its graph exists.
MEMORY_PREPARATION_PIPELINE_NAMES = (CACHING_PIPELINE_NAME, INDEXING_PIPELINE_NAME)
# Pipelines that stage, cache or index without building the knowledge graph.
# Every other named pipeline that completes (cognify, code graph, memify,
# custom graph-building pipelines) counts as a graph build.
NON_GRAPH_PIPELINE_NAMES = (ADD_PIPELINE_NAME, *MEMORY_PREPARATION_PIPELINE_NAMES)
