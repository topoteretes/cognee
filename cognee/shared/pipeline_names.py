"""Names of the pipelines cognify() runs.

A pipeline's name is what ``pipeline_runs.pipeline_name`` records for each run
and the key of the per-item completion stamp in ``Data.pipeline_status``
(``pipeline_status[pipeline_name][str(dataset_id)]``). Status readers select
runs by these names, so they live in one leaf module that imports nothing from
cognee.

``cognify()`` builds memory in two runs per dataset, in this order
(``cognee.modules.cognify.memory_pipelines``): indexing, then cognify.
"""

# add(): stages data; writes no graph datapoints.
ADD_PIPELINE_NAME = "add_pipeline"
# 1. Chunks the documents and stores + embeds the chunks: chunk search answers.
INDEXING_PIPELINE_NAME = "indexing_pipeline"
# 2. Entity/relationship extraction, summaries, the final write: graph search.
COGNIFY_PIPELINE_NAME = "cognify_pipeline"
# remember(content_type="code"): the deterministic enola code graph.
CODE_GRAPH_PIPELINE_NAME = "code_graph_pipeline"
