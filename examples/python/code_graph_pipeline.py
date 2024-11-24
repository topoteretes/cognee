import argparse
import asyncio
import os
import cognee
import json

import numpy as np
from networkx.classes.digraph import DiGraph

from cognee.modules.pipelines import Task, run_tasks
from cognee.shared.CodeGraphEntities import CodeFile, CodeRelationship, Repository
from cognee.shared.data_models import SummarizedContent
from cognee.tasks.code.get_local_dependencies_checker import (
    get_local_script_dependencies,
)
from cognee.tasks.graph.convert_graph_from_code_graph import (
    convert_graph_from_code_graph,
)
from cognee.tasks.repo_processor.get_repo_dependency_graph import (
    get_repo_dependency_graph,
)
from cognee.tasks.repo_processor.enrich_dependency_graph import enrich_dependency_graph
from cognee.tasks.summarization import summarize_code
from cognee.tasks.storage import index_data_points

async def print_results(pipeline):
    async for result in pipeline:
        print(result)

async def write_results(repo, pipeline):
    output_dir = os.path.join(repo, "code_pipeline_output", "")
    os.makedirs(output_dir, exist_ok = True)
    async for code_files, summaries in pipeline:
        for summary in summaries:
            file_name = os.path.split(summary.made_from.extracted_id)[-1]
            relpath = os.path.join(*os.path.split(os.path.relpath(summary.made_from.extracted_id, repo))[:-1])
            output_dir2 = os.path.join(repo, "code_pipeline_output", relpath)            
            os.makedirs(output_dir2, exist_ok=True)
            with open(os.path.join(output_dir2, file_name.replace(".py", ".json")), "w") as f:
                f.write(json.dumps({"summary": summary.text, "source_code": summary.made_from.source_code}))

async def reset_system():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    return(True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a file path")
    parser.add_argument("path", help="Path to the file")

    args = parser.parse_args()
    abspath = os.path.abspath(args.path)
    data = abspath
    tasks = [
        Task(get_repo_dependency_graph),
        Task(enrich_dependency_graph),
        Task(convert_graph_from_code_graph, repo_path = abspath),
        Task(index_data_points),
        Task(summarize_code, summarization_model=SummarizedContent),
    ]
    pipeline = run_tasks(tasks, data, "cognify_pipeline")

    asyncio.run(write_results(abspath, pipeline))
