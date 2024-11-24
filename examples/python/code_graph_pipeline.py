import argparse
import asyncio
import os

from cognee.modules.pipelines import Task, run_tasks
from cognee.shared.CodeGraphEntities import CodeRelationship, Repository
from cognee.shared.data_models import SummarizedContent
from cognee.tasks.code.get_local_dependencies_checker import (
    get_local_script_dependencies,
)
from cognee.tasks.graph.convert_graph_from_code_graph import (
    create_code_file,
    convert_graph_from_code_graph,
)
from cognee.tasks.repo_processor import (
    enrich_dependency_graph,
    expand_dependency_graph,
    get_repo_dependency_graph,
)
from cognee.tasks.summarization import summarize_code


async def print_results(pipeline):
    async for result in pipeline:
        print(result)


async def get_local_script_dependencies_wrapper(script_path, repo_path):
    dependencies = await get_local_script_dependencies(script_path, repo_path)
    return (script_path, dependencies)


async def scan_repo(path, condition):
    futures = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if condition(file):
                futures.append(
                    get_local_script_dependencies_wrapper(
                        os.path.abspath(f"{root}/{file}"), path
                    )
                )
    results = await asyncio.gather(*futures)

    code_files = {}
    code_relationships = []
    for abspath, dependencies in results:
        code_file, abspath = create_code_file(abspath, "python_file")
        code_files[abspath] = code_file

        for dependency in dependencies:
            dependency_code_file, dependency_abspath = create_code_file(
                dependency, "python_file"
            )
            code_files[dependency_abspath] = dependency_code_file
            code_relationship = CodeRelationship(
                source_id=abspath,
                target_id=dependency_abspath,
                type="files",
                relation="depends_on",
            )
            code_relationships.append(code_relationship)

    return (Repository(path=path), list(code_files.values()), code_relationships)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file path")
    parser.add_argument("path", help="Path to the file")

    args = parser.parse_args()
    abspath = os.path.abspath(args.path or ".")
    tasks = [
        Task(get_repo_dependency_graph),
        Task(enrich_dependency_graph),
        Task(expand_dependency_graph),
        Task(convert_graph_from_code_graph),
        Task(summarize_code, summarization_model = SummarizedContent),
    ]
    pipeline = run_tasks(tasks, abspath, "cognify_code_pipeline")
    asyncio.run(print_results(pipeline))
