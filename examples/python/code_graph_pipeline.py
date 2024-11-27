import asyncio
from cognee.modules.pipelines import Task, run_tasks
from cognee.tasks.repo_processor import (
    enrich_dependency_graph,
    expand_dependency_graph,
    get_repo_file_dependencies,
)
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_code


async def print_results(pipeline):
    async for result in pipeline:
        print(result)

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Process a file path")
    parser.add_argument("path", help="Path to the file")

    args = parser.parse_args()
    abspath = os.path.abspath(args.path or ".")
    '''

    abspath = '/Users/laszlohajdu/Documents/Github/RAW_GIT_REPOS/astropy__astropy-12907'
    tasks = [
        Task(get_repo_file_dependencies),
        Task(add_data_points),
        Task(enrich_dependency_graph),
        Task(expand_dependency_graph),
        Task(add_data_points),
        # Task(summarize_code, summarization_model = SummarizedContent),
    ]
    pipeline = run_tasks(tasks, abspath, "cognify_code_pipeline")

    asyncio.run(print_results(pipeline))
