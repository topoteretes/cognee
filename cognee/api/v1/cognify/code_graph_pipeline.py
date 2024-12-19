import asyncio
import logging
from pathlib import Path

from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.tasks.repo_processor import (enrich_dependency_graph,
                                         expand_dependency_graph,
                                         get_repo_file_dependencies)
from cognee.tasks.storage import add_data_points

from cognee.base_config import get_base_config
from cognee.shared.data_models import MonitoringTool

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe

from cognee.tasks.summarization import summarize_code

logger = logging.getLogger("code_graph_pipeline")
update_status_lock = asyncio.Lock()

@observe
async def run_code_graph_pipeline(repo_path):
    import os
    import pathlib
    import cognee
    from cognee.infrastructure.databases.relational import create_db_and_tables

    file_path = Path(__file__).parent
    data_directory_path = str(pathlib.Path(os.path.join(file_path, ".data_storage/code_graph")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(pathlib.Path(os.path.join(file_path, ".cognee_system/code_graph")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()

    tasks = [
        Task(get_repo_file_dependencies),
        Task(enrich_dependency_graph, task_config={"batch_size": 50}),
        Task(expand_dependency_graph, task_config={"batch_size": 50}),
        Task(summarize_code, task_config={"batch_size": 50}),
        Task(add_data_points, task_config={"batch_size": 50}),
    ]

    return run_tasks(tasks, repo_path, "cognify_code_pipeline")
