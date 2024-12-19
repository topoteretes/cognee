import asyncio
import logging
from pathlib import Path
from typing import Union

from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import \
    get_pipeline_status
from cognee.modules.pipelines.operations.log_pipeline_status import \
    log_pipeline_status
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.SourceCodeGraph import SourceCodeGraph
from cognee.shared.utils import send_telemetry
from cognee.tasks.documents import (check_permissions_on_documents,
                                    classify_documents,
                                    extract_chunks_from_documents)
from cognee.tasks.graph import extract_graph_from_code
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
