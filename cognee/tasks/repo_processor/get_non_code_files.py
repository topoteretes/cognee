import os

import aiofiles

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.methods import get_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.methods.get_datasets_by_name import \
    get_datasets_by_name
from cognee.modules.data.models import Data
from cognee.modules.data.operations.write_metadata import write_metadata
from cognee.modules.ingestion.data_types import BinaryData
from cognee.modules.users.methods import get_default_user
from cognee.shared.CodeGraphEntities import Repository


async def get_non_py_files(repo_path):
    """Get files that are not .py files and their contents"""
    if not os.path.exists(repo_path):
        return {}

    IGNORED_PATTERNS = {
        '.git', '__pycache__', '*.pyc', '*.pyo', '*.pyd',
        'node_modules', '*.egg-info'
    }

    def should_process(path):
        return not any(pattern in path for pattern in IGNORED_PATTERNS)

    non_py_files_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path) for file in files 
        if not file.endswith(".py") and should_process(os.path.join(root, file))
    ]
    return non_py_files_paths


async def get_data_list_for_user(_, dataset_name, user):
    # Note: This method is meant to be used as a Task in a pipeline.
    # By the nature of pipelines, the output of the previous Task will be passed as the first argument here,
    # but it is not needed here, hence the "_" input.
    datasets = await get_datasets_by_name(dataset_name, user.id)
    data_documents: list[Data] = []
    for dataset in datasets:
        data_docs: list[Data] = await get_dataset_data(dataset_id=dataset.id)
        data_documents.extend(data_docs)
    return data_documents