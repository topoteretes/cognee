from uuid import UUID
from asyncio import Queue
from typing import Optional

from cognee.modules.pipelines.models import PipelineRunInfo


pipeline_run_info_queues = {}


def initialize_queue(pipeline_run_id: UUID):
    pipeline_run_info_queues[str(pipeline_run_id)] = Queue()


def get_queue(pipeline_run_id: UUID) -> Optional[Queue]:
    if str(pipeline_run_id) not in pipeline_run_info_queues:
        initialize_queue(pipeline_run_id)

    return pipeline_run_info_queues.get(str(pipeline_run_id), None)


def remove_queue(pipeline_run_id: UUID):
    pipeline_run_info_queues.pop(str(pipeline_run_id))


def push_to_queue(pipeline_run_id: UUID, pipeline_run_info: PipelineRunInfo):
    queue = get_queue(pipeline_run_id)

    if queue:
        queue.put_nowait(pipeline_run_info)


def get_from_queue(pipeline_run_id: UUID) -> Optional[PipelineRunInfo]:
    queue = get_queue(pipeline_run_id)

    item = queue.get_nowait() if queue and not queue.empty() else None
    return item
