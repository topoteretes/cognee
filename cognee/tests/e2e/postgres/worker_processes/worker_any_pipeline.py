"""The worker of the "any pipeline" driver: ``worker_any_pipeline.py <shard id>``.

It imports ``any_pipeline``, which registers the notes pipeline in this process,
and installs the stand-in extraction.

WORKER_E2E_LOST_SHARD: the worker of the cognify run's shard with this index
ends as if it were killed (exit code 137) before it takes its shard.
"""

import asyncio
import os
import sys
from uuid import UUID

import any_pipeline
import stand_in_extraction

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.execution import PipelineRunSpec, worker
from cognee.modules.pipelines.models import PipelineRunShard
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME


async def is_the_shard_to_lose(shard_id: UUID) -> bool:
    index = os.getenv("WORKER_E2E_LOST_SHARD")
    if index is None:
        return False
    async with get_relational_engine().get_async_session() as session:
        shard = await session.get(PipelineRunShard, shard_id)
    pipeline_name = PipelineRunSpec.from_json(shard.spec).pipeline_name
    return pipeline_name == COGNIFY_PIPELINE_NAME and str(shard.shard_index) == index


async def main(shard_id: UUID) -> None:
    stand_in_extraction.install(any_pipeline.extract)
    if await is_the_shard_to_lose(shard_id):
        os._exit(137)
    await worker.main(shard_id)


if __name__ == "__main__":
    asyncio.run(main(UUID(sys.argv[1])))
