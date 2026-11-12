"""cognee's process worker with the stand-in extraction: ``worker.py <shard id>``.

What an application gives ``ProcessWorkerBackend`` as its worker command when
its workers need setup of their own before they run a shard.

WORKER_E2E_LOST_SHARD: the worker of the cognify run's shard with this index
ends as if it were killed (exit code 137) when it reaches its fourth document.
"""

import asyncio
import os
import sys
from uuid import UUID

import stand_in_extraction

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.execution import PipelineRunSpec, worker
from cognee.modules.pipelines.models import PipelineRunShard
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

DOCUMENTS_BEFORE_IT_ENDS = 3


async def is_the_shard_to_lose(shard_id: UUID) -> bool:
    index = os.getenv("WORKER_E2E_LOST_SHARD")
    if index is None:
        return False
    async with get_relational_engine().get_async_session() as session:
        shard = await session.get(PipelineRunShard, shard_id)
    pipeline_name = PipelineRunSpec.from_json(shard.spec).pipeline_name
    return pipeline_name == COGNIFY_PIPELINE_NAME and str(shard.shard_index) == index


async def main(shard_id: UUID) -> None:
    extract = stand_in_extraction.extract_content_graph
    if await is_the_shard_to_lose(shard_id):
        one_at_a_time, extracted = asyncio.Semaphore(1), 0

        async def extract(content, response_model, **kwargs):
            nonlocal extracted
            async with one_at_a_time:
                if extracted == DOCUMENTS_BEFORE_IT_ENDS:
                    # The documents before this one get to store their work.
                    await asyncio.sleep(2)
                    os._exit(137)
                extracted += 1
            return await stand_in_extraction.extract_content_graph(
                content, response_model, **kwargs
            )

    stand_in_extraction.install(extract)
    await worker.main(shard_id)


if __name__ == "__main__":
    asyncio.run(main(UUID(sys.argv[1])))
