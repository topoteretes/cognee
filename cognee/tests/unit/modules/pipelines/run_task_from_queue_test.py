import asyncio
from queue import Queue

import cognee
from cognee.modules.pipelines.operations.run_tasks import run_tasks_base
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.databases.relational import create_db_and_tables


async def pipeline(data_queue):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    async def queue_consumer():
        while not data_queue.is_closed:
            if not data_queue.empty():
                yield data_queue.get()
            else:
                await asyncio.sleep(0.3)

    async def add_one(num_list):
        yield num_list[0] + 1

    async def multiply_by_two(num_list):
        yield num_list[0] * 2

    await create_db_and_tables()
    user = await get_default_user()

    tasks_run = run_tasks_base(
        [
            Task(queue_consumer),
            Task(add_one),
            Task(multiply_by_two),
        ],
        data=None,
        user=user,
    )

    results = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    index = 0
    async for result in tasks_run:
        assert result[0] == results[index], f"at {index = }: {result = } != {results[index] = }"
        index += 1


async def run_queue():
    data_queue = Queue()
    data_queue.is_closed = False

    async def queue_producer():
        for i in range(0, 10):
            data_queue.put(i)
            await asyncio.sleep(0.1)
        data_queue.is_closed = True

    await asyncio.gather(pipeline(data_queue), queue_producer())


def test_run_tasks_from_queue():
    asyncio.run(run_queue())


if __name__ == "__main__":
    asyncio.run(run_queue())
