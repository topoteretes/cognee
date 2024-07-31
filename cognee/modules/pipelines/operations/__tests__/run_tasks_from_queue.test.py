import asyncio
from queue import Queue
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.Task import Task

async def pipeline(data_queue):
    async def queue_consumer():
        while not data_queue.is_closed:
            if not data_queue.empty():
                yield data_queue.get()
            else:
                await asyncio.sleep(0.3)

    async def add_one(num):
        yield num + 1

    async def multiply_by_two(num):
        yield num * 2

    tasks_run = run_tasks([
        Task(queue_consumer),
        Task(add_one),
        Task(multiply_by_two),
    ])

    results = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    index = 0
    async for result in tasks_run:
        print(result)
        assert result == results[index]
        index += 1

async def main():
    data_queue = Queue()
    data_queue.is_closed = False

    async def queue_producer():
        for i in range(0, 10):
            data_queue.put(i)
            await asyncio.sleep(0.1)
        data_queue.is_closed = True

    await asyncio.gather(pipeline(data_queue), queue_producer())

if __name__ == "__main__":
    asyncio.run(main())
