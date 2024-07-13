import asyncio
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.Task import Task


async def main():
    def number_generator(num):
        for i in range(num):
            yield i + 1

    async def add_one(num):
        yield num + 1

    async def multiply_by_two(nums):
        for num in nums:
            yield num * 2

    async def add_one_to_batched_data(num):
        yield num + 1

    pipeline = run_tasks([
        Task(number_generator, task_config = {"batch_size": 1}),
        Task(add_one, task_config = {"batch_size": 5}),
        Task(multiply_by_two, task_config = {"batch_size": 1}),
        Task(add_one_to_batched_data),
    ], 10)

    async for result in pipeline:
        print("\n")
        print(result)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
