import inspect
import logging
from ..tasks.Task import Task

logger = logging.getLogger("run_tasks(tasks: [Task], data)")

async def run_tasks(tasks: [Task], data):
    if len(tasks) == 0:
        yield data
        return

    running_task = tasks[0]
    batch_size = running_task.task_config["batch_size"]
    leftover_tasks = tasks[1:]
    next_task = leftover_tasks[0] if len(leftover_tasks) > 1 else None
    # next_task_batch_size = next_task.task_config["batch_size"] if next_task else 1

    if inspect.isasyncgenfunction(running_task.executable):
        logger.info(f"Running async generator task: `{running_task.executable.__name__}`")
        try:
            results = []

            async_iterator = running_task.run(data)

            async for partial_result in async_iterator:
                results.append(partial_result)

                if len(results) == batch_size:
                    async for result in run_tasks(leftover_tasks, results[0] if batch_size == 1 else results):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks(leftover_tasks, results):
                    yield result

                results = []

            logger.info(f"Finished async generator task: `{running_task.executable.__name__}`")
        except Exception as error:
            logger.error(
                "Error occurred while running async generator task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            raise error

    elif inspect.isgeneratorfunction(running_task.executable):
        logger.info(f"Running generator task: `{running_task.executable.__name__}`")
        try:
            results = []

            for partial_result in running_task.run(data):
                results.append(partial_result)

                if len(results) == batch_size:
                    async for result in run_tasks(leftover_tasks, results[0] if batch_size == 1 else results):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks(leftover_tasks, results):
                    yield result

                results = []

            logger.info(f"Running generator task: `{running_task.executable.__name__}`")
        except Exception as error:
            logger.error(
                "Error occurred while running generator task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            raise error

    elif inspect.iscoroutinefunction(running_task.executable):
        task_result = await running_task.run(data)

        async for result in run_tasks(leftover_tasks, task_result):
            yield result

    elif inspect.isfunction(running_task.executable):
        task_result = running_task.run(data)

        async for result in run_tasks(leftover_tasks, task_result):
            yield result
