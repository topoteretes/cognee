import inspect
import logging
from ..tasks.Task import Task

logger = logging.getLogger("run_tasks(tasks: [Task], data)")

async def run_tasks(tasks: [Task], data = None):
    if len(tasks) == 0:
        yield data
        return

    args = [data] if data is not None else []

    running_task = tasks[0]
    leftover_tasks = tasks[1:]
    next_task = leftover_tasks[0] if len(leftover_tasks) > 1 else None
    next_task_batch_size = next_task.task_config["batch_size"] if next_task else 1

    if inspect.isasyncgenfunction(running_task.executable):
        logger.info("Running async generator task: `%s`", running_task.executable.__name__)
        try:
            results = []

            async_iterator = running_task.run(*args)

            async for partial_result in async_iterator:
                results.append(partial_result)

                if len(results) == next_task_batch_size:
                    async for result in run_tasks(leftover_tasks, results[0] if next_task_batch_size == 1 else results):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks(leftover_tasks, results):
                    yield result

                results = []

            logger.info("Finished async generator task: `%s`", running_task.executable.__name__)
        except Exception as error:
            logger.error(
                "Error occurred while running async generator task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            raise error

    elif inspect.isgeneratorfunction(running_task.executable):
        logger.info("Running generator task: `%s`", running_task.executable.__name__)
        try:
            results = []

            for partial_result in running_task.run(*args):
                results.append(partial_result)

                if len(results) == next_task_batch_size:
                    async for result in run_tasks(leftover_tasks, results[0] if next_task_batch_size == 1 else results):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks(leftover_tasks, results):
                    yield result

                results = []

            logger.info("Finished generator task: `%s`", running_task.executable.__name__)
        except Exception as error:
            logger.error(
                "Error occurred while running generator task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            raise error

    elif inspect.iscoroutinefunction(running_task.executable):
        logger.info("Running coroutine task: `%s`", running_task.executable.__name__)
        try:
            task_result = await running_task.run(*args)

            async for result in run_tasks(leftover_tasks, task_result):
                yield result

            logger.info("Finished coroutine task: `%s`", running_task.executable.__name__)
        except Exception as error:
            logger.error(
                "Error occurred while running coroutine task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
                
    elif inspect.isfunction(running_task.executable):
        logger.info("Running function task: `%s`", running_task.executable.__name__)
        try:
            task_result = running_task.run(*args)

            async for result in run_tasks(leftover_tasks, task_result):
                yield result

            logger.info("Finished function task: `%s`", running_task.executable.__name__)
        except Exception as error:
            logger.error(
                "Error occurred while running function task: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
