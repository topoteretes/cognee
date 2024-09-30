import inspect
import logging
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from ..tasks.Task import Task

logger = logging.getLogger("run_tasks(tasks: [Task], data)")

async def run_tasks_base(tasks: [Task], data = None, user: User = None):
    if len(tasks) == 0:
        yield data
        return

    args = [data] if data is not None else []

    running_task = tasks[0]
    leftover_tasks = tasks[1:]
    next_task = leftover_tasks[0] if len(leftover_tasks) > 1 else None
    next_task_batch_size = next_task.task_config["batch_size"] if next_task else 1

    if inspect.isasyncgenfunction(running_task.executable):
        logger.info("Async generator task started: `%s`", running_task.executable.__name__)
        send_telemetry("Async Generator Task Started", user.id, {
            "task_name": running_task.executable.__name__,
        })
        try:
            results = []

            async_iterator = running_task.run(*args)

            async for partial_result in async_iterator:
                results.append(partial_result)

                if len(results) == next_task_batch_size:
                    async for result in run_tasks_base(
                        leftover_tasks,
                        results[0] if next_task_batch_size == 1 else results,
                        user = user,
                    ):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks_base(leftover_tasks, results, user):
                    yield result

                results = []

            logger.info("Async generator task completed: `%s`", running_task.executable.__name__)
            send_telemetry("Async Generator Task Completed", user.id, {
                "task_name": running_task.executable.__name__,
            })
        except Exception as error:
            logger.error(
                "Async generator task errored: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            send_telemetry("Async Generator Task Errored", user.id, {
                "task_name": running_task.executable.__name__,
            })
            raise error

    elif inspect.isgeneratorfunction(running_task.executable):
        logger.info("Generator task started: `%s`", running_task.executable.__name__)
        send_telemetry("Generator Task Started", user.id, {
            "task_name": running_task.executable.__name__,
        })
        try:
            results = []

            for partial_result in running_task.run(*args):
                results.append(partial_result)

                if len(results) == next_task_batch_size:
                    async for result in run_tasks_base(leftover_tasks, results[0] if next_task_batch_size == 1 else results, user):
                        yield result

                    results = []

            if len(results) > 0:
                async for result in run_tasks_base(leftover_tasks, results, user):
                    yield result

                results = []

            logger.info("Generator task completed: `%s`", running_task.executable.__name__)
            send_telemetry("Generator Task Completed", user_id = user.id, additional_properties = {
                "task_name": running_task.executable.__name__,
            })
        except Exception as error:
            logger.error(
                "Generator task errored: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            send_telemetry("Generator Task Errored", user_id = user.id, additional_properties = {
                "task_name": running_task.executable.__name__,
            })
            raise error

    elif inspect.iscoroutinefunction(running_task.executable):
        logger.info("Coroutine task started: `%s`", running_task.executable.__name__)
        send_telemetry("Coroutine Task Started", user_id = user.id, additional_properties = {
            "task_name": running_task.executable.__name__,
        })
        try:
            task_result = await running_task.run(*args)

            async for result in run_tasks_base(leftover_tasks, task_result, user):
                yield result

            logger.info("Coroutine task completed: `%s`", running_task.executable.__name__)
            send_telemetry("Coroutine Task Completed", user.id, {
                "task_name": running_task.executable.__name__,
            })
        except Exception as error:
            logger.error(
                "Coroutine task errored: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            send_telemetry("Coroutine Task Errored", user.id, {
                "task_name": running_task.executable.__name__,
            })
            raise error

    elif inspect.isfunction(running_task.executable):
        logger.info("Function task started: `%s`", running_task.executable.__name__)
        send_telemetry("Function Task Started", user.id, {
            "task_name": running_task.executable.__name__,
        })
        try:
            task_result = running_task.run(*args)

            async for result in run_tasks_base(leftover_tasks, task_result, user):
                yield result

            logger.info("Function task completed: `%s`", running_task.executable.__name__)
            send_telemetry("Function Task Completed", user.id, {
                "task_name": running_task.executable.__name__,
            })
        except Exception as error:
            logger.error(
                "Function task errored: `%s`\n%s\n",
                running_task.executable.__name__,
                str(error),
                exc_info = True,
            )
            send_telemetry("Function Task Errored", user.id, {
                "task_name": running_task.executable.__name__,
            })
            raise error

async def run_tasks(tasks: [Task], data = None, pipeline_name: str = "default_pipeline"):
    user = await get_default_user()

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry("Pipeline Run Started", user.id, {
            "pipeline_name": pipeline_name,
        })

        async for result in run_tasks_base(tasks, data, user):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry("Pipeline Run Completed", user.id, {
            "pipeline_name": pipeline_name,
        })
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info = True,
        )
        send_telemetry("Pipeline Run Errored", user.id, {
            "pipeline_name": pipeline_name,
        })

        raise error
