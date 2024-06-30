import os
import asyncio
import inspect
from typing import Union, Callable, Generator, AsyncGenerator, Coroutine, Any
from cognee.modules.cognify.vector import save_data_chunks
from cognee.modules.data.processing.process_documents import process_documents
from cognee.modules.data.extraction.knowledge_graph.expand_knowledge_graph import expand_knowledge_graph
from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.deletion.prune_system import prune_system


class Task():
    executable: Union[
        Callable[..., Any],
        Callable[..., Coroutine[Any, Any, Any]],
        Generator[Any, Any, Any],
        AsyncGenerator[Any, Any],
    ]
    task_config: dict[str, Any] = {
        "batch_size": 1,
    }
    default_params: dict[str, Any] = {}

    def __init__(self, executable, *args, task_config = None, **kwargs):
        self.executable = executable
        self.default_params = {
            "args": args,
            "kwargs": kwargs
        }

        if task_config is not None:
            self.task_config = task_config

    def run(self, *args, **kwargs):
        combined_args = self.default_params["args"] + args
        combined_kwargs = { **self.default_params["kwargs"], **kwargs }
        return self.executable(*combined_args, **combined_kwargs)


def run_tasks(tasks: [Task], data):
    if len(tasks) == 0:
        yield data
        return

    running_task = tasks[0]
    batch_size = running_task.task_config["batch_size"]

    if inspect.isasyncgenfunction(running_task.executable):
        results = []
        async_running_task = running_task.run(data)

        while True:
            try:
                partial_result = asyncio.run(anext(async_running_task))
                results.append(partial_result)

                if len(results) == batch_size:
                    yield from run_tasks(tasks[1:], results[0] if batch_size == 1 else results)
                    results = []
            except StopAsyncIteration:
                if len(results) > 0:
                    yield from run_tasks(tasks[1:], results[0] if batch_size == 1 else results)
                    results = []
                break

    elif inspect.isgeneratorfunction(running_task.executable):
        results = []

        for partial_result in running_task.run(data):
            results.append(partial_result)

            if len(results) == batch_size:
                yield from run_tasks(tasks[1:], results[0] if batch_size == 1 else results)
                results = []

        if len(results) > 0:
            yield from run_tasks(tasks[1:], results[0] if batch_size == 1 else results)
            results = []

    elif inspect.iscoroutinefunction(running_task.executable):
        result = asyncio.run(running_task.run(data))
        yield from run_tasks(tasks[1:], result)

    elif inspect.isfunction(running_task.executable):
        result = running_task.run(data)
        yield from run_tasks(tasks[1:], result)


if __name__ == "__main__":
    def main():
        tasks = [
            Task(process_documents, parent_node_id = "Boris's documents", task_config = { "batch_size": 10 }), # Classify documents and save them as a nodes in graph db, extract text chunks based on the document type
            Task(save_data_chunks, collection_name = "chunks"), # Save the document chunks in vector db and as nodes in graph db (connected to the document node and between each other)
            Task(expand_knowledge_graph, graph_model = KnowledgeGraph), # Generate knowledge graphs from the document chunks and attach it to chunk nodes
        ]

        test_file_path = os.path.join(os.path.dirname(__file__), "./__tests__/artificial-inteligence.pdf")

        test_document = PdfDocument(
            title = "Artificial intelligence",
            file_path = test_file_path,
        )

        asyncio.run(prune_system())

        for chunk in run_tasks(tasks, [test_document]):
            print(chunk)
            print("\n")

    main()
