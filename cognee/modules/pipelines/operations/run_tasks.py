import asyncio
import inspect
from typing import Union, Callable, Generator, AsyncGenerator, Coroutine, Any
from cognee.modules.data.chunking import chunk_by_paragraph
from cognee.modules.cognify.vector import save_data_chunks


class Task():
    executable: Union[
        Callable[..., Any],
        Callable[..., Coroutine[Any, Any, Any]],
        Generator[Any, Any, Any],
        AsyncGenerator[Any, Any],
    ]
    default_params: dict[str, Any] = {}

    def __init__(self, executable, *args, **kwargs):
        self.executable = executable
        self.default_params = {
            "args": args,
            "kwargs": kwargs
        }

    def run(self, *args, **kwargs):
        combined_args = self.default_params["args"] + args
        combined_kwargs = { **self.default_params["kwargs"], **kwargs }
        return self.executable(*combined_args, **combined_kwargs)


def run_tasks(tasks: [Task], data, batch_size = 1):
    if len(tasks) == 0:
        yield data
        return

    running_task = tasks[0]

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
                break

        if len(results) > 0:
            yield from run_tasks(tasks[1:], results)

    elif inspect.isgeneratorfunction(running_task.executable):
        results = []

        for partial_result in running_task.run(data):
            results.append(partial_result)

            if len(results) == batch_size:
                yield from run_tasks(tasks[1:], results[0] if batch_size == 1 else results)
                results = []

        if len(results) > 0:
            yield from run_tasks(tasks[1:], results)

    elif inspect.iscoroutinefunction(running_task.executable):
        result = asyncio.run(running_task.run(data))
        yield from run_tasks(tasks[1:], result)

    elif inspect.isfunction(running_task.executable):
        result = running_task.run(data)
        yield from run_tasks(tasks[1:], result)


if __name__ == "__main__":
    tasks = [
        Task(process_documents), # Identify documents and save then as a nodes in graph db
        Task(extract_document_text), # Extract text based on the document type
        Task(chunk_by_paragraph, paragraph_length = 1000, task_config = { "batch_size": 10 }), # Chunk the text into paragraphs
        Task(save_data_chunks, collection_name = "paragraphs"), # Save the paragraph chunks in vector db and as nodes in graph db (connected to the document node and between each other)
        Task(generate_knowledge_graph), # Generate knowledge graphs from the paragraph chunks and attach it to chunk nodes
    ]

    text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry... Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book… It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

        Why do we use it?
        It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout! The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).
    """

    for chunk in run_tasks(tasks, text):
        print(chunk)
        print("\n")
