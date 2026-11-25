"""What the "any pipeline" driver and its workers share.

- A pipeline of its own, registered by name, whose items are JSON values and no
  ``Data`` rows (:data:`NOTES`).
- A custom graph model and a stand-in for the model's extraction with it
  (:class:`Page`).
- Twelve CSV files, each a DLT source of its own (:func:`write_csv_files`).

A worker imports this module before it runs a shard: that is what registers the
notes pipeline in its process.
"""

from uuid import NAMESPACE_OID, uuid5

import stand_in_extraction

from cognee.infrastructure.engine import DataPoint
from cognee.modules.pipelines.execution import register_task_resolver_factory
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points

# ---- a pipeline of its own ----

NOTES = "e2e_notes"
NOTE_COUNT = 60


class Note(DataPoint):
    name: str
    text: str
    metadata: dict = {"index_fields": ["name"]}


def note_items() -> list[dict]:
    return [
        {"title": f"note {index:02d}", "text": f"Note {index:02d} is about item {index % 7}."}
        for index in range(NOTE_COUNT)
    ]


async def notes_from_items(items: list[dict], label: str) -> list[Note]:
    return [
        Note(
            id=uuid5(NAMESPACE_OID, f"{label}:{item['title']}"),
            name=f"{label} {item['title']}",
            text=item["text"],
        )
        for item in items
    ]


async def build_notes_tasks(*, user, label: str) -> list[Task]:
    """The notes pipeline's task list, built from its JSON argument."""
    return [
        Task(notes_from_items, label=label, needs_llm=False),
        Task(add_data_points, needs_llm=False),
    ]


register_task_resolver_factory(NOTES, build_notes_tasks)


# ---- a custom graph model ----


class Topic(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Page(DataPoint):
    name: str
    topics: list[Topic] = []
    metadata: dict = {"index_fields": ["name"]}


async def extract(content: str, response_model, **kwargs):
    """The stand-in extraction: the word graph for the default model, and for
    any other a page named after the text's first words with a topic per word.

    From a plain dict, so it fills the class a worker built from the model's
    schema as it fills the class the schema was made from.
    """
    if issubclass(response_model, KnowledgeGraph):
        return await stand_in_extraction.extract_content_graph(content, response_model, **kwargs)
    words = sorted({word.strip(".,").lower() for word in content.split() if len(word) > 3})
    return response_model.model_validate(
        {"name": " ".join(content.split()[:3]), "topics": [{"name": word} for word in words]}
    )


# ---- DLT sources ----

CSV_FILES = 12
ROWS_PER_FILE = 30


def write_csv_files(directory) -> list[str]:
    """Twelve CSV files of thirty rows each, and their paths. Every file is a DLT
    source of its own: one item of a cognify run."""
    paths = []
    for file in range(CSV_FILES):
        path = directory / f"orders_{file:02d}.csv"
        rows = "".join(
            f"{row},customer {file:02d}-{row:02d},{10 + row}\n" for row in range(ROWS_PER_FILE)
        )
        path.write_text("id,customer,amount\n" + rows)
        paths.append(str(path))
    return paths
