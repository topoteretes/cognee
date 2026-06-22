import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.pipelines.tasks.task import Task

task_getter_module = importlib.import_module(
    "cognee.eval_framework.corpus_builder.task_getters.get_default_tasks_by_indices"
)


@pytest.mark.asyncio
@patch.object(task_getter_module, "RDFLibOntologyResolver")
@patch.object(task_getter_module, "get_default_tasks", new_callable=AsyncMock)
async def test_get_no_summary_tasks_passes_ontology_resolver_via_config(
    mock_get_default_tasks,
    mock_ontology_resolver_class,
):
    mock_get_default_tasks.return_value = [
        Task(lambda: None),
        Task(lambda: None),
        Task(lambda: None),
    ]
    ontology_resolver = MagicMock()
    mock_ontology_resolver_class.return_value = ontology_resolver

    tasks = await task_getter_module.get_no_summary_tasks(ontology_file_path="ontology.rdf")

    graph_task = tasks[2]

    assert graph_task.executable is task_getter_module.extract_graph_from_data
    assert graph_task.default_params["kwargs"]["config"] == {
        "ontology_config": {"ontology_resolver": ontology_resolver}
    }
    assert "ontology_adapter" not in graph_task.default_params["kwargs"]


@pytest.mark.asyncio
@patch.object(task_getter_module, "get_default_tasks", new_callable=AsyncMock)
async def test_get_no_summary_tasks_omits_config_when_no_ontology_file(
    mock_get_default_tasks,
):
    mock_get_default_tasks.return_value = [
        Task(lambda: None),
        Task(lambda: None),
        Task(lambda: None),
    ]

    tasks = await task_getter_module.get_no_summary_tasks()

    graph_task = tasks[2]

    assert graph_task.executable is task_getter_module.extract_graph_from_data
    assert "config" not in graph_task.default_params["kwargs"]
