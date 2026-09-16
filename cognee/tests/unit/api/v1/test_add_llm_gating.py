from io import BytesIO
from types import SimpleNamespace

import pytest

from cognee.api.v1.add.add import _add_pipeline_needs_llm
from cognee.memify_pipelines.memify_default_tasks import (
    get_default_memify_enrichment_tasks,
    get_default_memify_extraction_tasks,
)
from cognee.modules.pipelines.tasks.task import pipeline_needs_llm
from cognee.tasks.ingestion.data_item import DataItem


@pytest.mark.parametrize(
    ("data", "preferred_loaders", "expected"),
    [
        ("Cognee turns documents into memory.", None, False),
        ([DataItem("Labeled document"), "More text"], None, False),
        ("https://example.com", None, True),
        (BytesIO(b"stream"), None, True),
        ("plain text", {"custom_loader": {}}, True),
    ],
)
def test_add_llm_requirement(data, preferred_loaders, expected):
    assert _add_pipeline_needs_llm(data, preferred_loaders) is expected


def test_add_file_input_keeps_llm_check(tmp_path):
    media_path = tmp_path / "image.png"
    media_path.write_bytes(b"not-an-image")

    assert _add_pipeline_needs_llm(str(media_path), preferred_loaders=None) is True


def test_default_memify_tasks_are_llm_free(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.cognify.config.get_cognify_config",
        lambda: SimpleNamespace(triplet_embedding=True),
    )

    tasks = [
        *get_default_memify_extraction_tasks(),
        *get_default_memify_enrichment_tasks(),
    ]

    assert pipeline_needs_llm(tasks) is False
