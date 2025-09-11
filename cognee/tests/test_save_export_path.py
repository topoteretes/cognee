import os
import asyncio
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_save_uses_custom_export_path(tmp_path, monkeypatch):
    # Import target after tmp fixtures are ready
    from cognee.api.v1.save import save as save_mod

    # Prepare two mock datasets
    class Dataset:
        def __init__(self, id_, name):
            self.id = id_
            self.name = name

    ds1 = Dataset(uuid4(), "dataset_alpha")
    ds2 = Dataset(uuid4(), "dataset_beta")

    # Mock dataset discovery
    async def mock_get_authorized_existing_datasets(datasets, permission_type, user):
        return [ds1, ds2]

    monkeypatch.setattr(
        save_mod, "get_authorized_existing_datasets", mock_get_authorized_existing_datasets
    )

    # Mock data items (with filename collision in ds1)
    class DataItem:
        def __init__(self, id_, name, original_path=None):
            self.id = id_
            self.name = name
            self.original_data_location = original_path

    ds1_items = [
        DataItem(uuid4(), "report.txt", "/root/a/report.txt"),
        DataItem(uuid4(), "report.txt", "/root/b/report.txt"),  # collision
    ]
    ds2_items = [
        DataItem(uuid4(), "notes.md", "/root/x/notes.md"),
    ]

    async def mock_get_dataset_data(dataset_id):
        if dataset_id == ds1.id:
            return ds1_items
        if dataset_id == ds2.id:
            return ds2_items
        return []

    monkeypatch.setattr(save_mod, "get_dataset_data", mock_get_dataset_data)

    # Mock summary retrieval
    async def mock_get_document_summaries_text(data_id: str) -> str:
        return "This is a summary."

    monkeypatch.setattr(save_mod, "_get_document_summaries_text", mock_get_document_summaries_text)

    # Mock questions
    async def mock_generate_questions(file_name: str, summary_text: str):
        return ["Q1?", "Q2?", "Q3?"]

    monkeypatch.setattr(save_mod, "_generate_questions", mock_generate_questions)

    # Mock searches per question
    async def mock_run_searches_for_question(question, dataset_id, search_types, top_k):
        return {st.value: [f"{question} -> ok"] for st in search_types}

    monkeypatch.setattr(save_mod, "_run_searches_for_question", mock_run_searches_for_question)

    # Use custom export path
    export_dir = tmp_path / "my_exports"
    export_dir_str = str(export_dir)

    # Run
    result = await save_mod.save(
        datasets=None,
        export_root_directory=export_dir_str,
        max_questions=3,
        search_types=["GRAPH_COMPLETION", "INSIGHTS", "CHUNKS"],
        top_k=2,
        include_summary=True,
        include_ascii_tree=True,
        concurrency=2,
        timeout=None,
    )

    # Verify returned mapping points to our custom path
    assert str(ds1.id) in result and str(ds2.id) in result
    assert result[str(ds1.id)].startswith(export_dir_str)
    assert result[str(ds2.id)].startswith(export_dir_str)

    # Verify directories and files exist
    ds1_dir = result[str(ds1.id)]
    ds2_dir = result[str(ds2.id)]

    assert os.path.isdir(ds1_dir)
    assert os.path.isdir(ds2_dir)

    # index.md present
    assert os.path.isfile(os.path.join(ds1_dir, "index.md"))
    assert os.path.isfile(os.path.join(ds2_dir, "index.md"))

    # File markdowns exist; collision handling: two files with similar base
    ds1_files = [f for f in os.listdir(ds1_dir) if f.endswith(".md") and f != "index.md"]
    assert len(ds1_files) == 2
    assert any(f == "report.txt.md" for f in ds1_files)
    assert any(f.startswith("report.txt__") and f.endswith(".md") for f in ds1_files)

    # Content sanity: ensure question headers exist in one file
    sample_md_path = os.path.join(ds1_dir, ds1_files[0])
    with open(sample_md_path, "r", encoding="utf-8") as fh:
        content = fh.read()
        assert "## Question ideas" in content
        assert "## Searches" in content
