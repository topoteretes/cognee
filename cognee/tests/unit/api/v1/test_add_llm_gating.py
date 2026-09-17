import importlib
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

preflight_mod = importlib.import_module("cognee.modules.preflight")


@pytest.fixture(autouse=True)
def with_llm(monkeypatch):
    """A configured LLM, so each case below tests what the *input* implies.

    Without this the answer is False for everything whenever the environment
    has no LLM_API_KEY, and these cases would silently stop testing anything.
    """
    monkeypatch.setattr(preflight_mod, "llm_available", lambda *_args, **_kwargs: True)


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(preflight_mod, "llm_available", lambda *_args, **_kwargs: False)


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


@pytest.mark.parametrize(
    ("data", "preferred_loaders"),
    [
        ("https://example.com", None),
        (BytesIO(b"stream"), None),
        ("plain text", {"custom_loader": {}}),
    ],
)
def test_keyless_setups_never_require_the_llm(data, preferred_loaders, no_llm):
    """With no LLM configured, the checks this gates can only restate the missing key.

    Keyless ingestion (local extractor, local embedder) has to work for the
    inputs that do not need an LLM, and the conservative "could be media" guess
    blocked every file upload. A media file that really does need a key fails in
    its loader instead — see test_media_loaders_require_llm.py.
    """
    assert _add_pipeline_needs_llm(data, preferred_loaders) is False


def test_keyless_file_upload_does_not_require_the_llm(tmp_path, no_llm):
    """The reported bug: a .txt upload 422'd before ingestion on a keyless setup."""
    document = tmp_path / "Natural_language_processing.txt"
    document.write_text("Natural language processing is a subfield of computer science.")

    assert _add_pipeline_needs_llm(str(document), preferred_loaders=None) is False


def test_keyless_media_upload_also_skips_the_check(tmp_path, no_llm):
    """Even media: the probe cannot report anything the loader will not report better."""
    media_path = tmp_path / "image.png"
    media_path.write_bytes(b"not-an-image")

    assert _add_pipeline_needs_llm(str(media_path), preferred_loaders=None) is False


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
