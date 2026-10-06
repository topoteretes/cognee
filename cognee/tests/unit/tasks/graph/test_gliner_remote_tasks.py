"""Remote GLiNER wired into the cognify tasks (SDK-980): settings, task routing, no install.

Deterministic: the worker is a fake adapter; gliner2 is never imported.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.cognify import config as cognify_config_module
from cognee.modules.data.processing.document_types import TextDocument
from cognee.tasks.graph.gliner_demo import (
    GlinerNotInstalledError,
    GlinerOptions,
    GlinerRunStats,
    GlinerSchema,
    extract_graph_and_summarize_with_gliner,
    get_gliner_demo_tasks,
    install,
)
from cognee.tasks.graph.gliner_demo import tasks as tasks_module
from cognee.tasks.graph.gliner_demo.remote import (
    GlinerRemoteConfigError,
    RemoteGlinerAdapter,
    create_remote_adapter,
    get_remote_gliner_settings,
    remote_gliner_configured,
)
from cognee.tasks.graph.gliner_demo.schema import LABEL_BANK_PROBE_SCHEMA

APPLE = {
    "entities": {
        "person": [{"text": "Tim Cook", "confidence": 0.9, "start": 0, "end": 8}],
        "organization": [{"text": "Apple Inc.", "confidence": 0.9, "start": 14, "end": 24}],
    },
    "relation_extraction": {
        "works_for": [
            {
                "head": {"text": "Tim Cook", "start": 0, "end": 8},
                "tail": {"text": "Apple Inc.", "start": 14, "end": 24},
            }
        ]
    },
}


@pytest.fixture
def remote_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("COGNEE_GLINER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("COGNEE_GLINER_TRANSPORT", "http")
    monkeypatch.setenv("COGNEE_GLINER_ENDPOINT", "http://worker:8080")


@pytest.fixture
def local_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("COGNEE_GLINER_"):
            monkeypatch.delenv(name)


class FakeRemote:
    """Stands in for RemoteGlinerAdapter at the task boundary."""

    model = "fastino/gliner2.5-base-v1"

    def __init__(self, result=APPLE, probe=None):
        self.result = result
        self.probe = probe or {"entities": {"person": ["Tim Cook"], "organization": ["Apple"]}}
        self.batches: list[tuple] = []
        self.probes: list[tuple] = []

    async def extract_batch(self, texts, schema, **options):
        self.batches.append((list(texts), schema, options))
        return [self.result for _ in texts]

    async def extract_once(self, text, schema, *, threshold):
        self.probes.append((text, schema, threshold))
        return self.probe if text else {}


def _chunk(text="Tim Cook runs Apple Inc."):
    document = TextDocument(
        name="doc.txt", raw_data_location="/tmp/doc.txt", external_metadata=None
    )
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=document,
        contains=[],
    )


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_defaults_are_local(local_env):
    settings = get_remote_gliner_settings()
    assert settings.transport == "local" and not settings.is_remote
    assert not remote_gliner_configured()
    assert create_remote_adapter() is None
    assert settings.max_concurrent_requests == 16
    assert settings.connect_timeout_secs == 5.0
    assert settings.request_timeout_secs == 300.0
    assert settings.amqp_queue == "gliner_worker.extract"


def test_environment_selects_a_remote_worker(remote_env, monkeypatch):
    monkeypatch.setenv("COGNEE_GLINER_API_KEY", " s3cret ")
    monkeypatch.setenv("COGNEE_GLINER_EXPECTED_MODEL", "  ")
    settings = get_remote_gliner_settings()
    assert settings.is_remote and remote_gliner_configured()
    assert settings.endpoint == "http://worker:8080"
    assert settings.expected_model is None
    assert "s3cret" not in repr(settings)
    assert isinstance(create_remote_adapter(), RemoteGlinerAdapter)


@pytest.mark.parametrize(
    ("value", "expected"), [("RabbitMQ", "amqp"), ("GRPC", "grpc"), ("", "local")]
)
def test_transport_names_are_normalized(local_env, monkeypatch, value, expected):
    monkeypatch.setenv("COGNEE_GLINER_TRANSPORT", value)
    assert get_remote_gliner_settings().transport == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("COGNEE_GLINER_TRANSPORT", "carrier-pigeon"),
        ("COGNEE_GLINER_REQUEST_TIMEOUT_SECS", "soon"),
        ("COGNEE_GLINER_MAX_CONCURRENT_REQUESTS", "0"),
        ("COGNEE_GLINER_CONNECT_TIMEOUT_SECS", "0"),
    ],
)
def test_malformed_settings_name_the_variable(local_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(GlinerRemoteConfigError, match=name):
        get_remote_gliner_settings()


def test_a_remote_transport_without_an_endpoint_is_refused(local_env, monkeypatch):
    monkeypatch.setenv("COGNEE_GLINER_TRANSPORT", "grpc")
    with pytest.raises(GlinerRemoteConfigError, match="COGNEE_GLINER_ENDPOINT"):
        create_remote_adapter()


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extraction_goes_to_the_worker_and_never_loads_a_model():
    remote = FakeRemote()
    chunk = _chunk()
    schema = GlinerSchema({"person": "", "organization": ""}, {"works_for": ""}, "caller")
    chunk.is_part_of._gliner_schema = schema
    stats = GlinerRunStats()
    options = GlinerOptions(threshold=0.4, batch_size=8, window_words=200, window_overlap_words=20)

    with (
        patch.object(tasks_module, "get_extractor", side_effect=AssertionError("local model")),
        patch.object(tasks_module, "extract_graph_from_data", AsyncMock()),
    ):
        summaries = await extract_graph_and_summarize_with_gliner(
            [chunk], stats=stats, options=options, remote=remote
        )

    [(texts, sent_schema, sent_options)] = remote.batches
    assert texts == ["Tim Cook runs Apple Inc."] and sent_schema is schema
    assert sent_options == {
        "threshold": 0.4,
        "batch_size": 8,
        "window_words": 200,
        "window_overlap_words": 20,
    }
    assert stats.model == "fastino/gliner2.5-base-v1"
    assert (stats.nodes, stats.kept_edges) == (2, 1)
    assert "Tim Cook works_for Apple Inc." in summaries[0].text


@pytest.mark.asyncio
async def test_label_bank_probe_runs_on_the_worker():
    remote = FakeRemote()
    document = TextDocument(name="people.txt", raw_data_location="p.txt", external_metadata=None)

    async def read(document, **_kwargs):
        yield SimpleNamespace(text="Tim Cook runs Apple.")

    with (
        patch.object(TextDocument, "read", read),
        patch.object(tasks_module, "get_extractor", side_effect=AssertionError("local model")),
    ):
        await tasks_module.prepare_gliner_schema(
            [document], schema=GlinerSchema(), max_chunk_size=512, threshold=0.3, remote=remote
        )

    [(text, schema, threshold)] = remote.probes
    assert text == "Tim Cook runs Apple." and threshold == 0.3
    assert schema is LABEL_BANK_PROBE_SCHEMA
    assert set(document._gliner_schema.entity_types) == {"person", "organization"}
    assert document._gliner_schema.source == "label_bank"


@pytest.mark.asyncio
async def test_remote_tasks_need_no_gliner2_and_check_the_worker_first(remote_env):
    with (
        patch.object(tasks_module, "require_gliner2", side_effect=GlinerNotInstalledError()),
        patch.object(tasks_module, "resolve_chunk_size", AsyncMock(return_value=512)),
        patch.object(RemoteGlinerAdapter, "ensure_ready", AsyncMock()) as ensure_ready,
    ):
        tasks = await get_gliner_demo_tasks(["person"])

    ensure_ready.assert_awaited_once()
    schema_remote = tasks[1].default_params["kwargs"]["remote"]
    extraction_remote = tasks[3].default_params["kwargs"]["remote"]
    assert isinstance(schema_remote, RemoteGlinerAdapter)
    # One adapter per run: both tasks share its latched failures and model.
    assert schema_remote is extraction_remote


@pytest.mark.asyncio
async def test_an_unready_worker_fails_before_any_document_is_read(remote_env):
    from cognee.tasks.graph.gliner_demo.remote import GlinerWorkerUnavailableError

    with (
        patch.object(
            RemoteGlinerAdapter,
            "ensure_ready",
            AsyncMock(side_effect=GlinerWorkerUnavailableError("down")),
        ),
        pytest.raises(GlinerWorkerUnavailableError),
    ):
        await get_gliner_demo_tasks(["person"], chunk_size=512)


@pytest.mark.asyncio
async def test_local_tasks_still_require_gliner2(local_env):
    with (
        patch.object(tasks_module, "require_gliner2", side_effect=GlinerNotInstalledError()),
        pytest.raises(GlinerNotInstalledError),
    ):
        await get_gliner_demo_tasks(["person"], chunk_size=512)


@pytest.mark.asyncio
async def test_a_remote_worker_skips_the_local_runtime_install(remote_env):
    config = cognify_config_module.CognifyConfig()
    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime") as run,
    ):
        await cognify_config_module.ensure_extractor_runtime("gliner_demo", config)
    run.assert_not_called()
