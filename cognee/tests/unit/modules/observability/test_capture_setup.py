"""The setup that produced a run's events (SDK-529): the allowlisted, secret-free
snapshot every manifest carries, its fingerprint, and the ``setup.changed`` event a
run emits when it starts on a different setup than the previous run of its kind."""

import importlib
import json
from uuid import uuid4

import pytest

from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_RUN_MANIFEST,
    KIND_SETUP_CHANGED,
    SETUP_FIELDS,
    capture_setup,
    setup_diff,
    setup_fingerprint,
)

pytestmark = pytest.mark.usefixtures("capture_reset")

settings_module = importlib.import_module("cognee.modules.settings.get_current_settings")

SETUP_KEYS = {key for key, _path in SETUP_FIELDS}


def _settings(llm_model="openai/gpt-test", graph_extractor="unknown", **overrides):
    """A get_current_settings() payload with the fields telemetry sanitizes but a
    file on disk must never carry: database URLs with embedded credentials."""
    payload = {
        "llm": {
            "provider": "openai",
            "model": llm_model,
            "configured": True,
            "structured_output": "litellm_native",
            "instructor_mode": "default",
        },
        "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
        "graph_extractor": graph_extractor,
        "graph": {"provider": "neo4j", "url": "bolt://neo4j:n3o4j-s3cret@graph.internal:7687"},
        "vector": {
            "provider": "pgvector",
            "url": "postgresql://cognee:p4ssw0rd@db.internal/cognee",
        },
        "relational": {"provider": "postgres", "url": "db.internal:5432"},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def fake_settings(monkeypatch):
    """Route capture_setup()'s lazy import of get_current_settings to a controllable fake."""
    state = {"payload": _settings(), "calls": []}

    def fake(*, graph_extractor=None, **_kwargs):
        state["calls"].append(graph_extractor)
        payload = dict(state["payload"])
        if graph_extractor is not None:
            payload["graph_extractor"] = graph_extractor
        return payload

    monkeypatch.setattr(settings_module, "get_current_settings", fake)
    return state


def _manifests(sink):
    return [record for record in sink.records if record["kind"] == KIND_RUN_MANIFEST]


def _setup_changes(sink):
    return [record for record in sink.records if record["kind"] == KIND_SETUP_CHANGED]


# ---------------------------------------------------------------------------
# The snapshot: allowlisted, scalar, secret-free
# ---------------------------------------------------------------------------


def test_snapshot_keeps_only_the_allowlisted_fields_and_never_a_url(fake_settings):
    setup = capture_setup(graph_extractor="llm")

    assert set(setup) == SETUP_KEYS
    assert setup["llm.model"] == "openai/gpt-test"
    assert setup["embedding.model"] == "BAAI/bge-small-en-v1.5"
    assert setup["graph_extractor"] == "llm"
    assert setup["graph.provider"] == "neo4j"
    assert setup["vector.provider"] == "pgvector"
    # The URLs — and the credentials inside them — must not reach a file on disk.
    serialized = json.dumps(setup)
    assert "url" not in serialized
    assert "n3o4j-s3cret" not in serialized
    assert "p4ssw0rd" not in serialized
    assert "db.internal" not in serialized


def test_snapshot_reports_no_extractor_for_a_run_that_extracts_no_graph(fake_settings):
    """A search does not resolve an extractor; telemetry says "unknown" there, the
    snapshot says None, so a search and a cognify on one configuration differ in
    exactly that key (and never trip each other's setup.changed, see below)."""
    assert capture_setup()["graph_extractor"] is None
    assert capture_setup(graph_extractor="gliner_demo")["graph_extractor"] == "gliner_demo"


def test_snapshot_against_the_real_settings_is_scalar_and_complete():
    setup = capture_setup(graph_extractor="llm")

    assert setup is not None
    assert set(setup) == SETUP_KEYS
    for key, value in setup.items():
        assert value is None or isinstance(value, (str, int, float, bool)), (key, value)
    assert setup["llm.provider"]
    assert setup["embedding.provider"]
    assert setup["graph_extractor"] == "llm"


def test_a_snapshot_that_cannot_be_taken_is_none_and_breaks_nothing(monkeypatch, fake_capture_sink):
    def explode(**_kwargs):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(settings_module, "get_current_settings", explode)

    assert capture_setup() is None
    assert setup_fingerprint(None) is None
    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        assert scope.setup is None
        assert scope.setup_fingerprint is None


# ---------------------------------------------------------------------------
# The fingerprint and the diff
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_order_insensitive_and_sensitive_to_the_model():
    base = {"llm.model": "a", "embedding.model": "e", "graph_extractor": "llm"}
    reordered = {"graph_extractor": "llm", "embedding.model": "e", "llm.model": "a"}
    other_model = dict(base, **{"llm.model": "b"})

    assert setup_fingerprint(base) == setup_fingerprint(base) == setup_fingerprint(reordered)
    assert setup_fingerprint(base) != setup_fingerprint(other_model)
    assert setup_fingerprint(base).startswith("sha256:")
    assert len(setup_fingerprint(base)) == len("sha256:") + 16


def test_diff_lists_only_the_keys_that_changed():
    previous = {"llm.model": "a", "embedding.model": "e", "graph_extractor": "llm"}
    current = {"llm.model": "b", "embedding.model": "e", "vector.provider": "lancedb"}

    assert setup_diff(previous, current) == {
        "graph_extractor": {"from": "llm", "to": None},
        "llm.model": {"from": "a", "to": "b"},
        "vector.provider": {"from": None, "to": "lancedb"},
    }
    assert setup_diff(previous, previous) == {}
    assert setup_diff(None, {"k": 1}) == {"k": {"from": None, "to": 1}}


# ---------------------------------------------------------------------------
# On the manifest, and the setup.changed event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_manifest_carries_the_setup_and_its_fingerprint(
    fake_settings, fake_capture_sink
):
    with capture.run_scope(uuid4(), uuid4(), kind="pipeline", graph_extractor="gliner_demo"):
        pass
    with capture.run_scope(uuid4(), kind="operation"):
        pass
    await capture.drain()

    pipeline, operation = (m["payload"] for m in _manifests(fake_capture_sink))
    assert pipeline["setup"]["graph_extractor"] == "gliner_demo"
    assert pipeline["setup"]["llm.model"] == "openai/gpt-test"
    assert pipeline["setup_fingerprint"] == setup_fingerprint(pipeline["setup"])
    assert operation["setup"]["graph_extractor"] is None
    assert operation["setup_fingerprint"] != pipeline["setup_fingerprint"]
    # The run's own snapshot was taken with its extractor, not a process-wide one.
    assert fake_settings["calls"] == ["gliner_demo", None]


@pytest.mark.asyncio
async def test_setup_changed_marks_a_model_change_between_runs_of_one_kind(
    fake_settings, fake_capture_sink
):
    """The first setup a process sees is the baseline (it is on that manifest);
    the event marks the run that first ran on a different one, with the diff."""
    first_run, same_run, changed_run = uuid4(), uuid4(), uuid4()

    with capture.run_scope(first_run, kind="pipeline", graph_extractor="llm"):
        pass
    with capture.run_scope(same_run, kind="pipeline", graph_extractor="llm"):
        pass
    await capture.drain()
    assert _setup_changes(fake_capture_sink) == []

    # The model changes — a local small model takes over from the hosted one.
    fake_settings["payload"] = _settings(llm_model="ollama/llama3.1:8b")
    fake_settings["payload"]["llm"]["provider"] = "ollama"
    with capture.run_scope(changed_run, kind="pipeline", graph_extractor="llm") as scope:
        pass
    await capture.drain()

    [event] = _setup_changes(fake_capture_sink)
    assert event["run_id"] == str(changed_run)  # filed under the run that ran on the new setup
    payload = event["payload"]
    assert payload["run_kind"] == "pipeline"
    assert payload["fingerprint"] == scope.setup_fingerprint
    assert payload["previous_fingerprint"] != payload["fingerprint"]
    assert payload["changed"] == {
        "llm.model": {"from": "openai/gpt-test", "to": "ollama/llama3.1:8b"},
        "llm.provider": {"from": "openai", "to": "ollama"},
    }
    assert payload["setup"] == scope.setup
    # Counted on the run it belongs to.
    manifests = {m["run_id"]: m["payload"] for m in _manifests(fake_capture_sink)}
    assert manifests[str(changed_run)]["events_emitted"] == 1
    assert manifests[str(first_run)]["events_emitted"] == 0


@pytest.mark.asyncio
async def test_setup_changed_tracks_pipelines_and_operations_separately(
    fake_settings, fake_capture_sink
):
    """A search carries no graph_extractor, so against a shared baseline every
    cognify -> search -> cognify alternation would look like a change."""
    for _ in range(2):
        with capture.run_scope(uuid4(), kind="pipeline", graph_extractor="llm"):
            pass
        with capture.run_scope(uuid4(), kind="operation"):
            pass
    await capture.drain()

    assert _setup_changes(fake_capture_sink) == []
    assert len(_manifests(fake_capture_sink)) == 4


@pytest.mark.asyncio
async def test_the_extractor_change_alone_is_a_setup_change(fake_settings, fake_capture_sink):
    """Switching cognify from the LLM to the local GLiNER demo model, same LLM
    configured, is the change extraction evals most need to see."""
    with capture.run_scope(uuid4(), kind="pipeline", graph_extractor="llm"):
        pass
    with capture.run_scope(uuid4(), kind="pipeline", graph_extractor="gliner_demo"):
        pass
    await capture.drain()

    [event] = _setup_changes(fake_capture_sink)
    assert event["payload"]["changed"] == {"graph_extractor": {"from": "llm", "to": "gliner_demo"}}


def test_setup_changed_is_in_the_record_kinds_and_sampling_never_gates_it():
    assert KIND_SETUP_CHANGED == "setup.changed"
    assert capture.should_capture(KIND_SETUP_CHANGED) is True
