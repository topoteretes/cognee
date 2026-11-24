"""Isolates the unit suite from the developer's own cognee setup.

Almost every env patch in this suite is ``clear=False`` — deliberately, because
each test declares only the handful of variables it cares about. That leaves the
rest of ``os.environ`` showing through, and the plugin reads two dozen
``COGNEE_*`` variables to decide its transport, its key, and its roots. On a CI
runner nothing is set and the suite is green; on the machine of anyone who
actually *runs* cognee it is not. A single exported variable is enough:

    COGNEE_API_KEY=...   -> 7 failures (TestApiKeyResolution: a configured key
                            skips the minting path those tests exercise)
    COGNEE_EMBEDDED=true -> 3 failures (TestTransportSelection: embedded mode
    COGNEE_TRANSPORT=sdk -> 3 failures  outranks the HTTP default under test)

So the baseline is scrubbed here instead of at two dozen call sites. Tests that
want a variable still set it themselves; what they no longer inherit is the
developer's shell. See SDK-454.
"""

import os
import sys
from pathlib import Path

import pytest

# The plugin's whole configuration surface: COGNEE_* (transport, key, roots,
# timeouts), LLM_* (model selection, and the creds a spawned server inherits),
# and HERMES_HOME (the profile whose cognee.json layers under the env).
_SCRUBBED_PREFIXES = ("COGNEE_", "LLM_", "HERMES_")

# The opt-in live modules are the exception: they talk to a real server on
# purpose, and ``server_bootstrap._spawn`` hands it ``dict(os.environ)`` — strip
# LLM_API_KEY here and the server they boot comes up without credentials. Their
# skip gates read the environment at import time, so scrubbing could not have
# unselected them anyway; this keeps the run they opted into intact.
_LIVE_MODULES = frozenset(
    {
        "test_integration_concurrency",
        "test_integration_roundtrip",
    }
)

# The test modules import their shared helpers as ``from _char_helpers import
# ...``, which resolves only because pytest prepends a test file's own directory
# to sys.path. Stating it here makes that dependency explicit rather than a
# property of the default import mode.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# The live modules budget in minutes, not seconds: a cold server boot is 120s, a
# graph write 600s, an improve 900s. The 60s ini timeout that bounds the hermetic
# suite would guillotine every one of them, so they get a ceiling above their own
# slowest wait instead — still a bound, just an honest one. A marker outranks the
# ini value, and setting it here keeps the whole hermetic/live split in one file.
_LIVE_TIMEOUT = 1800


def pytest_collection_modifyitems(items):
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in _LIVE_MODULES:
            item.add_marker(pytest.mark.timeout(_LIVE_TIMEOUT))


@pytest.fixture(autouse=True)
def hermetic_env(request, monkeypatch, tmp_path):
    """Drop every ambient cognee variable, for every test but the live ones.

    The shared state files under ``~/.cognee-plugin/hermes/`` are re-pointed at
    the test's tmp dir for the same reason the env is scrubbed: the dataset
    overrides a developer switched, the repos they indexed for code recall,
    their cached update check, their buffered capture writes and their cognee
    runtime venv would otherwise leak into (and be written by) the hermetic
    suite.
    """
    if request.module.__name__.rsplit(".", 1)[-1] in _LIVE_MODULES:
        return
    for name in list(os.environ):
        if name.startswith(_SCRUBBED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    # Memory credit (on in real use) adds tags and an instruction to recall
    # output; the suite's format tests assert that output exactly, so it is off
    # here and test_memory_credit.py turns it on.
    monkeypatch.setenv("COGNEE_PLUGIN_CITE_MEMORY", "off")

    from cognee_integration_hermes import (
        code_graph,
        dataset_overrides,
        runtime_env,
        update_check,
        write_buffer,
    )

    monkeypatch.setattr(code_graph, "_STATE_DIR", tmp_path / "code-graph")
    monkeypatch.setattr(dataset_overrides, "_OVERRIDES_PATH", tmp_path / "dataset-overrides.json")
    monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "update-check.json")
    monkeypatch.setattr(write_buffer, "WRITE_BUFFER_PATH", tmp_path / "pending-entries.json")
    # A developer's own runtime venv, ready or outdated, must not decide whether
    # a test's server spawn finds a runtime.
    monkeypatch.setattr(runtime_env, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(runtime_env, "VENV_DIR", tmp_path / "runtime" / "venv")
    monkeypatch.setattr(runtime_env, "READY_MARKER", tmp_path / "runtime" / "ready.json")
    # Nor may a developer's running cognee server: install-runtime would stop it
    # (or refuse to install while it answers), and its /health version would
    # decide whether a test's server is current.
    monkeypatch.setattr(runtime_env, "plugin_servers", lambda interpreters=None: [])
    monkeypatch.setattr(runtime_env, "_server_answering", lambda port: False)
    # The once-per-dataset recall note: its markers, and one waiting from another test.
    from cognee_integration_hermes import recall_notice

    monkeypatch.setattr(recall_notice, "NOTICE_DIR", tmp_path / "notices")
    monkeypatch.setattr(recall_notice, "_pending", {})
    from cognee_integration_hermes import server_bootstrap

    monkeypatch.setattr(server_bootstrap, "health_version", lambda url, timeout=2.0: "")


@pytest.fixture(autouse=True)
def fresh_process_state(monkeypatch):
    """Per-process state starts empty in every test, as in a new Hermes process.

    The keys a local server handed out (http_backend._RESOLVED_KEYS) and the
    one-time "no LLM key" warning would otherwise carry from one test to the next.
    """
    from cognee_integration_hermes import http_backend, server_bootstrap

    monkeypatch.setattr(http_backend, "_RESOLVED_KEYS", {})
    monkeypatch.setattr(server_bootstrap, "_warned_no_llm_key", False)
