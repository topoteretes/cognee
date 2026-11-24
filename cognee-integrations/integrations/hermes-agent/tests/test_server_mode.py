"""Tests for local-server mode: bootstrap helper, init mode routing, config.

Mode selection is a Hermes concern and stays in the provider: which transport
gets built, and whether a local server is spawned first. The transport itself is
faked, so these exercise pure routing logic and need neither cognee nor a network.
How a transport turns protocol calls into wire format lives in
``test_sdk_backend.py``.

Runs under pytest or standalone (``python3 tests/test_server_mode.py``).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import FakeBackend  # noqa: E402
from cognee_integration_hermes import config as config_mod  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402
from cognee_integration_hermes import server_bootstrap as sb  # noqa: E402
from cognee_integration_hermes.backend import SdkBackend  # noqa: E402
from cognee_integration_hermes.http_backend import HttpBackend  # noqa: E402


class _FakeResp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestHealthOk(unittest.TestCase):
    def test_2xx_is_healthy(self):
        with mock.patch.object(sb.urllib.request, "urlopen", return_value=_FakeResp(200)):
            self.assertTrue(sb.health_ok("http://127.0.0.1:8000"))

    def test_5xx_is_not_healthy(self):
        with mock.patch.object(sb.urllib.request, "urlopen", return_value=_FakeResp(503)):
            self.assertFalse(sb.health_ok("http://127.0.0.1:8000"))

    def test_connection_error_is_not_healthy(self):
        with mock.patch.object(sb.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertFalse(sb.health_ok("http://127.0.0.1:8000"))


def _live_child():
    """A spawned process that is still running (poll() -> None)."""
    proc = mock.MagicMock()
    proc.poll.return_value = None
    return proc


def _dead_child(returncode=3):
    proc = mock.MagicMock()
    proc.poll.return_value = returncode
    proc.returncode = returncode
    return proc


class TestEnsureLocalServer(unittest.TestCase):
    def setUp(self):
        # _spawn is faked throughout, so no interpreter with cognee is needed; the
        # no-runtime path is covered in test_local_mode_requirement.py.
        patcher = mock.patch.object(sb, "server_python", return_value=sys.executable)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_already_healthy_does_not_spawn(self):
        with (
            mock.patch.object(sb, "health_ok", return_value=True),
            mock.patch.object(sb, "_spawn") as spawn,
        ):
            url = sb.ensure_local_server(8000)
        self.assertEqual(url, "http://127.0.0.1:8000")
        spawn.assert_not_called()

    def test_spawns_then_polls_until_healthy(self):
        # First probe down (-> spawn), second probe up (-> return).
        with (
            mock.patch.object(sb, "health_ok", side_effect=[False, True]),
            mock.patch.object(sb, "_spawn", return_value=_live_child()) as spawn,
            mock.patch.object(sb.time, "sleep"),
        ):
            url = sb.ensure_local_server(8123)
        self.assertEqual(url, "http://127.0.0.1:8123")
        spawn.assert_called_once()

    def test_raises_when_never_healthy(self):
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "_spawn", return_value=_live_child()),
            mock.patch.object(sb.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError):
                sb.ensure_local_server(8000, boot_timeout=0.01)

    def test_a_dead_spawn_with_a_silent_port_fails_fast(self):
        # The 600s deadline exists for slow first boots, not broken installs — a
        # crashed child with no other owner on the port must not stall the
        # session start for 10 minutes.
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "_spawn", return_value=_dead_child(3)),
            mock.patch.object(sb, "port_bound", return_value=False),
            mock.patch.object(sb.time, "sleep") as slept,
        ):
            with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
                sb.ensure_local_server(8000, boot_timeout=600.0)
        slept.assert_not_called()

    def test_a_dead_spawn_that_lost_the_bind_race_keeps_waiting(self):
        # Our child dying because a concurrent starter won the port is a success
        # path: the winner owns the port and will become healthy.
        with (
            mock.patch.object(sb, "health_ok", side_effect=[False, False, True]),
            mock.patch.object(sb, "_spawn", return_value=_dead_child(1)),
            mock.patch.object(sb, "port_bound", return_value=True),
            mock.patch.object(sb.time, "sleep"),
        ):
            url = sb.ensure_local_server(8123, boot_timeout=600.0)
        self.assertEqual(url, "http://127.0.0.1:8123")

    def test_a_failed_spawn_with_a_silent_port_fails_fast(self):
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "_spawn", side_effect=OSError("permission denied")),
            mock.patch.object(sb, "port_bound", return_value=False),
            mock.patch.object(sb.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to launch"):
                sb.ensure_local_server(8000, boot_timeout=600.0)

    def test_the_error_points_at_the_server_log(self):
        with (
            mock.patch.object(sb, "health_ok", return_value=False),
            mock.patch.object(sb, "_spawn", return_value=_dead_child()),
            mock.patch.object(sb, "port_bound", return_value=False),
            mock.patch.object(sb.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "/var/log/cognee.log"):
                sb.ensure_local_server(8000, boot_timeout=600.0, log_path="/var/log/cognee.log")


class TestServerLogRotation(unittest.TestCase):
    """server.log is capped by rotation at spawn — it cannot be silenced (the
    overflow scan reads it), but it must not grow without bound either."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "server.log"

    def test_small_log_is_left_alone(self):
        self.log.write_bytes(b"x" * 100)
        self.assertFalse(sb.rotate_oversized_log(str(self.log), max_bytes=1000))
        self.assertTrue(self.log.exists())
        self.assertFalse((self.log.parent / "server.log.1").exists())

    def test_oversized_log_rotates_to_one_kept_generation(self):
        self.log.write_bytes(b"old" * 400)
        self.assertTrue(sb.rotate_oversized_log(str(self.log), max_bytes=1000))
        self.assertFalse(self.log.exists())
        self.assertEqual((self.log.parent / "server.log.1").read_bytes(), b"old" * 400)
        # A second rotation replaces the previous generation rather than stacking.
        self.log.write_bytes(b"new" * 400)
        self.assertTrue(sb.rotate_oversized_log(str(self.log), max_bytes=1000))
        self.assertEqual((self.log.parent / "server.log.1").read_bytes(), b"new" * 400)

    def test_missing_log_and_zero_cap_are_noops(self):
        self.assertFalse(sb.rotate_oversized_log(str(self.log), max_bytes=1000))
        self.log.write_bytes(b"x" * 5000)
        self.assertFalse(sb.rotate_oversized_log(str(self.log), max_bytes=0))
        self.assertTrue(self.log.exists())

    def test_cap_comes_from_env(self):
        with mock.patch.dict("os.environ", {sb.SERVER_LOG_MAX_BYTES_ENV: "1000"}):
            self.assertEqual(sb._server_log_cap(), 1000)
        with mock.patch.dict("os.environ", {sb.SERVER_LOG_MAX_BYTES_ENV: "nonsense"}):
            self.assertEqual(sb._server_log_cap(), sb.SERVER_LOG_MAX_BYTES)

    def test_spawn_rotates_before_handing_the_log_to_the_child(self):
        self.log.write_bytes(b"x" * 5000)
        with (
            mock.patch.dict("os.environ", {sb.SERVER_LOG_MAX_BYTES_ENV: "1000"}),
            mock.patch.object(sb.subprocess, "Popen", return_value=mock.MagicMock()),
        ):
            sb._spawn(8000, None, None, str(self.log))
        self.assertTrue((self.log.parent / "server.log.1").exists())
        self.assertLess(self.log.stat().st_size, 1000)  # fresh file for the new server


class TestSpawnEnvironment(unittest.TestCase):
    """The spawned server's environment — where the session cache lives or dies."""

    def _spawn_env(self, **env_overrides):
        captured = {}

        def fake_popen(args, env=None, **kwargs):
            captured.update(env or {})
            return mock.MagicMock()

        with (
            mock.patch.dict("os.environ", env_overrides, clear=False),
            mock.patch.object(sb.subprocess, "Popen", side_effect=fake_popen),
        ):
            sb._spawn(9999, "", "", "/dev/null")
        return captured

    def test_caching_is_enabled_for_the_session_tier(self):
        # Live-diagnosed: without CACHING, cognee's session manager reports
        # is_available=False and a session write is dropped while the API still
        # answers status="session_stored" — so turns silently never reach the graph.
        self.assertEqual(self._spawn_env()["CACHING"], "true")

    def test_auto_feedback_is_enabled(self):
        self.assertEqual(self._spawn_env()["AUTO_FEEDBACK"], "true")

    def test_agent_mode_is_enabled_so_the_server_can_retire(self):
        self.assertEqual(self._spawn_env()["COGNEE_AGENT_MODE"], "true")

    def test_the_env_matches_the_other_plugins_bootstraps(self):
        # The server must behave identically no matter which cognee plugin booted
        # it — these mirror claude-code's apply_cognee_env.
        env = self._spawn_env()
        self.assertEqual(env["CACHE_ROOT_DIRECTORY"], str(config_mod.SHARED_COGNEE_HOME / "cache"))
        self.assertEqual(env["LLM_INSTRUCTOR_MODE"], "json_schema_mode")
        self.assertEqual(env["COGNEE_IMPROVE_SUBMIT_TIMEOUT"], "420")

    def test_an_explicit_user_value_wins(self):
        self.assertEqual(self._spawn_env(CACHING="false")["CACHING"], "false")

    def test_data_roots_are_passed_through_when_given(self):
        captured = {}

        def fake_popen(args, env=None, **kwargs):
            captured.update(env or {})
            return mock.MagicMock()

        with mock.patch.object(sb.subprocess, "Popen", side_effect=fake_popen):
            sb._spawn(9999, "/tmp/data", "/tmp/system", "/dev/null")
        self.assertEqual(captured["DATA_ROOT_DIRECTORY"], "/tmp/data")
        self.assertEqual(captured["SYSTEM_ROOT_DIRECTORY"], "/tmp/system")

    def test_roots_are_omitted_when_empty(self):
        env = self._spawn_env()
        # Not asserting absence outright: the ambient environment may legitimately
        # carry them. What matters is we do not invent an empty value.
        self.assertNotEqual(env.get("DATA_ROOT_DIRECTORY", "unset"), "")


class TestOllamaEmbeddingPins(unittest.TestCase):
    """The spawn env caps a local Ollama embedder at its real context length.

    Without these, cognee sizes chunks from EMBEDDING_MAX_COMPLETION_TOKENS'
    8191 default, every substantial document overflows the model, and cognee
    mean-pools the pieces into lossy vectors while reporting success.

    ``clear=True``: pins are ``setdefault``s, so ambient EMBEDDING_* values in
    the developer's shell would otherwise leak in and flip the assertions.
    """

    def _spawn_env(self, **env_overrides):
        captured = {}

        def fake_popen(args, env=None, **kwargs):
            captured.update(env or {})
            return mock.MagicMock()

        with (
            mock.patch.dict("os.environ", env_overrides, clear=True),
            mock.patch.object(sb.subprocess, "Popen", side_effect=fake_popen),
        ):
            sb._spawn(9999, "", "", "/dev/null")
        return captured

    def test_a_known_model_gets_its_context_and_tokenizer(self):
        env = self._spawn_env(EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="all-minilm")
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "256")
        self.assertEqual(env["HUGGINGFACE_TOKENIZER"], "sentence-transformers/all-MiniLM-L6-v2")

    def test_model_names_are_normalized(self):
        env = self._spawn_env(
            EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="ollama/nomic-embed-text:latest"
        )
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "2048")

    def test_registry_namespaces_still_match(self):
        # cognee's own Ollama default model ships under the avr/ namespace.
        env = self._spawn_env(
            EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="avr/sfr-embedding-mistral:latest"
        )
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "2048")
        self.assertEqual(env["HUGGINGFACE_TOKENIZER"], "Salesforce/SFR-Embedding-Mistral")

    def test_the_longest_table_key_wins(self):
        # snowflake-arctic-embed2 must not resolve to snowflake-arctic-embed's 512.
        env = self._spawn_env(
            EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="snowflake-arctic-embed2"
        )
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "8192")

    def test_an_unknown_model_gets_the_conservative_ceiling_and_no_tokenizer_guess(self):
        env = self._spawn_env(EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="mystery-embed")
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "512")
        # A wrong tokenizer miscounts tokens and can reintroduce the overflow.
        self.assertNotIn("HUGGINGFACE_TOKENIZER", env)

    def test_no_pins_without_the_ollama_provider(self):
        for provider in ("", "openai", "openai_compatible"):
            env = self._spawn_env(EMBEDDING_PROVIDER=provider, EMBEDDING_MODEL="all-minilm")
            self.assertNotIn("EMBEDDING_MAX_COMPLETION_TOKENS", env)
            self.assertNotIn("HUGGINGFACE_TOKENIZER", env)

    def test_an_explicit_user_value_wins(self):
        env = self._spawn_env(
            EMBEDDING_PROVIDER="ollama",
            EMBEDDING_MODEL="all-minilm",
            EMBEDDING_MAX_COMPLETION_TOKENS="9000",
            HUGGINGFACE_TOKENIZER="my/tokenizer",
        )
        self.assertEqual(env["EMBEDDING_MAX_COMPLETION_TOKENS"], "9000")
        self.assertEqual(env["HUGGINGFACE_TOKENIZER"], "my/tokenizer")


class _Recorder(dict):
    """Reads the mode-routing decisions off a fake backend.

    Mode selection stays in the provider — which transport gets built, and
    whether a local server is spawned first, is a Hermes concern. What the
    provider then *asks* the transport to do is recorded here.
    """

    def __init__(self, backend):
        super().__init__(served=None, identity_called=False, roots_called=False)
        self._backend = backend

    def _refresh(self):
        connects = self._backend.kwargs_for("connect")
        if connects:
            self["served"] = (connects[0]["url"], connects[0]["api_key"])
        self["identity_called"] = bool(self._backend.kwargs_for("resolve_identity"))
        self["roots_called"] = bool(self._backend.kwargs_for("configure_local_roots"))

    def __getitem__(self, key):
        self._refresh()
        return super().__getitem__(key)


def _make_provider():
    """A provider on a fake transport, plus a recorder of what it was asked to do."""
    backend = FakeBackend()
    p = provider_mod.CogneeMemoryProvider(backend=backend)
    # A session that has stored something, so its close has something to improve.
    p._stored_since_improve = 1
    return p, _Recorder(backend)


_NO_URL = {"COGNEE_BASE_URL": "", "COGNEE_SERVICE_URL": ""}


class TestInitializeModes(unittest.TestCase):
    def test_remote_mode_serves_service_url_and_skips_local_identity(self):
        env = {**_NO_URL, "COGNEE_BASE_URL": "https://cloud.example/api", "COGNEE_API_KEY": "k"}
        p, rec = _make_provider()
        with mock.patch.dict("os.environ", env, clear=False):
            p.initialize("sid")
        self.assertTrue(p._remote_mode)
        self.assertEqual(rec["served"], ("https://cloud.example/api", "k"))
        self.assertFalse(rec["identity_called"])
        self.assertFalse(rec["roots_called"])

    def test_embedded_mode_configures_shared_roots_and_resolves_identity(self):
        env = {
            **_NO_URL,
            "COGNEE_EMBEDDED": "true",
            "COGNEE_DATA_ROOT": "",
            "COGNEE_SYSTEM_ROOT": "",
        }
        p, rec = _make_provider()
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict("os.environ", env, clear=False):
                p.initialize("sid", hermes_home=home)
            self.assertFalse(p._remote_mode)
            self.assertIsNone(rec["served"])
            self.assertTrue(rec["roots_called"])
            self.assertTrue(rec["identity_called"])
            roots = p._backend.only_call("configure_local_roots")
        # The store is the shared ~/.cognee — the same roots the other cognee
        # plugins pin — not scoped to the Hermes profile.
        self.assertEqual(roots["data_root"], str(config_mod.SHARED_COGNEE_HOME / "data"))
        self.assertEqual(roots["system_root"], str(config_mod.SHARED_COGNEE_HOME / "system"))

    def test_embedded_mode_without_hermes_home_still_configures_roots(self):
        env = {
            **_NO_URL,
            "COGNEE_EMBEDDED": "true",
            "COGNEE_DATA_ROOT": "",
            "COGNEE_SYSTEM_ROOT": "",
        }
        p, rec = _make_provider()
        with mock.patch.dict("os.environ", env, clear=False):
            p.initialize("sid")
        self.assertTrue(rec["roots_called"])

    def test_default_mode_ensures_local_server_and_serves_localhost(self):
        env = {**_NO_URL, "COGNEE_EMBEDDED": ""}
        p, rec = _make_provider()
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
            ) as ensure,
        ):
            p.initialize("sid")
        ensure.assert_called_once()
        # The port actually handed to the bootstrap, not just the mocked return:
        # 8011 keeps us off cognee's own default of 8000, so we never attach to a
        # server the user is running themselves.
        self.assertEqual(ensure.call_args.args[0], 8011)
        # The 600s deadline the other plugins use: a cold first boot runs
        # migrations, and giving up early turns memory off for the whole session.
        self.assertEqual(ensure.call_args.kwargs["boot_timeout"], 600.0)
        self.assertTrue(p._remote_mode)
        self.assertEqual(rec["served"], ("http://127.0.0.1:8000", ""))
        self.assertFalse(rec["identity_called"])

    def test_initialize_ensures_the_dataset(self):
        env = {**_NO_URL, "COGNEE_EMBEDDED": ""}
        p, rec = _make_provider()
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
            ),
        ):
            p.initialize("sid")
        self.assertEqual(p._backend.only_call("ensure_dataset")["dataset"], p._dataset)

    def test_a_failed_dataset_ensure_is_not_fatal(self):
        # A write creates the dataset implicitly anyway; failing here only costs
        # the recall-first case, never the session.
        env = {**_NO_URL, "COGNEE_EMBEDDED": ""}
        p, rec = _make_provider()
        p._backend.errors["ensure_dataset"] = RuntimeError("datasets route down")
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
            ),
        ):
            p.initialize("sid")
        self.assertTrue(p._initialized)

    def test_local_server_failure_raises_not_silently_embedded(self):
        # Falling back to embedded would reintroduce the DB-lock risk this PR
        # removes, so a server that won't start is a hard error.
        env = {**_NO_URL, "COGNEE_EMBEDDED": ""}
        p, rec = _make_provider()
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", side_effect=RuntimeError("no server")
            ),
        ):
            with self.assertRaises(RuntimeError):
                p.initialize("sid")
        self.assertFalse(rec["roots_called"])  # did NOT silently drop to embedded

    def test_remote_failure_raises_not_silently_local(self):
        # An explicit remote URL that fails must surface, not silently diverge to a
        # local graph (data divergence / masked config error).
        env = {**_NO_URL, "COGNEE_BASE_URL": "https://cloud.example/api"}
        p, rec = _make_provider()
        p._backend.errors["connect"] = RuntimeError("unreachable")
        with mock.patch.dict("os.environ", env, clear=False):
            with self.assertRaises(RuntimeError):
                p.initialize("sid")
        self.assertFalse(rec["roots_called"])


class TestInlineImproveFallback(unittest.TestCase):
    """When nothing can take the close, on_session_end improves here — and waits.

    The handoff covers every server mode, so reaching this path means either
    embedded (the work dies with this process) or a failed spawn (shutdown is
    about to unregister). Both make waiting the only way to keep the session.
    """

    def _run_session_end(self, *, remote_mode, env_override=None):
        p, _ = _make_provider()
        p._initialized = True
        p._writes_enabled = True
        p._improve_on_end = True
        p._remote_mode = remote_mode
        p._config = {"improve_timeout": 300, "improve_background": env_override or ""}
        p._is_breaker_open = lambda: False
        p.on_session_end([])
        return p._backend.only_call("improve")["background"]

    def test_an_unarmed_server_mode_improves_synchronously(self):
        self.assertFalse(self._run_session_end(remote_mode=True))

    def test_embedded_mode_runs_synchronously(self):
        self.assertFalse(self._run_session_end(remote_mode=False))

    def test_env_override_forces_background_in_embedded(self):
        self.assertTrue(self._run_session_end(remote_mode=False, env_override="true"))


class TestTransportSelection(unittest.TestCase):
    """Direct HTTP is the default; the SDK serves embedded mode and opt-out."""

    def _transport_for(self, env):
        # Real transport classes, so the type assertions mean something — but every
        # I/O-touching hook is stubbed: SdkBackend's cognee imports would cost
        # seconds, and connect() would reach for a socket.
        merged = {**_NO_URL, **env}
        provider = provider_mod.CogneeMemoryProvider()
        with (
            mock.patch.dict("os.environ", merged, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
            ),
            mock.patch.object(SdkBackend, "configure_models"),
            mock.patch.object(SdkBackend, "configure_local_roots"),
            mock.patch.object(SdkBackend, "resolve_identity"),
            mock.patch.object(SdkBackend, "connect"),
            mock.patch.object(HttpBackend, "connect"),
        ):
            provider.initialize("sid")
        return type(provider._backend).__name__

    def test_default_is_the_http_transport(self):
        self.assertEqual(self._transport_for({}), "HttpBackend")

    def test_cloud_mode_also_uses_http(self):
        self.assertEqual(
            self._transport_for({"COGNEE_BASE_URL": "https://cloud.example"}), "HttpBackend"
        )

    def test_the_sdk_can_be_selected_explicitly(self):
        self.assertEqual(self._transport_for({"COGNEE_TRANSPORT": "sdk"}), "SdkBackend")

    def test_embedded_mode_uses_the_sdk(self):
        # Embedded means "no server at all", which only the in-process SDK can do.
        self.assertEqual(self._transport_for({"COGNEE_EMBEDDED": "true"}), "SdkBackend")

    def test_embedded_mode_wins_over_the_http_default(self):
        self.assertEqual(
            self._transport_for({"COGNEE_EMBEDDED": "true", "COGNEE_TRANSPORT": "http"}),
            "SdkBackend",
        )

    def test_an_injected_transport_always_wins(self):
        backend = FakeBackend()
        provider = provider_mod.CogneeMemoryProvider(backend=backend)
        env = {**_NO_URL, "COGNEE_EMBEDDED": "true", "COGNEE_TRANSPORT": "http"}
        with mock.patch.dict("os.environ", env, clear=False):
            provider.initialize("sid")
        self.assertIs(provider._backend, backend)

    def test_the_http_transport_caches_its_key_in_the_shared_state_dir(self):
        # One principal key per machine, shared with the other cognee plugins —
        # whichever plugin mints first, the rest (including Hermes) reuse it.
        provider = provider_mod.CogneeMemoryProvider()
        with tempfile.TemporaryDirectory() as home:
            with (
                mock.patch.dict("os.environ", _NO_URL, clear=False),
                mock.patch.object(
                    provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
                ),
                mock.patch.object(HttpBackend, "connect"),
            ):
                provider.initialize("sid", hermes_home=home)
            self.assertIsInstance(provider._backend, HttpBackend)


class TestExitWatcherLifecycle(unittest.TestCase):
    """Crash insurance is armed only when a server outlives the process."""

    def _server_backend(self):
        backend = FakeBackend()
        backend.connection_info = lambda: {
            "url": "http://127.0.0.1:8011",
            "api_key": "k",
            "agent_session_name": "hermes",
        }
        return backend

    def _initialize(self, backend, arm=None, **init_kwargs):
        p = provider_mod.CogneeMemoryProvider(backend=backend)
        with (
            mock.patch.dict("os.environ", _NO_URL, clear=False),
            mock.patch.object(
                provider_mod, "ensure_local_server", return_value="http://127.0.0.1:8000"
            ),
            mock.patch.object(provider_mod.exit_watcher, "arm", arm or mock.MagicMock()) as armed,
        ):
            p.initialize("sid", **init_kwargs)
        return p, armed

    def test_no_watcher_without_a_surviving_server(self):
        # FakeBackend inherits connection_info() -> None (in-process semantics).
        p, armed = self._initialize(FakeBackend())
        armed.assert_not_called()
        self.assertIsNone(p._watcher_state_path)

    def test_armed_in_server_mode_with_the_connection_details(self):
        p, armed = self._initialize(self._server_backend())
        kwargs = armed.call_args.kwargs
        self.assertEqual(kwargs["url"], "http://127.0.0.1:8011")
        self.assertEqual(kwargs["api_key"], "k")
        self.assertEqual(kwargs["parent_pid"], os.getpid())
        self.assertEqual(kwargs["session_id"], p._session_cognee_id)
        self.assertEqual(kwargs["dataset"], p._dataset)
        self.assertTrue(kwargs["improve"])
        self.assertIsNotNone(p._watcher_state_path)

    def test_a_subagent_context_disables_the_improve_half(self):
        # A subagent must not write; its watcher may still unregister.
        p, armed = self._initialize(self._server_backend(), agent_context="subagent")
        self.assertFalse(armed.call_args.kwargs["improve"])

    def test_arm_failure_is_not_fatal(self):
        arm = mock.MagicMock(side_effect=OSError("no fork for you"))
        p, _ = self._initialize(self._server_backend(), arm=arm)
        self.assertTrue(p._initialized)
        self.assertIsNone(p._watcher_state_path)

    def _armed_provider(self):
        p, _ = _make_provider()
        p._initialized = True
        p._writes_enabled = True
        p._improve_on_end = True
        p._remote_mode = True
        p._config = {"improve_timeout": 300}
        p._watcher_state_path = Path("/tmp/watcher.json")
        return p

    def test_session_switch_repoints_the_watcher(self):
        p = self._armed_provider()
        with mock.patch.object(provider_mod.exit_watcher, "update") as update:
            p.on_session_switch("next-session")
        kwargs = update.call_args.kwargs
        self.assertEqual(kwargs["session_id"], p._session_cognee_id)
        self.assertTrue(kwargs["improve"])

    def test_session_end_hands_the_close_to_a_detached_worker(self):
        p = self._armed_provider()
        with mock.patch.object(
            provider_mod.exit_watcher, "finalize", return_value=True
        ) as finalize:
            p.on_session_end([])
        self.assertEqual(finalize.call_args.kwargs["session_id"], p._session_cognee_id)
        self.assertEqual(p._backend.kwargs_for("improve"), [])
        self.assertTrue(p._close_handed_off)
        # The state file now belongs to the worker, so it stays put.
        self.assertIsNotNone(p._watcher_state_path)

    def test_a_failed_handoff_still_closes_the_session_inline(self):
        p = self._armed_provider()
        with mock.patch.object(provider_mod.exit_watcher, "finalize", return_value=False):
            with mock.patch.object(provider_mod.exit_watcher, "update") as update:
                p.on_session_end([])
        self.assertIs(p._backend.only_call("improve")["background"], False)
        # Bridged here, so a later crash must not improve the same session again.
        self.assertIs(update.call_args.kwargs["improve"], False)
        self.assertFalse(p._close_handed_off)

    def test_a_failed_inline_improve_keeps_the_insurance(self):
        p = self._armed_provider()
        p._backend.errors["improve"] = RuntimeError("server hiccup")
        with mock.patch.object(provider_mod.exit_watcher, "finalize", return_value=False):
            with mock.patch.object(provider_mod.exit_watcher, "update") as update:
                p.on_session_end([])
        update.assert_not_called()

    def test_shutdown_disarms_before_closing_the_backend(self):
        p = self._armed_provider()
        order = []
        with (
            mock.patch.object(
                provider_mod.exit_watcher, "disarm", side_effect=lambda *a: order.append("disarm")
            ),
            mock.patch.object(p._backend, "close", side_effect=lambda **k: order.append("close")),
        ):
            p.shutdown()
        self.assertEqual(order, ["disarm", "close"])
        self.assertIsNone(p._watcher_state_path)

    def test_shutdown_after_a_handoff_neither_disarms_nor_unregisters(self):
        # Both would sabotage the worker: disarming deletes the state it is
        # closing from, and unregistering drops the agent count to zero while its
        # improve is still running, letting the watchdog retire the server.
        p = self._armed_provider()
        p._close_handed_off = True
        with mock.patch.object(provider_mod.exit_watcher, "disarm") as disarm:
            p.shutdown()
        disarm.assert_not_called()
        self.assertIs(p._backend.only_call("close")["unregister"], False)
        self.assertIsNotNone(p._watcher_state_path)

    def test_a_switch_after_a_handoff_leaves_the_workers_state_alone(self):
        p = self._armed_provider()
        p._close_handed_off = True
        with mock.patch.object(provider_mod.exit_watcher, "update") as update:
            p.on_session_switch("next-session")
        update.assert_not_called()


class TestConfigModes(unittest.TestCase):
    def test_base_url_preferred_over_service_url(self):
        env = {"COGNEE_BASE_URL": "https://canonical", "COGNEE_SERVICE_URL": "https://legacy"}
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = config_mod.load_config()
        self.assertEqual(cfg["service_url"], "https://canonical")

    def test_service_url_used_when_base_url_absent(self):
        env = {**_NO_URL, "COGNEE_SERVICE_URL": "https://legacy"}
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = config_mod.load_config()
        self.assertEqual(cfg["service_url"], "https://legacy")

    def test_embedded_and_port_defaults(self):
        env = {**_NO_URL, "COGNEE_EMBEDDED": "", "COGNEE_LOCAL_PORT": ""}
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = config_mod.load_config()
        self.assertFalse(cfg["embedded"])
        self.assertEqual(cfg["local_port"], 8011)

    def test_local_port_clamped(self):
        env = {**_NO_URL, "COGNEE_LOCAL_PORT": "999999"}
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = config_mod.load_config()
        self.assertEqual(cfg["local_port"], 65535)


class TestSpawnEnv(unittest.TestCase):
    """The env ``_spawn`` hands the server: default-user credentials.

    cognee >= 1.6.0 creates the default user only when DEFAULT_USER_PASSWORD is
    set, so the spawned server is told the plugin's literal defaults — never the
    COGNEE_USER_* login selection, since the server (and its once-set default
    password) is shared by every cognee plugin.
    """

    def _spawn_env(self, extra=None, drop=()):
        base = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("DEFAULT_USER_", "COGNEE_USER_")) and k not in drop
        }
        base.update(extra or {})
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log_path = str(Path(tmp.name) / "server.log")
        with (
            mock.patch.dict("os.environ", base, clear=True),
            mock.patch.object(sb.subprocess, "Popen") as popen,
        ):
            sb._spawn(8011, "", "", log_path)
        self.assertEqual(popen.call_count, 1)
        return popen.call_args.kwargs["env"]

    def test_defaults_match_the_login_the_transport_performs(self):
        from cognee_integration_hermes.http_backend import (
            DEFAULT_USER_EMAIL,
            DEFAULT_USER_PASSWORD,
        )

        env = self._spawn_env()
        self.assertEqual(env["DEFAULT_USER_EMAIL"], DEFAULT_USER_EMAIL)
        self.assertEqual(env["DEFAULT_USER_PASSWORD"], DEFAULT_USER_PASSWORD)
        self.assertEqual(env["DEFAULT_USER_EMAIL"], "default_user@example.com")
        self.assertEqual(env["DEFAULT_USER_PASSWORD"], "default_password")

    def test_cognee_user_login_selection_does_not_redefine_the_default_user(self):
        env = self._spawn_env(
            {"COGNEE_USER_EMAIL": "me@example.org", "COGNEE_USER_PASSWORD": "s3cret"}
        )
        self.assertEqual(env["DEFAULT_USER_EMAIL"], "default_user@example.com")
        self.assertEqual(env["DEFAULT_USER_PASSWORD"], "default_password")

    def test_an_explicit_default_user_export_wins(self):
        env = self._spawn_env(
            {
                "DEFAULT_USER_EMAIL": "ops@example.org",
                "DEFAULT_USER_PASSWORD": "operator-set",
                "COGNEE_USER_PASSWORD": "ignored-for-the-server",
            }
        )
        self.assertEqual(env["DEFAULT_USER_EMAIL"], "ops@example.org")
        self.assertEqual(env["DEFAULT_USER_PASSWORD"], "operator-set")

    def test_the_other_server_flags_are_still_set(self):
        env = self._spawn_env()
        self.assertEqual(env["CACHING"], "true")
        self.assertEqual(env["HTTP_API_PORT"], "8011")
        self.assertEqual(env["COGNEE_AGENT_MODE"], "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestServerEnvFile(unittest.TestCase):
    """The server must not pick up ~/.cognee-plugin/.env (the other plugins' keys)."""

    def _spawn_env(self, workdir, env_overrides=None):
        captured = {}

        def fake_popen(args, env=None, **kwargs):
            captured.update(env or {})
            return mock.MagicMock()

        with (
            mock.patch.dict("os.environ", env_overrides or {}, clear=False),
            mock.patch.object(sb, "server_workdir", return_value=workdir),
            mock.patch.object(sb.subprocess, "Popen", side_effect=fake_popen),
        ):
            sb._spawn(9999, "", "", "/dev/null")
        return captured

    def test_the_server_loads_an_empty_env_file_instead_of_searching(self):
        with tempfile.TemporaryDirectory() as workdir:
            env = self._spawn_env(workdir, {"COGNEE_ENV_FILE": ""})
            path = env["COGNEE_ENV_FILE"]
            self.assertEqual(os.path.dirname(path), workdir)
            with open(path, encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip() and not line.startswith("#")]
            self.assertEqual(lines, [])

    def test_an_explicit_env_file_wins(self):
        with tempfile.TemporaryDirectory() as workdir:
            env = self._spawn_env(workdir, {"COGNEE_ENV_FILE": "/etc/my-cognee.env"})
            self.assertEqual(env["COGNEE_ENV_FILE"], "/etc/my-cognee.env")
