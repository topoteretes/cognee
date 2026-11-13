"""The smaller openclaw-parity changes: config keys, chunk size, 402, where the
spawned server and exit watcher get their credentials, and the server's working
directory."""

import io
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import config as config_mod  # noqa: E402
from cognee_integration_hermes import exit_watcher as ew  # noqa: E402
from cognee_integration_hermes import server_bootstrap as sb  # noqa: E402
from cognee_integration_hermes.http_backend import CogneeHttpError  # noqa: E402
from test_http_backend import FakeOpener, _backend  # noqa: E402


class TestConfigKeys(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (
                "COGNEE_RECALL_PASSAGE_CHARS",
                "COGNEE_RECALL_CONTEXT_CHARS",
                "COGNEE_CHUNK_SIZE",
            ):
                os.environ.pop(name, None)
            config = config_mod.load_config(tempfile.mkdtemp())
        self.assertEqual(config["recall_passage_chars"], 2000)
        self.assertEqual(config["recall_context_chars"], 12000)
        self.assertEqual(config["chunk_size"], 0)

    def test_environment_and_junk(self):
        env = {
            "COGNEE_RECALL_PASSAGE_CHARS": "0",
            "COGNEE_RECALL_CONTEXT_CHARS": "-5",
            "COGNEE_CHUNK_SIZE": "1024",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = config_mod.load_config(tempfile.mkdtemp())
        self.assertEqual(config["recall_passage_chars"], 0)  # 0 disables
        self.assertEqual(config["recall_context_chars"], 12000)  # junk -> default
        self.assertEqual(config["chunk_size"], 1024)


class TestChunkSize(unittest.TestCase):
    def test_unset_sends_nothing(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.handle_tool_call("cognee_remember", {"content": "x"})
            self.assertNotIn("chunk_size", fake.only_call("remember_permanent"))

    def test_configured_size_reaches_the_backend(self):
        with fake_backend() as fake:
            provider = make_provider(config={"chunk_size": 1024})
            provider.handle_tool_call("cognee_remember", {"content": "x"})
            self.assertEqual(fake.only_call("remember_permanent")["chunk_size"], 1024)

    def test_http_sends_the_form_field_only_when_set(self):
        opener = FakeOpener({"/api/v1/remember": {"status": "completed"}})
        _backend(opener).remember_permanent(
            text="t", dataset="d", session_ids=[], timeout=5, chunk_size=1024
        )
        self.assertEqual(opener.multipart_fields("/api/v1/remember")["chunk_size"], "1024")
        opener = FakeOpener({"/api/v1/remember": {"status": "completed"}})
        _backend(opener).remember_permanent(text="t", dataset="d", session_ids=[], timeout=5)
        self.assertNotIn("chunk_size", opener.multipart_fields("/api/v1/remember"))


class TestLlmBudgetExhausted(unittest.TestCase):
    def test_inline_improve_402_is_not_a_breaker_failure(self):
        with fake_backend() as fake:
            fake.errors["improve"] = CogneeHttpError(402, "budget")
            provider = make_provider()
            with self.assertLogs("cognee_integration_hermes.provider", "WARNING") as logs:
                provider.on_session_end([])
        self.assertEqual(provider._consecutive_failures, 0)
        self.assertIn("LLM budget is exhausted (402)", "\n".join(logs.output))

    def test_the_exit_watcher_says_why(self):
        state = {"url": "http://127.0.0.1:1", "session_id": "hermes_s1", "improve": True}
        error = urllib.error.HTTPError("u", 402, "Payment Required", {}, io.BytesIO(b""))

        def post(url, path, payload, **kwargs):
            if path == "/api/v1/improve":
                raise error
            return 200

        logged = []
        with (
            mock.patch.object(ew, "_post", side_effect=post),
            mock.patch.object(ew, "_log", side_effect=logged.append),
        ):
            ew.fire(state)
        self.assertTrue(any("budget is exhausted (402)" in line for line in logged), logged)


class TestServerSpawn(unittest.TestCase):
    def _spawn(self, environ, scope=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        workdir = Path(tmp.name) / "state"
        with (
            mock.patch.dict(os.environ, environ, clear=False),
            mock.patch.object(
                config_mod, "_hermes_get_secret", (scope or {}).get if scope is not None else None
            ),
            mock.patch.object(sb, "server_workdir", return_value=str(workdir)),
            mock.patch.object(sb.subprocess, "Popen") as popen,
        ):
            sb._spawn(8011, "", "", os.devnull)
        return popen.call_args.kwargs, workdir

    def test_the_server_runs_in_the_plugin_state_dir(self):
        kwargs, workdir = self._spawn({})
        self.assertEqual(kwargs["cwd"], str(workdir))
        self.assertNotEqual(kwargs["cwd"], os.getcwd())

    def test_the_llm_key_comes_from_the_hermes_secret_scope(self):
        kwargs, _ = self._spawn({"LLM_API_KEY": "another-profile"}, scope={"LLM_API_KEY": "mine"})
        self.assertEqual(kwargs["env"]["LLM_API_KEY"], "mine")

    def test_an_inherited_llm_key_never_reaches_the_server(self):
        # The active profile has no key; the gateway's environment holds another's.
        kwargs, _ = self._spawn({"LLM_API_KEY": "another-profile"}, scope={})
        self.assertNotIn("LLM_API_KEY", kwargs["env"])

    def test_every_api_key_comes_from_the_scope_and_none_is_inherited(self):
        inherited = {name: "another-profile" for name in sb._PROFILE_SECRETS}
        kwargs, _ = self._spawn(inherited, scope={"EMBEDDING_API_KEY": "mine"})
        env = kwargs["env"]
        self.assertEqual(env["EMBEDDING_API_KEY"], "mine")
        for name in sb._PROFILE_SECRETS:
            if name != "EMBEDDING_API_KEY":
                self.assertNotIn(name, env, name)

    def test_the_list_covers_cognees_embedding_and_per_task_keys(self):
        for name in (
            "EMBEDDING_API_KEY",
            "LLM_EXTRACTION_API_KEY",
            "LLM_QUERY_API_KEY",
            "LLM_SUMMARIZATION_API_KEY",
        ):
            self.assertIn(name, sb._PROFILE_SECRETS)

    def test_non_secret_llm_settings_are_still_inherited(self):
        kwargs, _ = self._spawn({"LLM_MODEL": "gpt-x", "LLM_PROVIDER": "openai"}, scope={})
        self.assertEqual(
            (kwargs["env"]["LLM_MODEL"], kwargs["env"]["LLM_PROVIDER"]), ("gpt-x", "openai")
        )

    def test_a_missing_llm_key_is_warned_about_once(self):
        with self.assertLogs("cognee_integration_hermes.server_bootstrap", "WARNING") as logs:
            self._spawn({}, scope={})
            self._spawn({}, scope={})
        self.assertEqual(len(logs.output), 1)
        self.assertIn("hermes memory setup", logs.output[0])

    def test_a_keyless_provider_gets_no_warning(self):
        with mock.patch.object(sb.logger, "warning") as warning:
            self._spawn({"LLM_PROVIDER": "ollama"}, scope={})
        warning.assert_not_called()

    def test_no_warning_when_the_key_is_there(self):
        with mock.patch.object(sb.logger, "warning") as warning:
            self._spawn({}, scope={"LLM_API_KEY": "mine"})
        warning.assert_not_called()

    def test_no_dotenv_file_reaches_the_server(self):
        # ~/.cognee/.env is no longer read: a value only in such a file stays out.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env_path = Path(tmp.name) / ".env"
        env_path.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        os.environ.pop("LLM_MODEL", None)
        kwargs, _ = self._spawn({"COGNEE_ENV_FILE": str(env_path)})
        self.assertNotIn("LLM_MODEL", kwargs["env"])


class TestExitWatcherKey(unittest.TestCase):
    def _spawned_env(self, environ, api_key):
        with (
            mock.patch.dict(os.environ, environ, clear=False),
            mock.patch.object(ew.subprocess, "Popen") as popen,
        ):
            ew._spawn("/tmp/state.json", [], api_key=api_key, log_path="")
        return popen.call_args.kwargs["env"]

    def test_the_provider_resolved_key_is_handed_over(self):
        env = self._spawned_env({}, "ck_resolved")
        self.assertEqual(env["COGNEE_API_KEY"], "ck_resolved")

    def test_an_inherited_key_never_stands_in(self):
        env = self._spawned_env({"COGNEE_API_KEY": "inherited"}, "")
        self.assertNotIn("COGNEE_API_KEY", env)

    def test_fire_uses_only_the_environment_key(self):
        sent = []

        def post(url, path, payload, *, api_key, timeout):
            sent.append(api_key)
            return 200

        state = {"url": "http://127.0.0.1:1", "session_id": "s", "improve": True}
        with (
            mock.patch.dict(os.environ, {"COGNEE_API_KEY": "ck_env"}),
            mock.patch.object(ew, "_post", side_effect=post),
            mock.patch.object(ew, "_log"),
        ):
            ew.fire(state)
        self.assertTrue(sent)
        self.assertEqual(set(sent), {"ck_env"})

    def test_fire_without_a_key_says_so(self):
        logged = []
        state = {"url": "http://127.0.0.1:1", "session_id": "s", "improve": False}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(ew, "_post", return_value=200),
            mock.patch.object(ew, "_log", side_effect=logged.append),
        ):
            os.environ.pop("COGNEE_API_KEY", None)
            ew.fire(state)
        self.assertTrue(any("no API key was handed over" in line for line in logged), logged)

    def test_the_watcher_reads_no_key_file(self):
        source = Path(ew.__file__).read_text(encoding="utf-8")
        self.assertNotIn("api_key.json", source)
        self.assertFalse(hasattr(ew, "_cached_api_key"))

    def test_the_default_workdir_is_under_the_shared_state_dir(self):
        with mock.patch.object(sb, "SHARED_PLUGIN_STATE_DIR", Path(tempfile.mkdtemp())) as root:
            self.assertEqual(sb.server_workdir(), str(root / "hermes"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
