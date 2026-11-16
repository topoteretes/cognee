"""Characterization: config resolution, coercion, and the files we write.

The subtle one is precedence: ``cognee.json`` is applied *over* the environment,
so a saved config wins. Inverting that on rewrite would silently break every
documented env var for anyone who has run the setup wizard.

Run standalone with ``python3 tests/test_config_contract.py``.
"""

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import make_provider  # noqa: E402
from cognee_integration_hermes import config as config_mod  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402

# Every env var load_config reads. Removed (not blanked) before each test:
# some keys distinguish "absent" from "empty" — see TestEmptyEnvVarQuirks.
_MANAGED_KEYS = (
    "COGNEE_BASE_URL",
    "COGNEE_SERVICE_URL",
    "COGNEE_API_KEY",
    "COGNEE_EMBEDDED",
    "COGNEE_PLUGIN_DATASET",
    "COGNEE_DATASET",
    "COGNEE_TOP_K",
    "COGNEE_LOCAL_PORT",
    "COGNEE_SERVER_BOOT_TIMEOUT",
    "COGNEE_AUTO_ROUTE",
    "COGNEE_IMPROVE_ON_END",
    "COGNEE_IMPROVE_BACKGROUND",
    "COGNEE_SESSION_PREFIX",
    "COGNEE_DATA_ROOT",
    "COGNEE_SYSTEM_ROOT",
    "COGNEE_RECALL_TIMEOUT",
    "COGNEE_WRITE_TIMEOUT",
    "COGNEE_IMPROVE_TIMEOUT",
    "COGNEE_HERMES_USER_EMAIL",
    "COGNEE_HERMES_USER_PASSWORD",
    "LLM_API_KEY",
    "LLM_MODEL",
)


@contextlib.contextmanager
def _clean_env(overrides=None):
    """Run with every managed key *removed*, then apply *overrides*."""
    env = {key: value for key, value in os.environ.items() if key not in _MANAGED_KEYS}
    env.update(overrides or {})
    with mock.patch.dict(os.environ, env, clear=True):
        yield


class _TmpHome:
    """A temporary HERMES_HOME with an optional pre-written cognee.json."""

    def __init__(self, file_config=None):
        self._file_config = file_config
        self._tmp = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        home = Path(self._tmp.name)
        if self._file_config is not None:
            (home / "cognee.json").write_text(json.dumps(self._file_config), encoding="utf-8")
        return home

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False


def _load(env=None, file_config=None):
    with _TmpHome(file_config) as home:
        with _clean_env(env):
            return config_mod.load_config(home)


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


class TestPrecedence(unittest.TestCase):
    def test_file_config_overrides_the_environment(self):
        cfg = _load(
            env={"COGNEE_DATASET": "from_env"},
            file_config={"dataset": "from_file"},
        )
        self.assertEqual(cfg["dataset"], "from_file")

    def test_environment_is_used_when_the_file_omits_the_key(self):
        cfg = _load(env={"COGNEE_DATASET": "from_env"}, file_config={"top_k": 3})
        self.assertEqual(cfg["dataset"], "from_env")

    def test_null_file_values_do_not_override(self):
        cfg = _load(env={"COGNEE_DATASET": "from_env"}, file_config={"dataset": None})
        self.assertEqual(cfg["dataset"], "from_env")

    def test_unreadable_file_is_ignored_rather_than_fatal(self):
        with _TmpHome() as home:
            (home / "cognee.json").write_text("{not json", encoding="utf-8")
            with _clean_env():
                cfg = config_mod.load_config(home)
        self.assertEqual(cfg["dataset"], config_mod.DEFAULT_DATASET)

    def test_base_url_wins_over_the_deprecated_alias(self):
        cfg = _load(
            env={"COGNEE_BASE_URL": "https://canonical", "COGNEE_SERVICE_URL": "https://legacy"}
        )
        self.assertEqual(cfg["service_url"], "https://canonical")

    def test_plugin_dataset_wins_over_the_hermes_alias(self):
        # COGNEE_PLUGIN_DATASET is what the other cognee plugins read; the
        # 0.1.x-era COGNEE_DATASET stays as a lower-precedence alias.
        cfg = _load(env={"COGNEE_PLUGIN_DATASET": "shared", "COGNEE_DATASET": "legacy"})
        self.assertEqual(cfg["dataset"], "shared")

    def test_the_hermes_dataset_alias_still_works(self):
        self.assertEqual(_load(env={"COGNEE_DATASET": "legacy"})["dataset"], "legacy")


# --------------------------------------------------------------------------
# Defaults and coercion
# --------------------------------------------------------------------------


class TestDefaults(unittest.TestCase):
    def test_documented_defaults(self):
        cfg = _load()
        # The dataset is shared with the other cognee agent plugins on purpose.
        self.assertEqual(cfg["dataset"], "agent_sessions")
        self.assertEqual(cfg["top_k"], 5)
        self.assertEqual(cfg["local_port"], 8011)
        self.assertEqual(cfg["server_boot_timeout"], 600)
        self.assertEqual(cfg["session_prefix"], "hermes")
        self.assertEqual(cfg["recall_timeout"], 120)
        self.assertEqual(cfg["write_timeout"], 120)
        self.assertEqual(cfg["improve_timeout"], 300)
        self.assertIs(cfg["auto_route"], True)
        self.assertIs(cfg["improve_on_end"], True)
        self.assertIs(cfg["embedded"], False)

    def test_improve_background_is_an_empty_tristate_by_default(self):
        # "" means auto (background in server/remote, synchronous in embedded).
        self.assertEqual(_load()["improve_background"], "")

    def test_improve_background_can_be_forced(self):
        self.assertEqual(
            _load(env={"COGNEE_IMPROVE_BACKGROUND": "false"})["improve_background"], "false"
        )


class TestCoercion(unittest.TestCase):
    def test_top_k_floor_is_one(self):
        self.assertEqual(_load(file_config={"top_k": 0})["top_k"], 1)
        self.assertEqual(_load(file_config={"top_k": -5})["top_k"], 1)

    def test_non_numeric_top_k_falls_back_to_the_default(self):
        self.assertEqual(_load(file_config={"top_k": "abc"})["top_k"], 5)

    def test_local_port_clamped_to_the_valid_range(self):
        self.assertEqual(_load(env={"COGNEE_LOCAL_PORT": "999999"})["local_port"], 65535)
        self.assertEqual(_load(env={"COGNEE_LOCAL_PORT": "0"})["local_port"], 1)

    def test_timeout_floors_are_one(self):
        cfg = _load(file_config={"recall_timeout": 0, "write_timeout": -1, "improve_timeout": 0})
        self.assertEqual(cfg["recall_timeout"], 1)
        self.assertEqual(cfg["write_timeout"], 1)
        self.assertEqual(cfg["improve_timeout"], 1)

    def test_server_boot_timeout_floor_is_one(self):
        self.assertEqual(_load(file_config={"server_boot_timeout": 0})["server_boot_timeout"], 1)

    def test_string_booleans_from_the_config_file_are_coerced(self):
        cfg = _load(file_config={"auto_route": "no", "improve_on_end": "0", "embedded": "yes"})
        self.assertIs(cfg["auto_route"], False)
        self.assertIs(cfg["improve_on_end"], False)
        self.assertIs(cfg["embedded"], True)


class TestEmptyEnvVarQuirks(unittest.TestCase):
    """An empty env var is not the same as an absent one — and not uniformly so.

    Plain ``os.environ.get(key, DEFAULT)`` reads take the empty string as a real
    value, while the ``str_to_int`` / ``str_to_bool`` reads fall back to their
    default. Pinned because it is exactly the kind of asymmetry a rewrite tidies
    up by accident, changing behaviour for anyone who exports a blank var.
    """

    def test_blank_dataset_is_honoured_as_empty(self):
        self.assertEqual(_load(env={"COGNEE_DATASET": ""})["dataset"], "")

    def test_blank_session_prefix_is_honoured_as_empty(self):
        self.assertEqual(_load(env={"COGNEE_SESSION_PREFIX": ""})["session_prefix"], "")

    def test_blank_numeric_vars_fall_back_to_defaults(self):
        cfg = _load(env={"COGNEE_TOP_K": "", "COGNEE_LOCAL_PORT": ""})
        self.assertEqual(cfg["top_k"], 5)
        self.assertEqual(cfg["local_port"], 8011)

    def test_blank_boolean_vars_fall_back_to_defaults(self):
        cfg = _load(env={"COGNEE_AUTO_ROUTE": "", "COGNEE_EMBEDDED": ""})
        self.assertIs(cfg["auto_route"], True)
        self.assertIs(cfg["embedded"], False)

    def test_the_provider_re_defaults_a_blank_dataset_at_use_time(self):
        # provider.initialize does `config.get("dataset") or DEFAULT_DATASET`, so a
        # blank dataset never reaches cognee — the recovery lives there, not in
        # load_config. Both halves have to survive together.
        provider = make_provider(dataset=config_mod.DEFAULT_DATASET)
        self.assertEqual(
            str(_load(env={"COGNEE_DATASET": ""}).get("dataset") or config_mod.DEFAULT_DATASET),
            provider._dataset,
        )


class TestStrToBool(unittest.TestCase):
    def test_truthy_spellings(self):
        for value in ("1", "true", "TRUE", "yes", "y", "on", " True "):
            self.assertTrue(config_mod.str_to_bool(value, False), value)

    def test_falsy_spellings(self):
        for value in ("0", "false", "no", "n", "off", " Off "):
            self.assertFalse(config_mod.str_to_bool(value, True), value)

    def test_unrecognized_and_none_use_the_default(self):
        self.assertTrue(config_mod.str_to_bool("maybe", True))
        self.assertFalse(config_mod.str_to_bool("maybe", False))
        self.assertTrue(config_mod.str_to_bool(None, True))

    def test_actual_booleans_pass_through(self):
        self.assertTrue(config_mod.str_to_bool(True, False))
        self.assertFalse(config_mod.str_to_bool(False, True))


# --------------------------------------------------------------------------
# Files we write
# --------------------------------------------------------------------------


class _FakeHermesConfig:
    """Stand-ins for hermes_cli.config / hermes_constants, recording each call."""

    def __init__(self, refuse=()):
        self.saved: list[tuple[str, str]] = []
        self.homes: list[str] = []
        self.refuse = set(refuse)

    def save_env_value(self, key, value):
        if key in self.refuse:
            raise ValueError("denylisted")
        self.saved.append((key, value))

    def modules(self):
        cli = types.ModuleType("hermes_cli")
        cli_config = types.ModuleType("hermes_cli.config")
        cli_config.save_env_value = self.save_env_value
        constants = types.ModuleType("hermes_constants")
        constants.set_hermes_home_override = lambda home: self.homes.append(home) or "token"
        constants.reset_hermes_home_override = lambda token: self.homes.append(f"reset:{token}")
        return {"hermes_cli": cli, "hermes_cli.config": cli_config, "hermes_constants": constants}


class TestSaveEnvVars(unittest.TestCase):
    def test_values_go_through_hermes_writer_under_the_profile_home(self):
        fake = _FakeHermesConfig()
        with mock.patch.dict(sys.modules, fake.modules()):
            config_mod.save_env_vars({"COGNEE_API_KEY": "secret", "COGNEE_SERVICE_URL": ""}, "/h")
        self.assertEqual(fake.saved, [("COGNEE_API_KEY", "secret"), ("COGNEE_SERVICE_URL", "")])
        self.assertEqual(fake.homes, ["/h", "reset:token"])

    def test_a_refused_name_is_skipped_not_fatal(self):
        fake = _FakeHermesConfig(refuse={"BAD"})
        with mock.patch.dict(sys.modules, fake.modules()):
            config_mod.save_env_vars({"BAD": "x", "LLM_API_KEY": "k"}, "/h")
        self.assertEqual(fake.saved, [("LLM_API_KEY", "k")])

    def test_no_values_touches_nothing(self):
        fake = _FakeHermesConfig()
        with mock.patch.dict(sys.modules, fake.modules()):
            config_mod.save_env_vars({}, "/h")
        self.assertEqual((fake.saved, fake.homes), ([], []))

    def test_the_plugin_never_opens_a_dotenv_itself(self):
        source = Path(config_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn('".env"', source)


class TestSecret(unittest.TestCase):
    def test_reads_through_hermes_get_secret_when_available(self):
        calls = []

        def get_secret(name):
            calls.append(name)
            return " from-scope "

        with mock.patch.object(config_mod, "_hermes_get_secret", get_secret):
            self.assertEqual(config_mod.secret("COGNEE_API_KEY"), "from-scope")
        self.assertEqual(calls, ["COGNEE_API_KEY"])

    def test_falls_back_to_the_environment_without_hermes(self):
        with (
            mock.patch.object(config_mod, "_hermes_get_secret", None),
            mock.patch.dict(os.environ, {"LLM_API_KEY": "env-key"}),
        ):
            self.assertEqual(config_mod.secret("LLM_API_KEY"), "env-key")

    def test_load_config_takes_credentials_from_get_secret(self):
        scope = {"COGNEE_API_KEY": "ck", "LLM_API_KEY": "sk", "COGNEE_BASE_URL": "https://c"}
        with (
            _TmpHome() as home,
            mock.patch.object(config_mod, "_hermes_get_secret", scope.get),
        ):
            config = config_mod.load_config(home)
        self.assertEqual(
            (config["api_key"], config["llm_api_key"], config["service_url"]),
            ("ck", "sk", "https://c"),
        )


class TestSaveConfig(unittest.TestCase):
    def test_merges_into_an_existing_file(self):
        with _TmpHome(file_config={"dataset": "old", "top_k": 3}) as home:
            config_mod.save_config({"dataset": "new"}, home)
            written = json.loads((home / "cognee.json").read_text(encoding="utf-8"))
        self.assertEqual(written["dataset"], "new")
        self.assertEqual(written["top_k"], 3)

    def test_skips_none_values(self):
        with _TmpHome(file_config={"dataset": "keep"}) as home:
            config_mod.save_config({"dataset": None, "top_k": 4}, home)
            written = json.loads((home / "cognee.json").read_text(encoding="utf-8"))
        self.assertEqual(written["dataset"], "keep")
        self.assertEqual(written["top_k"], 4)

    def test_provider_save_config_never_writes_secrets(self):
        provider = make_provider()
        with _TmpHome() as home:
            provider.save_config(
                {
                    "dataset": "hermes",
                    "api_key": "ck_secret",
                    "llm_api_key": "sk_secret",
                },
                str(home),
            )
            written = json.loads((home / "cognee.json").read_text(encoding="utf-8"))
        self.assertEqual(written, {"dataset": "hermes"})

    def test_provider_save_config_writes_nothing_when_only_secrets_are_given(self):
        provider = make_provider()
        with _TmpHome() as home:
            provider.save_config({"api_key": "ck", "llm_api_key": "sk"}, str(home))
            self.assertFalse((home / "cognee.json").exists())


class TestResolveLocalRoots(unittest.TestCase):
    """Where cognee stores things — the shared ~/.cognee unless told otherwise."""

    def test_the_shared_cognee_home_by_default(self):
        data, system = config_mod.resolve_local_roots({})
        self.assertEqual(data, str(config_mod.SHARED_COGNEE_HOME / "data"))
        self.assertEqual(system, str(config_mod.SHARED_COGNEE_HOME / "system"))

    def test_the_shared_home_matches_the_other_plugins(self):
        # claude-code/codex/openclaw pin ~/.cognee/{data,system}; the point of
        # the default is that any plugin's server serves the same store.
        self.assertEqual(config_mod.SHARED_COGNEE_HOME, Path.home() / ".cognee")
        self.assertEqual(config_mod.SHARED_PLUGIN_STATE_DIR, Path.home() / ".cognee-plugin")

    def test_explicit_config_wins(self):
        data, system = config_mod.resolve_local_roots(
            {"data_root": "/mnt/graph", "system_root": "/mnt/sys"}
        )
        self.assertEqual(data, "/mnt/graph")
        self.assertEqual(system, "/mnt/sys")

    def test_a_single_override_leaves_the_other_shared(self):
        data, system = config_mod.resolve_local_roots({"data_root": "/mnt/graph"})
        self.assertEqual(data, "/mnt/graph")
        self.assertEqual(system, str(config_mod.SHARED_COGNEE_HOME / "system"))


class TestBackupPaths(unittest.TestCase):
    """``hermes backup`` walks HERMES_HOME itself; declare only what is outside it."""

    def test_the_shared_default_store_is_declared(self):
        # The default roots live in ~/.cognee, outside any HERMES_HOME — a backup
        # deliberately includes the machine's shared memory store.
        with _TmpHome() as home:
            with _clean_env():
                provider = make_provider()
                with mock.patch.object(provider_mod, "resolve_hermes_home", return_value=home):
                    paths = provider.backup_paths()
        self.assertEqual(
            paths,
            [
                str(config_mod.SHARED_COGNEE_HOME / "data"),
                str(config_mod.SHARED_COGNEE_HOME / "system"),
            ],
        )

    def test_externally_configured_roots_are_declared(self):
        with _TmpHome() as home:
            with _clean_env({"COGNEE_DATA_ROOT": "/mnt/graph", "COGNEE_SYSTEM_ROOT": "/mnt/sys"}):
                provider = make_provider()
                with mock.patch.object(provider_mod, "resolve_hermes_home", return_value=home):
                    paths = provider.backup_paths()
        self.assertEqual(sorted(paths), ["/mnt/graph", "/mnt/sys"])

    def test_roots_inside_hermes_home_need_no_declaring(self):
        with _TmpHome() as home:
            env = {
                "COGNEE_DATA_ROOT": str(home / "cognee" / "data"),
                "COGNEE_SYSTEM_ROOT": str(home / "cognee" / "system"),
            }
            with _clean_env(env):
                provider = make_provider()
                with mock.patch.object(provider_mod, "resolve_hermes_home", return_value=home):
                    self.assertEqual(provider.backup_paths(), [])

    def test_callable_without_initialize(self):
        # Upstream requires this: backup runs on providers that were never started.
        with _clean_env():
            provider = provider_mod.CogneeMemoryProvider()
            self.assertIsInstance(provider.backup_paths(), list)


class TestConfigPath(unittest.TestCase):
    def test_resolves_under_the_given_hermes_home(self):
        with _TmpHome() as home:
            self.assertEqual(config_mod.config_path(home), home / "cognee.json")

    def test_is_none_when_hermes_home_cannot_be_resolved(self):
        # hermes_constants is not importable outside a Hermes install.
        self.assertIsNone(config_mod.config_path(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
