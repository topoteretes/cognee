"""The recall mode: configured, chosen by a conversation, and stored (SDK-1393).

``recall_mode`` (``COGNEE_PLUGIN_RECALL_MODE``, or the setup wizard's
``recall_mode`` in cognee.json) decides: "auto" names no dataset in graph recall
and the server picks; "pinned" keeps recall on the active dataset (reason
``configured_pin``). With no mode set, the default dataset is auto and a
configured non-default dataset keeps recall pinned to it (reason
``configured_pin``), as before SDK-1393; such a user is told once that auto
exists. ``COGNEE_PLUGIN_DATASET=auto`` only means the default dataset. The
conversation's own choice wins: ``action='auto'``, ``action='pin'`` or a switch;
``reset`` goes back to the configured dataset and mode. A mode chosen without a
switch is stored as the mode only, so it applies to whatever dataset is active.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import FakeBackend, fake_backend  # noqa: E402
from cognee_integration_hermes import config as config_mod  # noqa: E402
from cognee_integration_hermes import dataset_overrides  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402

_ENV = {"COGNEE_BASE_URL": "", "COGNEE_SERVICE_URL": "", "COGNEE_EMBEDDED": "true"}


def _initialize(fake, dataset=None, session_id="s-1", recall_mode=None, **env):
    """The real initialize() in embedded mode against ``fake``."""
    provider = provider_mod.CogneeMemoryProvider(backend=fake)
    if dataset is not None:
        env["COGNEE_PLUGIN_DATASET"] = dataset
    if recall_mode is not None:
        env["COGNEE_PLUGIN_RECALL_MODE"] = recall_mode
    with mock.patch.dict("os.environ", {**_ENV, **env}, clear=False):
        provider.initialize(session_id)
    return provider


def _call(provider, args):
    return json.loads(provider.handle_tool_call("cognee_switch_dataset", args))


def _graph_datasets(provider, fake):
    fake.calls.clear()
    provider.handle_tool_call("cognee_recall", {"query": "what did we decide?"})
    (kwargs,) = [k for k in fake.kwargs_for("recall") if k["scope"] == ["graph"]]
    return kwargs["datasets"]


# -- the configured mode -------------------------------------------------------


@pytest.mark.parametrize(("dataset", "mode"), [(None, None), (None, "auto"), ("work", "auto")])
def test_recall_is_auto_for_the_default_dataset_or_a_configured_auto(dataset, mode):
    with fake_backend() as fake:
        provider = _initialize(fake, dataset, recall_mode=mode)
        assert _graph_datasets(provider, fake) is None
        current = _call(provider, {"action": "current"})
    assert current["recall"] == "auto"
    assert "reason" not in current


@pytest.mark.parametrize("dataset", ["work", config_mod.DEFAULT_DATASET])
def test_a_configured_dataset_with_no_mode_keeps_recall_pinned(dataset):
    with fake_backend() as fake:
        provider = _initialize(fake, dataset)
        datasets = _graph_datasets(provider, fake)
        current = _call(provider, {"action": "current"})
    if dataset == config_mod.DEFAULT_DATASET:
        assert datasets is None and current["recall"] == "auto"
        return
    assert datasets == ["work"]
    assert (current["recall"], current["reason"]) == ("pinned", "configured_pin")
    assert "no recall mode is set" in current["message"]


def test_a_configured_dataset_with_no_mode_in_cognee_json_keeps_recall_pinned(tmp_path):
    (tmp_path / "cognee.json").write_text(json.dumps({"dataset": "work"}))
    assert config_mod.effective_recall_mode(config_mod.load_config(tmp_path)) == "pinned"
    (tmp_path / "cognee.json").write_text(json.dumps({"dataset": "work", "recall_mode": "auto"}))
    assert config_mod.effective_recall_mode(config_mod.load_config(tmp_path)) == "auto"


def test_the_conversations_own_auto_wins_over_the_dataset_pin():
    with fake_backend() as fake:
        provider = _initialize(fake, "work")
        out = _call(provider, {"action": "auto"})
        assert (out["recall"], out["lifted"]) == ("auto", "configured_pin")
        resumed = _initialize(fake, "work")
        assert _graph_datasets(resumed, fake) is None


@pytest.mark.parametrize(("dataset", "mode"), [("work", "pinned"), (None, " Pinned ")])
def test_a_configured_pinned_mode_pins_recall_to_the_configured_dataset(dataset, mode):
    expected = dataset or config_mod.DEFAULT_DATASET
    with fake_backend() as fake:
        provider = _initialize(fake, dataset, recall_mode=mode)
        assert _graph_datasets(provider, fake) == [expected]
        current = _call(provider, {"action": "current"})
    assert (current["dataset"], current["recall"], current["reason"]) == (
        expected,
        "pinned",
        "configured_pin",
    )


@pytest.mark.parametrize("value", ["bogus", "pin"])
def test_an_unknown_recall_mode_is_auto(value):
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode=value)
        assert _graph_datasets(provider, fake) is None
    with mock.patch.dict("os.environ", {"COGNEE_PLUGIN_RECALL_MODE": value}):
        assert config_mod.load_config()["recall_mode"] == "auto"


def test_a_configured_auto_dataset_is_the_default_and_leaves_the_mode_alone():
    with fake_backend() as fake:
        provider = _initialize(fake, " AUTO ")
        assert provider._dataset == config_mod.DEFAULT_DATASET
        assert _graph_datasets(provider, fake) is None
        provider.handle_tool_call("cognee_remember", {"content": "We ship on Fridays."})
        assert fake.wait("remember_permanent")
        saved = fake.kwargs_for("remember_permanent")[0]["dataset"]
    assert saved == config_mod.DEFAULT_DATASET


def test_the_config_schema_offers_the_recall_mode():
    schema = provider_mod.CogneeMemoryProvider(backend=FakeBackend()).get_config_schema()
    (entry,) = [row for row in schema if row["key"] == "recall_mode"]
    assert entry["default"] == "auto"
    assert entry["choices"] == ["auto", "pinned"]
    assert entry["env_var"] == "COGNEE_PLUGIN_RECALL_MODE"


@pytest.mark.parametrize(
    ("current", "answer", "saved"),
    [
        ({"recall_mode": "pinned"}, "", "pinned"),
        ({}, "pinned", "pinned"),
        ({}, "nonsense", "auto"),
    ],
)
def test_the_setup_wizard_asks_for_the_recall_mode(current, answer, saved):
    written = {}

    def prompt(label, default=""):
        if label.startswith("Recall mode"):
            return answer or default
        return {"Mode": "remote"}.get(label, default)

    with (
        mock.patch.object(provider_mod, "load_config", return_value=current),
        mock.patch.object(provider_mod, "_prompt", side_effect=prompt),
        mock.patch.object(provider_mod, "_prompt_secret", return_value=""),
        mock.patch.object(provider_mod, "_prompt_bool", side_effect=lambda label, default: default),
        mock.patch.object(provider_mod.runtime_env, "local_ready", return_value=True),
        mock.patch.object(
            provider_mod,
            "save_plugin_config",
            side_effect=lambda values, home: written.update(values),
        ),
        mock.patch.object(provider_mod, "save_env_vars"),
    ):
        provider_mod.CogneeMemoryProvider(backend=FakeBackend()).post_setup("/tmp/hh", {})
    assert written["recall_mode"] == saved


@pytest.mark.parametrize(
    ("current", "dataset", "offered"),
    [
        ({"recall_mode_source": "default"}, "work", "pinned"),
        ({"recall_mode_source": "default"}, config_mod.DEFAULT_DATASET, "auto"),
        ({"recall_mode": "auto", "recall_mode_source": "file"}, "work", "auto"),
        ({"recall_mode": "auto", "recall_mode_source": "env"}, "work", "auto"),
    ],
)
def test_the_setup_wizard_offers_pinned_for_a_non_default_dataset(current, dataset, offered):
    defaults = {}

    def prompt(label, default=""):
        if label.startswith("Recall mode"):
            defaults["recall"] = default
            return default
        return {"Mode": "remote", "Dataset": dataset}.get(label, default)

    with (
        mock.patch.object(provider_mod, "load_config", return_value=current),
        mock.patch.object(provider_mod, "_prompt", side_effect=prompt),
        mock.patch.object(provider_mod, "_prompt_secret", return_value=""),
        mock.patch.object(provider_mod, "_prompt_bool", side_effect=lambda label, default: default),
        mock.patch.object(provider_mod.runtime_env, "local_ready", return_value=True),
        mock.patch.object(provider_mod, "save_plugin_config"),
        mock.patch.object(provider_mod, "save_env_vars"),
    ):
        provider_mod.CogneeMemoryProvider(backend=FakeBackend()).post_setup("/tmp/hh", {})
    assert defaults["recall"] == offered


def test_the_setup_wizard_asks_the_recall_mode_once_and_keeps_auto_route():
    # Asked right after the recall mode, "Auto-route recall" read as the same
    # question twice; auto_route (the search type) keeps its saved value.
    asked, written = [], {}

    def prompt(label, default=""):
        asked.append(label)
        return {"Mode": "remote"}.get(label, default)

    def prompt_bool(label, default):
        asked.append(label)
        return default

    with (
        mock.patch.object(provider_mod, "load_config", return_value={"auto_route": False}),
        mock.patch.object(provider_mod, "_prompt", side_effect=prompt),
        mock.patch.object(provider_mod, "_prompt_secret", return_value=""),
        mock.patch.object(provider_mod, "_prompt_bool", side_effect=prompt_bool),
        mock.patch.object(provider_mod.runtime_env, "local_ready", return_value=True),
        mock.patch.object(
            provider_mod,
            "save_plugin_config",
            side_effect=lambda values, home: written.update(values),
        ),
        mock.patch.object(provider_mod, "save_env_vars"),
    ):
        provider_mod.CogneeMemoryProvider(backend=FakeBackend()).post_setup("/tmp/hh", {})
    assert [label for label in asked if "recall" in label.lower()] == ["Recall mode (auto/pinned)"]
    assert "auto_route" not in written  # the saved False is left as it is


# -- the conversation's own choice ---------------------------------------------


def test_conversation_auto_lifts_a_configured_pin_and_survives_a_resume():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
        out = _call(provider, {"action": "auto"})
        assert (out["reason"], out["recall"], out["lifted"]) == (
            "recall_auto",
            "auto",
            "configured_pin",
        )
        assert _graph_datasets(provider, fake) is None
        # Saves stay in the configured dataset.
        assert provider._dataset == "work"
        resumed = _initialize(FakeBackend(), "work", recall_mode="pinned")
    assert dataset_overrides.load_override("s-1")["recall"] == "auto"
    assert resumed._recall_pinned is False


def test_conversation_auto_wins_over_a_later_configured_pin():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="auto")
        out = _call(provider, {"action": "auto"})
        assert out["reason"] == "already_auto"
        resumed = _initialize(fake, "work", recall_mode="pinned")
        assert _graph_datasets(resumed, fake) is None
    stored = dataset_overrides.load_override("s-1")
    assert stored["recall"] == "auto"
    assert "dataset" not in stored


def test_a_new_conversation_gets_the_configured_pin_back():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
        _call(provider, {"action": "auto"})
        provider.on_session_switch("s-2")
        current = _call(provider, {"action": "current"})
    assert current["reason"] == "configured_pin"


def test_reset_goes_back_to_the_configured_mode():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
        _call(provider, {"action": "auto"})
        out = _call(provider, {"action": "reset"})
        current = _call(provider, {"action": "current"})
    assert (out["reason"], out["recall"], out["recall_reason"]) == (
        "recall_reset",
        "pinned",
        "configured_pin",
    )
    assert current["reason"] == "configured_pin"
    # Following the config without a switch leaves nothing to store.
    assert dataset_overrides.load_override("s-1") == {}


@pytest.mark.parametrize(
    ("recall_mode", "recall", "reason"),
    [("pinned", "pinned", "configured_pin"), ("auto", "auto", None)],
)
def test_reset_after_a_switch_reports_the_configured_mode(recall_mode, recall, reason):
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode=recall_mode)
        _call(provider, {"action": "switch", "dataset": "other"})
        out = _call(provider, {"action": "reset"})
        # The reset follows the config, so a resume picks up a changed one.
        resumed = _initialize(fake, "work", recall_mode="pinned")
        assert _graph_datasets(resumed, fake) == ["work"]
    assert out["switched"] is True
    assert (out["dataset"], out["recall"], out.get("recall_reason")) == ("work", recall, reason)


# -- action='pin' --------------------------------------------------------------


def test_pin_pins_the_active_dataset_without_a_new_session():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="auto")
        fake.calls.clear()
        out = _call(provider, {"action": "pin"})
        assert fake.calls == []
        assert _graph_datasets(provider, fake) == ["work"]
        again = _call(provider, {"action": "pin"})
    assert (out["switched"], out["reason"], out["recall"], out["recall_reason"]) == (
        False,
        "recall_restricted",
        "pinned",
        "dataset_picked",
    )
    assert again["reason"] == "already_active"
    assert provider._session_cognee_id == "hermes_s-1"
    stored = dataset_overrides.load_override("s-1")
    assert stored["recall"] == "pinned"
    assert "dataset" not in stored and "counter" not in stored


def test_pin_wins_over_a_configured_pin_and_a_later_configured_auto():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
        out = _call(provider, {"action": "pin"})
        assert (out["reason"], out["recall_reason"]) == ("already_active", "dataset_picked")
        resumed = _initialize(fake, "work", recall_mode="auto")
        assert _graph_datasets(resumed, fake) == ["work"]
        assert _call(resumed, {"action": "current"})["reason"] == "dataset_picked"


def test_after_a_config_dataset_change_a_pin_follows_the_new_active_dataset():
    with fake_backend() as fake:
        provider = _initialize(fake, "work")
        _call(provider, {"action": "pin"})
        resumed = _initialize(fake, "research")
        assert resumed._dataset == "research"
        assert _graph_datasets(resumed, fake) == ["research"]
        current = _call(resumed, {"action": "current"})
    assert (current["dataset"], current["recall"], current["reason"]) == (
        "research",
        "pinned",
        "dataset_picked",
    )


# -- legacy entries ------------------------------------------------------------


def test_a_pre_sdk_1393_entry_on_the_configured_dataset_follows_the_config():
    dataset_overrides._save({"s-1": {"dataset": "work", "counter": 2}})
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
    assert provider._recall_pin_reason == "configured_pin"
    assert provider._session_cognee_id.endswith("__2")
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="auto")
    assert provider._recall_pinned is False


def test_a_pre_sdk_1393_switch_entry_stays_pinned():
    dataset_overrides._save({"s-1": {"dataset": "other", "counter": 2}})
    with fake_backend() as fake:
        provider = _initialize(fake, "work")
    assert (provider._dataset, provider._recall_pin_reason) == ("other", "dataset_picked")


# -- one message per result ----------------------------------------------------


def test_every_result_carries_the_message_for_its_mode_and_reason():
    with fake_backend() as fake:
        provider = _initialize(fake, "work", recall_mode="pinned")
        results = {
            "current_configured": _call(provider, {"action": "current"}),
            "auto": _call(provider, {"action": "auto"}),
            "pin": _call(provider, {"action": "pin"}),
            "switch": _call(provider, {"action": "switch", "dataset": "other"}),
            "reset": _call(provider, {"action": "reset"}),
            "current": _call(provider, {"action": "current"}),
        }
    for name, out in results.items():
        assert out["message"], name
    assert "configured recall mode" in results["current_configured"]["message"]
    assert "Recall is auto" in results["auto"]["message"]
    assert "own choice" in results["pin"]["message"]
    assert "'other'" in results["switch"]["message"]
    assert "configured recall mode" in results["reset"]["message"]


@pytest.mark.parametrize(
    ("file_mode", "env_mode", "mode", "source"),
    [
        # The exported variable wins over cognee.json, which the setup wizard
        # always writes: otherwise COGNEE_PLUGIN_RECALL_MODE would do nothing
        # after setup.
        ("pinned", "auto", "auto", "env"),
        ("auto", "pinned", "pinned", "env"),
        ("pinned", None, "pinned", "file"),
        (None, None, "auto", "default"),
    ],
)
def test_the_exported_recall_mode_wins_over_cognee_json(
    tmp_path, monkeypatch, file_mode, env_mode, mode, source
):
    file_config = {"dataset": "work"}
    if file_mode is not None:
        file_config["recall_mode"] = file_mode
    (tmp_path / "cognee.json").write_text(json.dumps(file_config))
    if env_mode is not None:
        monkeypatch.setenv("COGNEE_PLUGIN_RECALL_MODE", env_mode)
    config = config_mod.load_config(tmp_path)
    assert config["recall_mode"] == mode
    assert config["recall_mode_source"] == source


@pytest.mark.parametrize(
    ("source", "named", "not_named"),
    [
        ("file", "cognee.json", "COGNEE_PLUGIN_RECALL_MODE"),
        ("env", "COGNEE_PLUGIN_RECALL_MODE=pinned", "cognee.json"),
    ],
)
def test_a_configured_pin_names_where_it_was_configured(source, named, not_named):
    message = provider_mod._recall_message(
        "pinned", provider_mod.RECALL_PIN_CONFIGURED, "work", source
    )
    assert named in message
    assert not_named not in message


# -- the one-time note that auto exists ----------------------------------------


def _reply(text="Here you go.", session_id="s-1"):
    from cognee_integration_hermes import transform_llm_output

    return transform_llm_output(response_text=text, platform="cli", session_id=session_id)


def test_the_dataset_pin_note_is_appended_once_per_dataset(caplog):
    from cognee_integration_hermes import recall_notice

    caplog.set_level("INFO", logger=recall_notice.__name__)
    with fake_backend() as fake:
        _initialize(fake, "work")
        first = _reply()
        assert _reply() is None  # appended to the first reply only
        _initialize(fake, "work", session_id="s-2")
        assert _reply(session_id="s-2") is None  # once per machine and dataset
        _initialize(fake, "research", session_id="s-3")
        other = _reply(session_id="s-3")
    assert first == "Here you go.\n\n" + recall_notice.notice_text("work")
    assert first.endswith(
        "Cognee: recall stays on `work`. Cognee can also pick the best-matching dataset for "
        "each prompt from all you can read: set `recall_mode: auto` in cognee.json (or "
        "`COGNEE_PLUGIN_RECALL_MODE=auto`, or rerun `hermes memory setup cognee`), or ask to "
        "set recall to auto for this conversation."
    )
    assert "`research`" in other
    logged = [r.getMessage() for r in caplog.records if "recall stays on" in r.getMessage()]
    assert len(logged) == 2


@pytest.mark.parametrize(
    ("dataset", "mode"),
    [(None, None), ("work", "pinned"), ("work", "auto")],
)
def test_no_note_for_the_default_dataset_or_a_set_mode(dataset, mode):
    with fake_backend() as fake:
        _initialize(fake, dataset, recall_mode=mode)
        assert _reply() is None


@pytest.mark.parametrize("choice", ["pinned", "auto"])
def test_no_note_for_a_conversation_with_its_own_mode(choice):
    dataset_overrides.save_override("s-1", None, 0, recall=choice)
    with fake_backend() as fake:
        provider = _initialize(fake, "work")
        assert provider._recall_choice == choice
        assert _reply() is None


def test_the_note_is_not_stored_with_the_answer():
    from cognee_integration_hermes import recall_notice

    with fake_backend() as fake:
        provider = _initialize(fake, "work")
        answer = _reply("We ship on Fridays.")
        assert recall_notice.notice_text("work") in answer
        fake.calls.clear()
        provider.sync_turn("when do we ship?", answer, session_id="s-1")
        assert fake.wait("remember_session")
    stored = json.dumps(fake.calls, default=str)
    assert "recall stays on" not in stored
    assert "We ship on Fridays." in stored
