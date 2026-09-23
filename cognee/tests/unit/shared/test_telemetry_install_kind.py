"""``install_kind`` on every telemetry event, and the exception-type rule (SDK-775)."""

from collections.abc import Coroutine
from typing import Any

import pytest

from cognee.shared import utils
from cognee.shared.utils import get_install_kind, send_telemetry, telemetry_exception_type


@pytest.fixture(autouse=True)
def _fresh_install_kind(monkeypatch, tmp_path):
    get_install_kind.cache_clear()
    monkeypatch.delenv(utils.INSTALL_KIND_ENV, raising=False)
    monkeypatch.setattr(utils, "_CONTAINER_MARKER", str(tmp_path / "no-container"))
    monkeypatch.setattr(utils, "is_source_checkout", lambda: False)
    yield
    get_install_kind.cache_clear()


def test_explicit_kind_from_our_images_wins(monkeypatch):
    monkeypatch.setenv(utils.INSTALL_KIND_ENV, "docker")
    monkeypatch.setattr(utils, "is_source_checkout", lambda: True)

    assert get_install_kind() == "docker"


def test_unknown_explicit_value_is_not_echoed(monkeypatch):
    monkeypatch.setenv(utils.INSTALL_KIND_ENV, "my-k8s-cluster")

    assert get_install_kind() == "package"


def test_container_marker_means_docker(monkeypatch, tmp_path):
    marker = tmp_path / "dockerenv"
    marker.write_text("")
    monkeypatch.setattr(utils, "_CONTAINER_MARKER", str(marker))
    monkeypatch.setattr(utils, "is_source_checkout", lambda: True)

    assert get_install_kind() == "docker"


def test_source_tree_means_git(monkeypatch):
    monkeypatch.setattr(utils, "is_source_checkout", lambda: True)

    assert get_install_kind() == "git"


def test_installed_package_is_the_default():
    assert get_install_kind() == "package"


def test_every_event_carries_install_kind(monkeypatch):
    payloads: list[dict[str, Any]] = []

    def capture_payload(payload: dict[str, Any]) -> Coroutine[Any, Any, None]:
        payloads.append(payload)

        async def noop() -> None:
            return None

        return noop()

    class FakeTask:
        def add_done_callback(self, callback) -> None:
            callback(self)

    class CapturingLoop:
        def create_task(self, coroutine: Coroutine[Any, Any, None]) -> "FakeTask":
            coroutine.close()
            return FakeTask()

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils, "_send_telemetry_request", capture_payload)
    monkeypatch.setattr(utils.asyncio, "get_running_loop", lambda: CapturingLoop())

    send_telemetry("test_event", "user-123", {})

    assert payloads[0]["properties"]["install_kind"] == "package"
    assert payloads[0]["properties"]["install_kind"] in utils.INSTALL_KINDS


def test_exception_type_is_the_class_name_never_the_message():
    root = ValueError("Dataset 'customer-secrets' not found.")
    wrapper = RuntimeError("Pipeline run failed.")
    wrapper.first_error = root

    assert telemetry_exception_type(root) == "ValueError"
    assert telemetry_exception_type(wrapper) == "ValueError"
    assert telemetry_exception_type(RuntimeError("plain")) == "RuntimeError"
