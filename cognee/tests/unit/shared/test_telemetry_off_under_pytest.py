"""A pytest session never reports telemetry (gh #5702).

``cognee/tests/conftest.py`` pins ``TELEMETRY_DISABLED`` before importing
cognee; without it, a plain ``pytest`` run counted as an install and the
suite's deliberately failing tasks as production task errors. A test that
wants telemetry on clears the variable itself, as ``test_telemetry_identity``
does, so this only pins the default.
"""

import os
import uuid

from cognee.shared import utils
from cognee.shared.utils import send_telemetry


def test_pytest_session_has_telemetry_disabled():
    assert os.environ.get("TELEMETRY_DISABLED")


def test_send_telemetry_is_a_no_op_under_pytest(monkeypatch):
    sent = []
    monkeypatch.setattr(utils, "_send_telemetry_request", lambda payload: sent.append(payload))
    # Even with ENV pointing at a reporting environment, the pinned default wins.
    monkeypatch.setenv("ENV", "prod")

    send_telemetry("cognee.remember", user_id=uuid.uuid4())

    assert sent == []
