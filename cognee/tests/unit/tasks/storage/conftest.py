"""Storage-task test fixtures.

The eval-capture fixtures (SDK-529) — ``capture_reset`` and ``fake_capture_sink`` —
are defined in the pipeline-runner conftest under ``tests/unit/modules/``, which
pytest does not apply to this directory; re-exporting them here makes them
available to the ``add_data_points`` tests without a second copy of the
loop-ordering subtleties documented on ``capture_reset``.
"""

from cognee.tests.unit.modules.conftest import capture_reset, fake_capture_sink  # noqa: F401
