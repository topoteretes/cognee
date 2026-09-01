"""CaptureConfig (SDK-529): env-named fields, derived directory, validation, and
the import-cost guarantee that ``import cognee`` never loads the capture package."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cognee.base_config import get_base_config
from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    CaptureConfig,
    get_capture_config,
    hook,
    prompt_fingerprint,
)

pytestmark = pytest.mark.usefixtures("capture_reset")

REPO_ROOT = Path(__file__).resolve().parents[5]

CAPTURE_ENV_VARS = (
    "COGNEE_CAPTURE_ENABLED",
    "COGNEE_CAPTURE_DIR",
    "COGNEE_CAPTURE_QUEUE_SIZE",
    "COGNEE_CAPTURE_BATCH_SIZE",
    "COGNEE_CAPTURE_FLUSH_INTERVAL_S",
    "COGNEE_CAPTURE_SAMPLE_RATE",
    "COGNEE_CAPTURE_SINK_TIMEOUT_S",
)


@pytest.fixture
def clean_capture_env(monkeypatch):
    for name in CAPTURE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_and_derived_dir(clean_capture_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path))
    get_base_config.cache_clear()
    try:
        config = CaptureConfig()
    finally:
        get_base_config.cache_clear()

    assert config.cognee_capture_enabled is False
    assert config.cognee_capture_queue_size == 512
    assert config.cognee_capture_batch_size == 64
    assert config.cognee_capture_flush_interval_s == 2.0
    assert config.cognee_capture_sample_rate == 1.0
    assert config.cognee_capture_sink_timeout_s == 30.0
    assert config.cognee_capture_dir == os.path.join(str(tmp_path), "capture")
    assert config.to_dict() == {
        "cognee_capture_enabled": False,
        "cognee_capture_dir": os.path.join(str(tmp_path), "capture"),
        "cognee_capture_queue_size": 512,
        "cognee_capture_batch_size": 64,
        "cognee_capture_flush_interval_s": 2.0,
        "cognee_capture_sample_rate": 1.0,
        "cognee_capture_sink_timeout_s": 30.0,
    }


def test_env_vars_populate_fields(clean_capture_env, monkeypatch):
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_DIR", "/tmp/somewhere")
    monkeypatch.setenv("COGNEE_CAPTURE_QUEUE_SIZE", "8")
    monkeypatch.setenv("COGNEE_CAPTURE_BATCH_SIZE", "2")
    monkeypatch.setenv("COGNEE_CAPTURE_FLUSH_INTERVAL_S", "0.5")
    monkeypatch.setenv("COGNEE_CAPTURE_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK_TIMEOUT_S", "7.5")
    get_capture_config.cache_clear()

    config = get_capture_config()

    assert config.cognee_capture_enabled is True
    assert config.cognee_capture_dir == "/tmp/somewhere"
    assert config.cognee_capture_queue_size == 8
    assert config.cognee_capture_batch_size == 2
    assert config.cognee_capture_flush_interval_s == 0.5
    assert config.cognee_capture_sample_rate == 0.25
    assert config.cognee_capture_sink_timeout_s == 7.5
    assert get_capture_config() is config  # cached


def test_sample_rate_out_of_range_raises(clean_capture_env, monkeypatch):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\], got 1\.5"):
        CaptureConfig(cognee_capture_sample_rate=1.5)

    monkeypatch.setenv("COGNEE_CAPTURE_SAMPLE_RATE", "-0.1")
    get_capture_config.cache_clear()
    with pytest.raises(ValueError, match=r"must be in \[0, 1\], got -0\.1"):
        get_capture_config()


def test_sink_timeout_must_be_positive(clean_capture_env):
    with pytest.raises(ValueError, match=r"SINK_TIMEOUT_S must be positive, got 0\.0"):
        CaptureConfig(cognee_capture_sink_timeout_s=0.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cognee_capture_queue_size", 0, r"QUEUE_SIZE must be at least 1, got 0"),
        ("cognee_capture_batch_size", 0, r"BATCH_SIZE must be at least 1, got 0"),
        ("cognee_capture_batch_size", -3, r"BATCH_SIZE must be at least 1, got -3"),
        ("cognee_capture_flush_interval_s", 0.0, r"FLUSH_INTERVAL_S must be positive, got 0\.0"),
        ("cognee_capture_flush_interval_s", -1.0, r"FLUSH_INTERVAL_S must be positive, got -1\.0"),
    ],
)
def test_degenerate_queue_knobs_are_rejected(clean_capture_env, field, value, message):
    # A batch size of 0 made the flusher pop nothing and spin without yielding.
    with pytest.raises(ValueError, match=message):
        CaptureConfig(**{field: value})


def test_bad_batch_size_in_env_disables_capture_instead_of_freezing(clean_capture_env, monkeypatch):
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_BATCH_SIZE", "0")
    get_capture_config.cache_clear()

    # The bad knob is rejected at initialization: capture stays off and the
    # cached knobs keep their defaults, so the observed operation is unaffected.
    assert capture.is_active() is False
    assert hook.BATCH_SIZE == 64


def test_prompt_fingerprints():
    assert prompt_fingerprint("hello") == "sha256:2cf24dba5fb0a30e"
    assert prompt_fingerprint("hello") != prompt_fingerprint("hello!")
    assert prompt_fingerprint("") == prompt_fingerprint("")


def test_import_cognee_does_not_import_capture_package():
    probe = (
        "import sys\n"
        "import cognee\n"
        "print('CAPTURE_LOADED=' + str('cognee.modules.observability.capture' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "CAPTURE_LOADED=False" in result.stdout
