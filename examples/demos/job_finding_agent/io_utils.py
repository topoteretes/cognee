"""I/O helpers for demo inputs."""

from __future__ import annotations

import json
from pathlib import Path

from .agent.agent_models import JobFeedbackTriplet


def read_mock_jobs(path: Path) -> list[JobFeedbackTriplet]:
    """Load mocked jobs with feedback branches from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("mock_jobs.json must be a list")
    return [JobFeedbackTriplet.model_validate(item) for item in payload]

