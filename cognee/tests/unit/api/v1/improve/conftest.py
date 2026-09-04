"""Shared harness for the improve() orchestrator tests.

Everything below the orchestrator is stubbed: dataset resolution, the
migrations gate, the operation record, the remote client, telemetry, tracing
and the graph capability probe. Tests then either swap ``DEFAULT_STAGES`` for
fake stages (orchestration tests) or patch the module a real stage calls into
(stage tests).
"""

import importlib
import types
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.improve import GraphCapabilities, ImproveConfig
from cognee.modules.improve.result import StageResult
from cognee.modules.improve.stage import BaseStage


class DummySpan:
    def __init__(self):
        self.attributes: Dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeStage(BaseStage):
    """A stage whose gate and run are scripted by the test."""

    def __init__(
        self,
        name: str,
        *,
        kind: str = "graph",
        fatal: bool = False,
        after=(),
        gate_reason: Optional[str] = None,
        run=None,
        calls: Optional[List[Any]] = None,
    ):
        self.name = name
        self.kind = kind
        self.fatal = fatal
        self.after = tuple(after)
        self.label = name
        self.summary = name
        self.effects = []
        self._gate_reason = gate_reason
        self._run = run
        self.calls = calls if calls is not None else []
        self.gate_calls = 0
        self.seen_inputs = []

    def gate(self, inputs):
        self.gate_calls += 1
        return self._gate_reason

    async def run(self, inputs):
        self.calls.append(self.name)
        self.seen_inputs.append(inputs)
        if self._run is None:
            return StageResult.completed(self.name, items=1)
        outcome = self._run(inputs)
        if hasattr(outcome, "__await__"):
            outcome = await outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ImproveHarness:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.improve_mod = importlib.import_module("cognee.api.v1.improve.improve")
        self.user = types.SimpleNamespace(id=uuid4(), tenant_id=None)
        self.dataset = types.SimpleNamespace(id=uuid4(), name="docs", owner_id=self.user.id)
        self.span = DummySpan()
        self.telemetry: List[dict] = []
        self.resolve_calls: List[Any] = []
        self.config = ImproveConfig()
        self.capabilities = GraphCapabilities.assume_supported("FakeAdapter")
        self._install()

    def _install(self):
        mp = self.monkeypatch
        improve_mod = self.improve_mod

        async def fake_resolve(dataset, user):
            self.resolve_calls.append(dataset)
            return user, [self.dataset]

        mp.setattr(improve_mod, "resolve_authorized_user_datasets", fake_resolve)
        mp.setattr(improve_mod, "new_span", lambda _name: self.span)
        mp.setattr(improve_mod, "get_improve_config", lambda: self.config)
        mp.setattr(
            improve_mod,
            "resolve_graph_capabilities",
            AsyncMock(side_effect=lambda *_a, **_k: self.capabilities),
        )

        @asynccontextmanager
        async def fake_record_operation(_name):
            from cognee.modules.operations.record_operation import OperationContext

            yield OperationContext(_name)

        mp.setattr(improve_mod, "record_operation", fake_record_operation)

        state_mod = importlib.import_module("cognee.api.v1.serve.state")
        mp.setattr(state_mod, "get_remote_client", lambda: None)

        utils_mod = importlib.import_module("cognee.shared.utils")

        def fake_send_telemetry(event, user=None, additional_properties=None, **_kwargs):
            self.telemetry.append(
                {"event": event, "user": user, "properties": dict(additional_properties or {})}
            )

        mp.setattr(utils_mod, "send_telemetry", fake_send_telemetry)

        startup_mod = importlib.import_module("cognee.modules.migrations.startup")
        mp.setattr(startup_mod, "run_migrations_and_block", AsyncMock(return_value=None))

    def use_stages(self, stages):
        self.monkeypatch.setattr(self.improve_mod, "DEFAULT_STAGES", list(stages))
        return stages

    def set_config(self, **overrides):
        self.config = ImproveConfig(**overrides)
        return self.config

    def set_capabilities(self, **fields):
        base = self.capabilities.model_dump()
        base.update(fields)
        self.capabilities = GraphCapabilities(**base)
        return self.capabilities

    async def improve(self, **kwargs):
        kwargs.setdefault("user", self.user)
        kwargs.setdefault("dataset", "docs")
        return await self.improve_mod.improve(**kwargs)


@pytest.fixture
def harness(monkeypatch):
    return ImproveHarness(monkeypatch)
