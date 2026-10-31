"""``get_dataset_custom_prompt`` returns the stored prompt, and None for a blank one."""

import sys
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from cognee.modules.data.methods import get_dataset_custom_prompt

# The package __init__ rebinds the attribute to the function, so the module
# itself must come from sys.modules.
module = sys.modules["cognee.modules.data.methods.get_dataset_custom_prompt"]


class _FakeEngine:
    def __init__(self, stored_prompt):
        self._stored_prompt = stored_prompt
        self.statements = []

    @asynccontextmanager
    async def get_async_session(self):
        engine = self

        class _Session:
            async def scalar(self, statement):
                engine.statements.append(statement)
                return engine._stored_prompt

        yield _Session()


@pytest.mark.asyncio
async def test_returns_the_stored_prompt(monkeypatch):
    engine = _FakeEngine("Extract companies and deals.")
    monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
    dataset_id = uuid4()

    assert await get_dataset_custom_prompt(dataset_id) == "Extract companies and deals."
    assert dataset_id in engine.statements[0].compile().params.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_prompt", [None, "", "   "])
async def test_no_or_blank_prompt_returns_none(monkeypatch, stored_prompt):
    monkeypatch.setattr(module, "get_relational_engine", lambda: _FakeEngine(stored_prompt))

    assert await get_dataset_custom_prompt(uuid4()) is None
