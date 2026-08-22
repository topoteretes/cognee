from importlib import import_module
from uuid import uuid4

import pytest


configuration_module = import_module("cognee.modules.users.methods.get_principal_configuration")


class _Result:
    class _Scalars:
        @staticmethod
        def first():
            return None

    @staticmethod
    def scalars():
        return _Result._Scalars()


class _Session:
    statement = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def execute(self, statement):
        self.statement = statement
        return _Result()


class _Engine:
    def __init__(self, session):
        self.session = session

    def get_async_session(self):
        return self.session


@pytest.mark.asyncio
async def test_configuration_query_is_scoped_to_principal(monkeypatch):
    config_id = uuid4()
    principal_id = uuid4()
    session = _Session()
    monkeypatch.setattr(
        configuration_module,
        "get_relational_engine",
        lambda: _Engine(session),
    )

    result = await configuration_module.get_principal_configuration(
        config_id=config_id,
        principal_id=principal_id,
    )

    assert result == {}
    compiled = session.statement.compile()
    assert "principal_configuration.id =" in str(compiled)
    assert "principal_configuration.owner_id =" in str(compiled)
    assert set(compiled.params.values()) == {config_id, principal_id}
