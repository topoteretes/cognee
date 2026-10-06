"""add() and remember() take the grouped ``dlt_config`` and still the bare DLT kwargs."""

import inspect

import pytest

pytest.importorskip("dlt")

from cognee.api.v1.add.add import add
from cognee.api.v1.remember.remember import _ADD_ONLY, RememberKwargs
from cognee.tasks.ingestion.dlt_config import DLT_OPTION_NAMES


def test_add_takes_dlt_config():
    assert "dlt_config" in inspect.signature(add).parameters


def test_remember_forwards_dlt_config_to_add():
    assert "dlt_config" in _ADD_ONLY
    assert "dlt_config" in RememberKwargs.__annotations__


def test_every_add_only_kwarg_is_an_add_parameter_or_a_dlt_option():
    """remember() forwards these blindly; add() must know each one."""
    accepted = set(inspect.signature(add).parameters) | DLT_OPTION_NAMES
    assert _ADD_ONLY <= accepted, _ADD_ONLY - accepted
