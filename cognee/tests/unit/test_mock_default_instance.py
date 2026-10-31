"""The generic mock fallback must construct ANY response model, not just the
ones with all-default fields.

The example jobs on the PR gate run through install_mocks({}) (see
cognee/tests/utils/run_mocked.py): every structured-output call not covered
by the replay map returns a default instance of the requested model. The
memify example broke the old ``response_model()`` fallback with a required
field ("Field required" ValidationError); these tests pin the synthesis.
"""

import enum
from typing import Literal, Optional
from uuid import UUID

import pytest
from pydantic import BaseModel

from cognee.tests.utils.mock_ingestion.mock_ingestion import _default_instance


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class _Inner(BaseModel):
    name: str
    weight: float


class _Demanding(BaseModel):
    """Required fields of every shape the examples' models use."""

    title: str
    count: int
    ratio: float
    flag: bool
    ident: UUID
    kind: Literal["a", "b"]
    color: _Color
    inner: _Inner
    items: list[str]
    mapping: dict[str, int]
    maybe: str | None = None


def test_all_defaults_model_uses_plain_construction():
    class Easy(BaseModel):
        note: str = "n"

    assert _default_instance(Easy).note == "n"


def test_required_fields_are_synthesized_not_raised():
    obj = _default_instance(_Demanding)
    assert isinstance(obj, _Demanding)
    assert obj.title == "" and obj.count == 0 and obj.ratio == 0.0 and obj.flag is False
    assert isinstance(obj.ident, UUID)
    assert obj.kind == "a"
    assert obj.color is _Color.RED
    assert isinstance(obj.inner, _Inner) and obj.inner.name == ""
    assert obj.items == [] and obj.mapping == {}
    assert obj.maybe is None


def test_scalars_and_str():
    assert _default_instance(str) == ""
    assert _default_instance(int) == 0
    assert _default_instance(bool) is False


@pytest.mark.asyncio
async def test_install_mocks_empty_map_serves_any_model(monkeypatch):
    """End to end: after install_mocks({}), the gateway returns a valid
    instance for a required-field model with zero network calls."""
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.tests.utils.mock_ingestion import install_mocks

    original = LLMGateway.acreate_structured_output
    try:
        install_mocks({}, mock_embeddings=False)
        result = await LLMGateway.acreate_structured_output("x", "y", _Demanding)
        assert isinstance(result, _Demanding)
        assert await LLMGateway.acreate_structured_output("x", "y", str) == ""
    finally:
        LLMGateway.acreate_structured_output = original
