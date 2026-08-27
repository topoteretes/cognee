import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.search.types import ContextFormat


def test_context_format_values_are_the_two_documented_shapes():
    assert [member.value for member in ContextFormat] == ["context", "prompt"]


def test_parse_accepts_strings_members_and_none():
    assert ContextFormat.parse("prompt") is ContextFormat.PROMPT
    assert ContextFormat.parse(ContextFormat.CONTEXT) is ContextFormat.CONTEXT
    assert ContextFormat.parse(None) is ContextFormat.CONTEXT


@pytest.mark.parametrize("bad", ["Prompt", "bogus", "", 1])
def test_parse_rejects_anything_else_with_one_shared_error(bad):
    """Every entry point raises this same name and wording for the same input."""
    with pytest.raises(CogneeValidationError) as excinfo:
        ContextFormat.parse(bad)
    assert excinfo.value.name == "InvalidContextFormatError"
    assert "context_format" in excinfo.value.message
    assert "['context', 'prompt']" in excinfo.value.message
