import pytest
from pydantic import BaseModel
from cognee.modules.retrieval.context_preview import (
    CONTEXT_FORMAT_CONTEXT,
    CONTEXT_FORMAT_PROMPT,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types.SearchType import SearchType


class DealBrief(BaseModel):
    deal_name: str = ""
    health: str = ""


def test_search_result_payload_with_string_completion():
    """Test that normal string completion still works."""
    payload = SearchResultPayload(
        completion="a normal string answer", search_type=SearchType.GRAPH_COMPLETION
    )
    assert payload.completion == "a normal string answer"


def test_search_result_payload_with_pydantic_model():
    """Test that Pydantic BaseModel is accepted (core bug fix)."""
    deal = DealBrief(deal_name="Acme Corp", health="Good")
    payload = SearchResultPayload(completion=deal, search_type=SearchType.GRAPH_COMPLETION)

    assert isinstance(payload.completion, DealBrief)
    assert payload.completion.deal_name == "Acme Corp"
    assert payload.model_dump()["completion"] == {"deal_name": "Acme Corp", "health": "Good"}


def test_search_result_payload_with_list_of_models():
    """Test list of Pydantic models."""
    deals = [DealBrief(deal_name="Deal 1"), DealBrief(deal_name="Deal 2")]
    payload = SearchResultPayload(completion=deals, search_type=SearchType.GRAPH_COMPLETION)
    assert isinstance(payload.completion, list)
    assert len(payload.completion) == 2
    assert isinstance(payload.completion[0], DealBrief)
    assert payload.completion[0].deal_name == "Deal 1"


def test_search_result_payload_only_context():
    """Test only_context flag behavior."""
    payload = SearchResultPayload(
        context="Some context here", only_context=True, search_type=SearchType.GRAPH_COMPLETION
    )
    assert payload.result == "Some context here"


def test_search_result_payload_only_context_default_format_is_unchanged():
    """The historical shape is the default: a bare context, no envelope."""
    payload = SearchResultPayload(
        context="Some context here",
        only_context=True,
        question="why?",
        session_context="## Active session guidance\n- be terse",
        user_prompt="The question is: `why?`",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload.context_format == CONTEXT_FORMAT_CONTEXT
    assert payload.result == "Some context here"


def test_search_result_payload_prompt_format_returns_the_envelope():
    payload = SearchResultPayload(
        context="Some context here",
        only_context=True,
        context_format=CONTEXT_FORMAT_PROMPT,
        question="why?",
        session_context="## Active session guidance\n- be terse",
        user_prompt="The question is: `why?`",
        system_prompt="history\nTASK:answer",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload.result == {
        "question": "why?",
        "context": "Some context here",
        "session_context": "## Active session guidance\n- be terse",
        "user_prompt": "The question is: `why?`",
        "system_prompt": "history\nTASK:answer",
    }


def test_search_result_payload_prompt_format_ignored_without_only_context():
    """context_format shapes only_context results; a real completion still wins."""
    payload = SearchResultPayload(
        completion=["answer"],
        context="Some context here",
        context_format=CONTEXT_FORMAT_PROMPT,
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload.result == ["answer"]


def test_search_result_payload_with_plain_dict():
    """A plain dict completion validates as a dict — it must not be coerced
    into an empty bare BaseModel (which would silently drop every field)."""
    payload = SearchResultPayload(
        completion={"deal_name": "Acme Corp", "health": "Good"},
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload.completion == {"deal_name": "Acme Corp", "health": "Good"}
    assert payload.model_dump()["completion"] == {"deal_name": "Acme Corp", "health": "Good"}


def test_search_result_payload_model_json_round_trip():
    """model_dump_json must carry subclass fields, not bare-BaseModel emptiness."""
    import json

    deal = DealBrief(deal_name="Acme Corp", health="Good")
    payload = SearchResultPayload(completion=deal, search_type=SearchType.GRAPH_COMPLETION)
    dumped = json.loads(payload.model_dump_json())
    assert dumped["completion"] == {"deal_name": "Acme Corp", "health": "Good"}
