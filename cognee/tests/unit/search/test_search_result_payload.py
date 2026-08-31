import pytest
from pydantic import BaseModel
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


def test_search_result_payload_complex_serialization():
    """Test that result_object handles complex/nested models and UUIDs correctly."""
    from uuid import UUID
    import json

    deal = DealBrief(deal_name="Acme Corp", health="Good")
    nested_data = {
        "deals": [deal],
        "transaction_id": UUID("12345678-1234-5678-1234-567812345678"),
        "active": True,
    }

    payload = SearchResultPayload(
        result_object=nested_data,
        search_type=SearchType.GRAPH_COMPLETION,
    )

    # Test default snake_case output
    dumped = json.loads(payload.model_dump_json())
    assert dumped["result_object"]["deals"] == [{"deal_name": "Acme Corp", "health": "Good"}]
    assert dumped["result_object"]["transaction_id"] == "12345678-1234-5678-1234-567812345678"
    assert dumped["result_object"]["active"] is True

    # Test camelCase output when serialized by_alias=True
    dumped_alias = json.loads(payload.model_dump_json(by_alias=True))
    assert dumped_alias["resultObject"]["deals"] == [{"deal_name": "Acme Corp", "health": "Good"}]
    assert dumped_alias["resultObject"]["transaction_id"] == "12345678-1234-5678-1234-567812345678"
    assert dumped_alias["resultObject"]["active"] is True


def test_search_result_payload_falsy_completion():
    """Test that empty or falsy completions are returned correctly instead of falling back."""
    payload_empty_str = SearchResultPayload(
        completion="",
        context="Fallback context",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload_empty_str.result == ""

    payload_empty_list = SearchResultPayload(
        completion=[],
        context="Fallback context",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload_empty_list.result == []
