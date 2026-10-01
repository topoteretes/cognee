import pytest
from pydantic import BaseModel, ValidationError

from cognee.modules.search.models.EvidenceReference import EvidenceReference
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


def test_search_result_payload_only_context_returns_the_user_prompt_when_one_was_built():
    """The user prompt replaces the bare context as the result; the system prompt rides on
    its own field and the context stays."""
    payload = SearchResultPayload(
        context="Some context here",
        only_context=True,
        user_prompt="The question is: `why?` ... Some context here",
        system_prompt="history\nTASK:answer",
        search_type=SearchType.GRAPH_COMPLETION,
    )
    assert payload.result == payload.user_prompt
    assert payload.system_prompt == "history\nTASK:answer"
    assert payload.context == "Some context here"


def test_search_result_payload_only_context_without_a_prompt_returns_the_context():
    """Retrieval-only types never build a prompt, so their shape is unchanged."""
    payload = SearchResultPayload(
        context=["chunk-a", "chunk-b"], only_context=True, search_type=SearchType.CHUNKS
    )
    assert payload.user_prompt is None
    assert payload.system_prompt is None
    assert payload.result == ["chunk-a", "chunk-b"]


def test_search_result_payload_prompt_is_ignored_without_only_context():
    """The prompt only stands in for a completion that was not generated."""
    payload = SearchResultPayload(
        completion=["answer"],
        context="Some context here",
        prompt="SYSTEM:\nTASK:answer\n\nUSER:\nq",
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


def test_search_result_payload_serializes_structured_evidence():
    payload = SearchResultPayload(
        completion="answer",
        evidence=[
            EvidenceReference(
                kind="segment",
                artifact_id="chunk-1",
                data_id="data-1",
                chunk_id="chunk-1",
                rank=0,
            )
        ],
        search_type=SearchType.RAG_COMPLETION,
    )

    assert payload.evidence[0].role == "used_as_context"
    assert payload.model_dump(mode="json")["evidence"] == [
        {
            "kind": "segment",
            "artifact_id": "chunk-1",
            "role": "used_as_context",
            "dataset_id": None,
            "source_ref_key": None,
            "data_id": "data-1",
            "chunk_id": "chunk-1",
            "chunk_index": None,
            "document_name": None,
            "source_node_id": None,
            "target_node_id": None,
            "relationship_name": None,
            "assertion_id": None,
            "label": None,
            "rank": 0,
            "score": None,
        }
    ]
