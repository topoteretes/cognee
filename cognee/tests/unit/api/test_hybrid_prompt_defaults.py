from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO
from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO


def test_http_payloads_do_not_override_retriever_system_prompt_by_default():
    assert RecallPayloadDTO(query="q").system_prompt is None
    assert SearchPayloadDTO(query="q").system_prompt is None
