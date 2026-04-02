import pytest
from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from cognee.eval_framework.benchmark_adapters.logistics_system_adapter import (
    LogisticsSystemAdapter,
)
from cognee.eval_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter
from cognee.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from cognee.eval_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from unittest.mock import patch, mock_open


MOCK_JSONL_DATA = """\
{"id": "1", "question": "What is AI?", "answer": "Artificial Intelligence", "paragraphs": [{"paragraph_text": "AI is a field of computer science."}]}
{"id": "2", "question": "What is ML?", "answer": "Machine Learning", "paragraphs": [{"paragraph_text": "ML is a subset of AI."}]}
"""

MOCK_HOTPOT_CORPUS = [
    {
        "_id": "1",
        "question": "Next to which country is Germany located?",
        "answer": "Netherlands",
        # HotpotQA uses "level"; TwoWikiMultiHop uses "type".
        "level": "easy",
        "type": "comparison",
        "context": [
            ["Germany", ["Germany is in Europe."]],
            ["Netherlands", ["The Netherlands borders Germany."]],
        ],
        "supporting_facts": [["Netherlands", 0]],
    }
]


ADAPTER_CLASSES = [
    HotpotQAAdapter,
    LogisticsSystemAdapter,
    MusiqueQAAdapter,
    DummyAdapter,
    TwoWikiMultihopAdapter,
]


@pytest.mark.parametrize("AdapterClass", ADAPTER_CLASSES)
def test_adapter_can_instantiate_and_load(AdapterClass):
    """
    Basic smoke test: instantiate each adapter, call load_corpus with no limit,
    and ensure it returns the expected tuple of (list, list).
    """
    if AdapterClass == MusiqueQAAdapter:
        with (
            patch.object(MusiqueQAAdapter, "_musique_download_file"),
            patch("builtins.open", new_callable=mock_open, read_data=MOCK_JSONL_DATA),
            patch("os.path.exists", return_value=True),
        ):
            adapter = AdapterClass()
            result = adapter.load_corpus()

    elif AdapterClass in (HotpotQAAdapter, TwoWikiMultihopAdapter):
        with patch.object(AdapterClass, "_get_raw_corpus", return_value=MOCK_HOTPOT_CORPUS):
            adapter = AdapterClass()
            result = adapter.load_corpus()

    else:
        adapter = AdapterClass()
        result = adapter.load_corpus()

    assert isinstance(result, tuple), f"{AdapterClass.__name__} did not return a tuple."
    assert len(result) == 2, f"{AdapterClass.__name__} returned tuple of length != 2."

    corpus_list, qa_pairs = result
    assert isinstance(corpus_list, list), f"{AdapterClass.__name__} corpus_list is not a list."
    assert isinstance(qa_pairs, list), (
        f"{AdapterClass.__name__} question_answer_pairs is not a list."
    )


@pytest.mark.parametrize("AdapterClass", ADAPTER_CLASSES)
def test_adapter_returns_some_content(AdapterClass):
    """
    Verify that the adapter returns some data and that each QA dict
    at least has a 'question' and 'answer' key (you can extend or remove as needed).
    """
    limit = 3
    if AdapterClass == MusiqueQAAdapter:
        with (
            patch.object(MusiqueQAAdapter, "_musique_download_file"),
            patch("builtins.open", new_callable=mock_open, read_data=MOCK_JSONL_DATA),
            patch("os.path.exists", return_value=True),
        ):
            adapter = AdapterClass()
            corpus_list, qa_pairs = adapter.load_corpus(limit=limit)
    elif AdapterClass in (HotpotQAAdapter, TwoWikiMultihopAdapter):
        with patch.object(AdapterClass, "_get_raw_corpus", return_value=MOCK_HOTPOT_CORPUS):
            adapter = AdapterClass()
            corpus_list, qa_pairs = adapter.load_corpus(limit=limit)
    else:
        adapter = AdapterClass()
        corpus_list, qa_pairs = adapter.load_corpus(limit=limit)

    # We don't know how large the dataset is, but we expect at least 1 item
    assert len(corpus_list) > 0, f"{AdapterClass.__name__} returned an empty corpus_list."
    assert len(qa_pairs) > 0, f"{AdapterClass.__name__} returned an empty question_answer_pairs."
    assert len(qa_pairs) <= limit, (
        f"{AdapterClass.__name__} returned more QA items than requested limit={limit}."
    )

    for item in qa_pairs:
        assert "question" in item, f"{AdapterClass.__name__} missing 'question' key in QA pair."
        assert "answer" in item, f"{AdapterClass.__name__} missing 'answer' key in QA pair."


def test_logistics_adapter_creates_world_when_missing(tmp_path):
    adapter = LogisticsSystemAdapter(
        world_name="test_world",
        worlds_root=tmp_path,
        user_count=3,
        retailer_count=2,
        package_count=2,
    )

    corpus_list, qa_pairs = adapter.load_corpus(load_golden_context=True)

    assert adapter.world_file.exists()
    assert len(list(tmp_path.rglob("*.txt"))) > 0
    assert len(corpus_list) > 0
    assert len(qa_pairs) == 6
    assert all(item["world_name"] == "test_world" for item in qa_pairs)
    assert all("golden_context" in item for item in qa_pairs)
    assert all("golden_context_data_sources" in item for item in qa_pairs)
    assert all("package_context" in item for item in qa_pairs)
    assert {item["question_type"] for item in qa_pairs} == {
        "carrier",
        "delivery_days",
        "transport_cost",
    }

    answers_by_type = {item["question_type"]: item for item in qa_pairs[:3]}
    assert isinstance(answers_by_type["delivery_days"]["answer"], str)
    assert isinstance(answers_by_type["transport_cost"]["answer"], str)
    assert isinstance(answers_by_type["carrier"]["answer"], str)
    assert isinstance(answers_by_type["delivery_days"]["golden_answer"], (int, type(None)))
    assert isinstance(answers_by_type["transport_cost"]["golden_answer"], (float, int, type(None)))
    assert isinstance(answers_by_type["delivery_days"]["golden_context"], str)


def test_logistics_adapter_loads_existing_world(tmp_path):
    adapter = LogisticsSystemAdapter(
        world_name="persisted_world",
        worlds_root=tmp_path,
        user_count=3,
        retailer_count=2,
        package_count=2,
    )
    _, first_qa_pairs = adapter.load_corpus()
    first_world_contents = adapter.world_file.read_text(encoding="utf-8")

    second_adapter = LogisticsSystemAdapter(
        world_name="persisted_world",
        worlds_root=tmp_path,
        user_count=10,
        retailer_count=10,
        package_count=10,
    )
    _, second_qa_pairs = second_adapter.load_corpus()
    second_world_contents = second_adapter.world_file.read_text(encoding="utf-8")

    assert first_world_contents == second_world_contents
    assert [item["id"] for item in first_qa_pairs] == [item["id"] for item in second_qa_pairs]
