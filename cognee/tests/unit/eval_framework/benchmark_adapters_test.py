import pytest
from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from cognee.eval_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter
from cognee.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from cognee.eval_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from unittest.mock import patch, mock_open


MOCK_JSONL_DATA = """\
{"id": "1", "question": "What is AI?", "answer": "Artificial Intelligence", "paragraphs": [{"paragraph_text": "AI is a field of computer science."}]}
{"id": "2", "question": "What is ML?", "answer": "Machine Learning", "paragraphs": [{"paragraph_text": "ML is a subset of AI."}]}
"""


ADAPTER_CLASSES = [
    HotpotQAAdapter,
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
