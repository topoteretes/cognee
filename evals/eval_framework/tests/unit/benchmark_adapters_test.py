import pytest
import random
from evals.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from evals.eval_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter
from evals.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from evals.eval_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter


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
    adapter = AdapterClass()
    result = adapter.load_corpus()

    assert isinstance(result, tuple), f"{AdapterClass.__name__} did not return a tuple."
    assert len(result) == 2, f"{AdapterClass.__name__} returned tuple of length != 2."

    corpus_list, qa_pairs = result
    assert isinstance(corpus_list, list), f"{AdapterClass.__name__} corpus_list is not a list."
    assert isinstance(qa_pairs, list), f"{AdapterClass.__name__} question_answer_pairs is not a list."


@pytest.mark.parametrize("AdapterClass", ADAPTER_CLASSES)
def test_adapter_returns_some_content(AdapterClass):
    """
    Verify that the adapter returns some data and that each QA dict
    at least has a 'question' and 'answer' key (you can extend or remove as needed).
    """
    adapter = AdapterClass()

    corpus_list, qa_pairs = adapter.load_corpus(limit=3)  # small limit
    # We don't know how large the dataset is, but we expect at least 1 item
    assert len(corpus_list) > 0, f"{AdapterClass.__name__} returned an empty corpus_list."
    assert len(qa_pairs) > 0, f"{AdapterClass.__name__} returned an empty question_answer_pairs."

    # Check the shape
    assert len(corpus_list) == len(qa_pairs), (
        f"{AdapterClass.__name__} corpus_list and question_answer_pairs "
        "should typically be the same length. Adjust if your adapter differs."
    )

    for item in qa_pairs:
        assert "question" in item, f"{AdapterClass.__name__} missing 'question' key in QA pair."
        assert "answer" in item, f"{AdapterClass.__name__} missing 'answer' key in QA pair."


@pytest.mark.parametrize("AdapterClass", ADAPTER_CLASSES)
def test_adapter_limit(AdapterClass):
    """
    Check that the `limit` parameter correctly restricts the amount of data returned.
    We'll test with limit=5.
    """
    adapter = AdapterClass()

    limit = 5
    corpus_list, qa_pairs = adapter.load_corpus(limit=limit)

    # Confirm that we didn't receive more than 'limit'
    # (Some adapters might be allowed to return fewer if the dataset is small)
    assert len(corpus_list) <= limit, (
        f"{AdapterClass.__name__} returned more items than requested limit={limit}."
    )
    assert len(qa_pairs) <= limit, (
        f"{AdapterClass.__name__} returned more QA items than requested limit={limit}."
    )
