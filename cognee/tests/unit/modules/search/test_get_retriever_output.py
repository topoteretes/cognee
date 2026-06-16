from cognee.modules.search.methods.get_retriever_output import _count_retrieved_objects


def test_count_retrieved_objects_counts_structured_lists():
    assert _count_retrieved_objects({"chunks": [1, 2], "entities": [3]}) == 3


def test_count_retrieved_objects_preserves_existing_shapes():
    assert _count_retrieved_objects(None) == 0
    assert _count_retrieved_objects(["a", "b"]) == 2
    assert _count_retrieved_objects({"triplets": []}) == 0
    assert _count_retrieved_objects({"metadata": "value"}) == 1
    assert _count_retrieved_objects("answer") == 1
