from cognee_community_connector_meta_ads.reconcile import reconcile_ids


def test_reconcile_deletes_only_after_complete_snapshot():
    deleted = []
    removed = reconcile_ids(
        {"1", "2", "3"},
        [[{"id": "1"}], [{"id": "3"}]],
        deleted.append,
    )
    assert removed == {"2"}
    assert deleted == [{"2"}]


def test_partial_listing_failure_never_deletes():
    deleted = []

    def pages():
        yield [{"id": "1"}]
        raise RuntimeError("page request failed")

    removed = reconcile_ids({"1", "2", "3"}, pages(), deleted.append)
    assert removed == set()
    assert deleted == []
