"""The ingest path, the incremental cursor, and forget-on-delete.

Nothing the text leaves out may move a fingerprint, and what the text says is frozen once
released, because it is part of a document's identity: each text opens with its kind, and
each row carries the node set ``pubmed:<resource_name>``. The window is listed in full
every run; only new PMIDs are fetched, and an id that left a complete listing is verified
once before it is forgotten. A run cut by a budget or a 429 keeps its state and resumes
without false deletes. The walk is driven with a plain dict as dlt resource state: no
network, no LLM.
"""

import pytest
from fake_pubmed import FakePubMed

from cognee_connector_pubmed.source import (
    EFETCH_PATH,
    ESEARCH_PATH,
    PAGE_SIZE,
    PubMedAPIError,
    PubMedAuthError,
    PubMedRateLimitedError,
    _hash,
    _iter_rows,
    parse_article,
    render_article,
)
from cognee_connectors_core.dlt_conventions import NODE_SET_COLUMN

END = "2026/06/01"
ROW_COLUMNS = {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN}
# Every count a run reports, whichever way it ends.
COUNT_KEYS = {"scanned", "skipped", "fetched", "deleted", "failed", "requests"}
ABSTRACT = ["Abstract: BACKGROUND: Sleep helps memory.", "RESULTS: It really does."]


def _sync(fake: FakePubMed, state: dict, **kwargs) -> tuple[list[dict], dict]:
    stats: dict[str, int] = {}
    state.setdefault("name", "test_query")
    rows = list(_iter_rows(fake, state, stats, end_date=END, **kwargs))
    return rows, stats


def _ids(rows: list[dict]) -> set[str]:
    return {row["id"] for row in rows}


def _tombstones(rows: list[dict]) -> set[str]:
    return {row["id"] for row in rows if row.get("_deleted")}


def _corpus() -> FakePubMed:
    fake = FakePubMed()
    fake.add_article("100", "Sleep and the consolidation of declarative memory")
    fake.add_article("101", "Amyloid clearance during slow-wave sleep", edat="2026/02/01")
    return fake


# --- the ingest path ---


def test_a_first_sync_emits_every_article_with_the_frozen_row_shape():
    rows, stats = _sync(_corpus(), {})

    assert _ids(rows) == {"pmid:100", "pmid:101"}
    assert all(set(row) == ROW_COLUMNS for row in rows)
    assert all(row["_deleted"] is False and row["title"] and row["content"] for row in rows)
    assert all(row[NODE_SET_COLUMN] == ["pubmed:test_query"] for row in rows)
    urls = {row["id"]: row["url"] for row in rows}
    assert urls["pmid:100"] == "https://pubmed.ncbi.nlm.nih.gov/100/"
    assert stats["failed"] == 0


def test_each_text_opens_with_a_sentence_that_names_its_kind():
    rows, _ = _sync(_corpus(), {})
    content = {row["id"]: row["content"] for row in rows}

    assert content["pmid:100"].splitlines() == [
        'PubMed article "Sleep and the consolidation of declarative memory".',
        "Authors: Smith J, Doe A",
        "Journal: Journal of Memory Studies",
        "Year: 2026",
        "DOI: 10.1000/example",
        "Publication types: Journal Article",
        "Keywords: sleep, memory",
        *ABSTRACT,
    ]


def test_every_row_carries_the_node_set_of_its_resource_name():
    rows, _ = _sync(_corpus(), {"name": "pubmed_alzheimers"})

    assert {tuple(row[NODE_SET_COLUMN]) for row in rows} == {("pubmed:pubmed_alzheimers",)}


def test_the_node_set_is_part_of_the_row_and_its_fingerprint():
    """The node set is hashed into the row at fetch time: two connections syncing the same
    article render different rows, and a stored document must be replaced by one carrying
    the new set, not skipped because its text did not change."""
    from fake_pubmed import article_xml

    article = parse_article(article_xml(_corpus().articles["100"]))

    row = render_article(article, ["pubmed:a"])
    same = render_article(article, ["pubmed:a"])
    other = render_article(article, ["pubmed:b"])

    assert row[NODE_SET_COLUMN] == ["pubmed:a"]
    assert _hash(row) == _hash(same)
    assert _hash(row) != _hash(other)


def test_a_collective_author_renders_as_a_name_and_an_empty_part_is_dropped():
    fake = FakePubMed()
    fake.add_article("100", "GWAS of sleep duration", authors=["Smith J", "Consortium Group"])
    fake.add_article("101", "No authors here", authors=[])

    rows, _ = _sync(fake, {})

    content = {row["id"]: row["content"] for row in rows}
    assert "Authors: Smith J, Consortium Group" in content["pmid:100"].splitlines()
    assert not any(line.startswith("Authors:") for line in content["pmid:101"].splitlines())


def test_parse_article_refuses_an_answer_without_an_article():
    with pytest.raises(PubMedAPIError, match="without an article"):
        parse_article("<PubmedArticleSet></PubmedArticleSet>")
    with pytest.raises(PubMedAPIError, match="unreadable"):
        parse_article("not xml at all")


def test_render_article_titles_an_article_without_a_title():
    row = render_article(
        {
            "pmid": "1",
            "title": "",
            "authors": [],
            "journal": "",
            "year": "",
            "doi": "",
            "types": [],
            "keywords": [],
            "abstract": "",
            "edat": "",
        },
        ["pubmed:test_query"],
    )

    assert row["title"] == "Untitled article"
    assert row["content"] == 'PubMed article "Untitled article".'


# --- the window the listing asks for ---


def test_the_listing_asks_for_the_edat_window_with_the_query_term():
    fake = _corpus()

    _sync(fake, {}, query="sleep[title/abstract]", start_date="2026/01/15")

    (_, params), *_ = [call for call in fake.calls if call[0] == ESEARCH_PATH]
    assert params["term"] == "sleep[title/abstract]"
    assert params["datetype"] == "edat"
    assert params["mindate"] == "2026/01/15"
    assert params["maxdate"] == END
    assert params["db"] == "pubmed"


def test_the_query_selects_what_the_window_reaches():
    fake = _corpus()
    fake.add_article(
        "102",
        "Cooking techniques of the Andes",
        edat="2026/01/05",
        abstract=[("", "Potatoes and quinoa at altitude.")],
    )

    rows, stats = _sync(fake, {}, query="sleep", start_date="2026/01/01")

    assert _ids(rows) == {"pmid:100", "pmid:101"}
    assert stats["fetched"] == 2


def test_a_start_date_after_an_articles_edat_leaves_it_out():
    rows, _ = _sync(_corpus(), {}, start_date="2026/01/15")

    assert _ids(rows) == {"pmid:101"}, "pmid 100 entered the index on 2026/01/01"


# --- the incremental cursor: the fingerprints ---


def test_an_unchanged_resync_lists_but_fetches_nothing():
    fake, state = _corpus(), {}
    _sync(fake, state)

    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["scanned"] == 2
    assert stats["skipped"] == 2
    assert stats["fetched"] == 0
    assert stats["requests"] == 1, "one esearch call; nothing needed efetch"


def test_a_new_article_is_the_only_row_of_the_next_run():
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.add_article("102", "Napping and creativity", edat="2026/03/01")
    rows, stats = _sync(fake, state)

    assert _ids(rows) == {"pmid:102"}
    assert stats["fetched"] == 1
    fetches = [params for path, params in fake.calls if path == EFETCH_PATH]
    assert fetches[-1]["id"] == "102", "the new article alone was fetched"


def test_fingerprints_and_edat_are_the_cursor_the_state_keeps():
    fake, state = _corpus(), {}
    _sync(fake, state)

    assert set(state["fingerprints"]) == {"pmid:100", "pmid:101"}
    assert state["edat"] == {"pmid:100": "2026/01/01", "pmid:101": "2026/02/01"}


def test_an_edit_is_not_re_synced_because_edat_does_not_move():
    """The documented limit: a correction does not re-date the record, so the listing
    still lists it, the fingerprint still matches, and nothing is re-fetched."""
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.articles["100"]["title"] = "Sleep and memory, corrected"
    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["skipped"] == 2


# --- forget-on-delete ---


def test_an_article_removed_from_the_index_is_tombstoned_after_a_check():
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.removed.add("100")
    rows, stats = _sync(fake, state)

    assert rows == [{"id": "pmid:100", "_deleted": True}]
    assert stats["deleted"] == 1
    assert fake.count(ESEARCH_PATH, with_uid=True) == 1
    assert "pmid:100" not in state["fingerprints"]


def test_an_article_hidden_from_the_listing_but_still_indexed_is_kept():
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.hidden.add("100")
    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["failed"] == 0
    assert fake.count(ESEARCH_PATH, with_uid=True) == 1
    assert "pmid:100" in state["fingerprints"]


def test_a_deletion_check_that_fails_keeps_the_record_and_counts():
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.hidden.add("100")
    fake.removed.add("101")
    # The listing passes, the first uid check passes, the second one errors.
    fake.fail_after = 2
    fake.fail_with = PubMedAPIError("PubMed request failed: HTTP 500")
    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["failed_deletion_check"] == 1
    assert stats["failed"] == 1
    assert "pmid:101" in state["fingerprints"], "the errored check kept the record"
    fake.fail_after = None

    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"pmid:101"}
    assert stats["failed"] == 0
    assert "pmid:100" in state["fingerprints"]


def test_a_narrowed_window_forgets_what_it_deselected_at_once():
    fake, state = _corpus(), {}
    _sync(fake, state, start_date="2026/01/01")

    rows, stats = _sync(fake, state, start_date="2026/01/15")

    assert _tombstones(rows) == {"pmid:100"}
    assert fake.count(ESEARCH_PATH, with_uid=True) == 0, "deselected needs no check"
    assert stats["failed"] == 0


def test_a_widened_window_syncs_what_the_new_floor_reaches():
    fake, state = _corpus(), {}
    _sync(fake, state, start_date="2026/01/15")
    assert set(state["fingerprints"]) == {"pmid:101"}

    rows, stats = _sync(fake, state, start_date="2026/01/01")

    assert _ids(rows) == {"pmid:100"}
    assert stats["failed"] == 0


def test_a_capped_listing_never_tombstones():
    fake, state = _corpus(), {}
    _sync(fake, state)

    fake.removed.add("100")
    fake.total_override = 60_000
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == set()
    assert stats["failed_cap"] == 1
    assert stats["failed"] == 1
    assert "pmid:100" in state["fingerprints"]


def test_a_non_numeric_id_from_an_answer_is_never_checked():
    fake, state = _corpus(), {}
    _sync(fake, state)

    state["fingerprints"]["pmid:1;drop"] = "deadbeef"
    state["edat"]["pmid:1;drop"] = "2026/01/01"
    _, stats = _sync(fake, state)

    assert stats["failed_deletion_check"] == 1
    assert "pmid:1;drop" in state["fingerprints"], "a check that did not run keeps it"
    terms = [params.get("term", "") for _, params in fake.calls]
    assert not any("1;drop" in term for term in terms)


def test_a_listing_answer_without_an_id_list_raises_instead_of_forgetting():
    fake, state = _corpus(), {}
    _sync(fake, state)
    before = dict(state["fingerprints"])
    original = fake._esearch

    def broken(params):
        if params.get("retmax") == PAGE_SIZE:
            return {"esearchresult": {"count": "2"}}
        return original(params)

    fake._esearch = broken
    with pytest.raises(PubMedAPIError, match="id list"):
        _sync(fake, state)

    assert state["fingerprints"] == before


def test_a_fetch_answer_missing_a_requested_pmid_raises_instead_of_trusting_it():
    fake, state = _corpus(), {}
    original = fake._efetch

    def partial(params):
        ids = params["id"].split(",")[:1]
        return original({"id": ",".join(ids)})

    fake._efetch = partial
    with pytest.raises(PubMedAPIError, match="without PMID"):
        _sync(fake, state)

    assert state.get("fingerprints") == {}


# --- budget, rate limit and resuming ---


def test_a_budget_cut_during_the_fetch_resumes_without_loss_or_duplicates():
    fake, state = _corpus(), {}
    _sync(fake, state)
    fake.add_article("102", "Napping and creativity", edat="2026/03/01")

    rows, stats = _sync(fake, state, max_requests=1)
    assert stats["failed_budget"] == 1
    assert rows == []
    assert state["pending_fetch"] == ["102"]

    rows, stats = _sync(fake, state)
    assert _ids(rows) == {"pmid:102"}
    assert stats["failed"] == 0
    assert stats["fetched"] == 1


def test_a_budget_cut_mid_listing_resumes_at_the_saved_offset():
    fake = FakePubMed()
    for n in range(PAGE_SIZE + 50):
        fake.add_article(f"{1000 + n}", f"Study number {n}", edat="2026/01/01")
    state: dict = {}

    rows, stats = _sync(fake, state, max_requests=2)
    assert stats["failed_budget"] == 1
    assert len(rows) == 200, "one listing page and its fetch batch, then the next page was cut"
    assert state["listing"]["retstart"] == PAGE_SIZE
    assert len(state["pending_fetch"]) == PAGE_SIZE - 200

    rows, stats = _sync(fake, state)
    assert len(rows) == PAGE_SIZE - 200 + 50
    assert stats["failed"] == 0
    assert len(state["fingerprints"]) == PAGE_SIZE + 50
    assert _tombstones(rows) == set()


def test_a_cut_during_the_deletion_checks_finishes_them_next_run():
    fake, state = _corpus(), {}
    _sync(fake, state)
    fake.removed.update({"100", "101"})

    rows, stats = _sync(fake, state, max_requests=2)
    assert _tombstones(rows) == {"pmid:100"}
    assert stats["failed_budget"] == 1
    assert state["pending_deletes"] == ["pmid:101"]

    rows, stats = _sync(fake, state)
    assert _tombstones(rows) == {"pmid:101"}
    assert stats["failed"] == 0
    assert state["pending_deletes"] == []


def test_a_rate_limited_run_stops_cleanly_and_resumes():
    fake, state = _corpus(), {}
    _sync(fake, state)
    fake.add_article("102", "Napping and creativity", edat="2026/03/01")

    fake.fail_after = 1
    fake.fail_with = PubMedRateLimitedError("PubMed rate limit exceeded (HTTP 429)")
    rows, stats = _sync(fake, state)
    assert rows == []
    assert stats["failed_rate_limit"] == 1

    fake.fail_after = None
    rows, stats = _sync(fake, state)
    assert _ids(rows) == {"pmid:102"}
    assert stats["failed"] == 0


def test_an_auth_error_before_any_row_raises():
    fake = _corpus()
    fake.fail_after = 0
    fake.fail_with = PubMedAuthError("PubMed rejected the API key (HTTP 400)")

    with pytest.raises(PubMedAuthError):
        _sync(fake, {})


def test_a_key_rejected_after_rows_were_emitted_cuts_the_run_and_keeps_them():
    fake, state = _corpus(), {}
    _sync(fake, state)
    for n in range(201, 451):
        fake.add_article(str(n), f"Crowd study {n}", edat="2026/03/01")
    total = 2 + 250

    # The listing and the first fetch batch pass, the second fetch batch is rejected.
    fake.fail_after = 2
    fake.fail_with = PubMedAuthError("PubMed rejected the API key (HTTP 400)")
    rows, stats = _sync(fake, state)

    assert stats["failed_auth"] == 1
    assert len(rows) == 200
    assert len(state["fingerprints"]) == 202, "the emitted rows are kept in state"
    assert len(state["pending_fetch"]) == 50

    fake.fail_after = None
    rows, stats = _sync(fake, state)
    assert len(rows) == 50
    assert stats["failed"] == 0
    assert len(state["fingerprints"]) == total


def test_a_revoked_run_is_not_reported_as_a_spent_budget():
    fake, stats = _corpus(), {}

    def check_active():
        if len(fake.calls) >= 1:
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError):
        list(_iter_rows(fake, {"name": "t"}, stats, end_date=END, check_active=check_active))

    assert "failed_budget" not in stats


def test_check_active_runs_before_every_request_and_can_stop_the_sync():
    fake, checks = _corpus(), []

    def check_active():
        checks.append(len(fake.calls))
        if len(checks) > 1:
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError):
        _sync(fake, {}, check_active=check_active)

    assert checks == [0, 1]
    assert len(fake.calls) == 1


def test_a_run_counts_the_requests_it_made():
    fake = _corpus()

    _, stats = _sync(fake, {})

    assert stats["requests"] == len(fake.calls)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param({}, id="plain"),
        pytest.param({"max_requests": 1}, id="cut_by_budget"),
        pytest.param({"total_override": 60_000}, id="capped"),
    ],
)
def test_every_run_reports_every_count_key_however_it_ends(scenario):
    fake, options = _corpus(), dict(scenario)
    if "total_override" in options:
        fake.total_override = options.pop("total_override")

    _, stats = _sync(fake, {}, **options)

    assert COUNT_KEYS <= stats.keys()


# --- the source factory's validation ---


def test_the_source_factory_refuses_bad_arguments():
    pytest.importorskip("dlt")
    from cognee_connector_pubmed import pubmed_source

    with pytest.raises(ValueError, match="resource_name"):
        pubmed_source(resource_name=" ")
    with pytest.raises(ValueError, match="start_date"):
        pubmed_source(resource_name="t", start_date="2026-01-01")
    with pytest.raises(ValueError, match="end_date"):
        pubmed_source(resource_name="t", end_date="01/01/2026")
    with pytest.raises(ValueError, match="after end_date"):
        pubmed_source(resource_name="t", start_date="2026/06/02", end_date=END)
    with pytest.raises(ValueError, match="query"):
        pubmed_source(resource_name="t", query=42)
    with pytest.raises(ValueError, match="max_requests"):
        pubmed_source(resource_name="t", max_requests=0)
    with pytest.raises(ValueError, match="today"):
        pubmed_source(resource_name="t", today=lambda: "yesterday")
    source = pubmed_source(resource_name="t", end_date=END)
    assert source.name == "t"
