# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A bad argument fails when apollo_source() is built, not halfway through a run.

That covers a bad key, a missing resource_name and an empty or unknown ``include``.
``include=None`` and ``custom_field_ids`` reach the walk the way a host passes them, through a
real dlt pipeline, and the counts a host reads after a run keep every key the walk reports.
"""

import pytest
from fake_apollo import FakeApollo

from cognee.modules.integrations import ingestion
from cognee.tasks.ingestion.connectors import apollo as apollo_module
from cognee.tasks.ingestion.connectors import apollo_source
from cognee.tasks.ingestion.connectors.apollo import (
    PAGE_SIZE,
    ApolloAccessError,
    ApolloAPIError,
    ApolloAuthError,
    ApolloSourceError,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
)

TABLE = "apollo_test"
PHONE = "+1 555 0199"
ALL_ROWS = ["account:a1", "contact:c1", "contact:c2", "sequence:s1"]
# The wording Notion gives the same guard: the column is what the core version is needed for.
VERSION_GUARD = (
    "Apollo sync requires a Cognee build that reads per-row node sets "
    "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
)
# Lookups of a contacts-only run: contact stages, lists and the sequence list.
CONTACT_LOOKUPS = 3
# What a run does to a record whose deletion check keeps failing (apollo.md).
FAILURES_BEFORE_BACKOFF = 5


def _workspace() -> FakeApollo:
    fake = FakeApollo()
    fake.add_sequence("s1", "Q4 Outreach")
    fake.add_account("a1", "Acme")
    fake.add_contact("c1", "Ada Lovelace", account_id="a1")
    fake.add_contact("c2", "Ben Turing", account_id="a1")
    return fake


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion():
    resource = apollo_source(service=FakeApollo(), resource_name=TABLE)

    assert resource.name == TABLE
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "apollo"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource_name": TABLE},
        {"resource_name": "  ", "api_key": "key"},
        {"resource_name": TABLE, "api_key": "key", "include": ()},
        {"resource_name": TABLE, "api_key": "key", "include": ("contacts", "people")},
        {"resource_name": TABLE, "api_key": "key", "include": "contacts"},
        {"resource_name": TABLE, "api_key": "bad key"},
    ],
)
def test_bad_arguments_are_rejected_when_the_resource_is_built(kwargs):
    with pytest.raises(ValueError):
        apollo_source(**kwargs)


@pytest.mark.parametrize(
    "argument",
    [
        "contact_stage_ids",
        "contact_label_ids",
        "account_stage_ids",
        "account_label_ids",
        "custom_field_ids",
    ],
)
def test_one_id_given_as_a_string_is_refused_instead_of_read_as_its_letters(argument):
    # A stage filter of "cs1" read as ["c", "s", "1"] matches nothing, so the walk would
    # find every synced record outside the selection and forget it.
    with pytest.raises(ValueError, match=rf"{argument} must be a list or tuple of id strings"):
        apollo_source(service=FakeApollo(), resource_name=TABLE, **{argument: "cs1"})


# Anything that is not a list or tuple of strings: bytes and sets would iterate into
# something else, a dict into its keys, and a number or None inside a list is no id.
@pytest.mark.parametrize("ids", [b"cs1", {"cs1"}, {"cs1": 1}, ["cs1", 2], [None], [b"cs1"], 7])
def test_ids_of_the_wrong_type_are_refused(ids):
    with pytest.raises(ValueError, match="contact_stage_ids must be a list or tuple"):
        apollo_source(service=FakeApollo(), resource_name=TABLE, contact_stage_ids=ids)


# One shape: what GET /fields?source=custom lists. A bare id, a modality the source does not
# render, an empty id and a list with one bad entry are all refused.
@pytest.mark.parametrize(
    "ids",
    [["f1"], ["deal.f1"], ["contact."], [""], ["contact.f1", "f2"], ["Contact.f1"]],
)
def test_a_custom_field_id_in_another_shape_than_fields_lists_is_refused(ids):
    with pytest.raises(ValueError, match=r"custom_field_ids must be ids as GET /fields"):
        apollo_source(service=FakeApollo(), resource_name=TABLE, custom_field_ids=ids)


def test_custom_field_ids_of_both_modalities_are_accepted():
    apollo_source(
        service=FakeApollo(), resource_name=TABLE, custom_field_ids=("contact.f1", "account.f3")
    )


def test_resource_name_has_no_default():
    with pytest.raises(TypeError, match="resource_name"):
        apollo_source(api_key="key")


# Version 2 is the first that reads the cognee_node_set column. An older core would ingest
# every row with no node set, and the next release would then change every document's id.
@pytest.mark.parametrize("version", [0, 1])
def test_refuses_core_that_does_not_read_node_sets(monkeypatch, version):
    monkeypatch.setattr(apollo_module.dlt_utils, "DOCUMENT_SYNC_VERSION", version)

    with pytest.raises(RuntimeError) as raised:
        apollo_source(service=FakeApollo(), resource_name=TABLE)

    assert str(raised.value) == VERSION_GUARD


def test_accepts_the_first_core_that_reads_node_sets(monkeypatch):
    monkeypatch.setattr(apollo_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 2)

    apollo_source(service=FakeApollo(), resource_name=TABLE)


def _pipeline(tmp_path):
    import dlt

    return dlt.pipeline(
        pipeline_name="apollo_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'a.db'}"),
        dataset_name="apollo",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


# ingest_dlt_source runs every document source with these hints: without the json hint dlt would
# turn the node set column into a child table, and a tombstone is tested against that.
def _run(tmp_path, source, table: str = TABLE) -> list[str]:
    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id FROM "{table}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return [row[0] for row in cursor.fetchall()]


def _contents(tmp_path, source, table: str = TABLE) -> dict[str, str]:
    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{table}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return {row[0]: row[1] for row in cursor.fetchall()}


def test_state_and_deletes_survive_between_real_dlt_runs(tmp_path):
    def run(source) -> list[str]:
        return _run(tmp_path, source)

    fake = _workspace()
    first = apollo_source(service=fake, resource_name=TABLE)
    assert run(first) == ALL_ROWS
    assert first.cognee_sync_stats["failed"] == 0

    quiet = apollo_source(service=fake, resource_name=TABLE)
    run(quiet)
    assert quiet.cognee_sync_stats["skipped"] == quiet.cognee_sync_stats["scanned"] == 4

    del fake.contacts["c2"]
    forget = apollo_source(service=fake, resource_name=TABLE)
    assert run(forget) == ["account:a1", "contact:c1", "sequence:s1"]
    assert forget.cognee_sync_stats["deleted"] == 1


def test_include_none_syncs_every_kind(tmp_path):
    """A host with no selection passes ``include=None``, which used to fail on ``tuple(None)``."""
    source = apollo_source(service=_workspace(), resource_name=TABLE, include=None)

    assert _run(tmp_path, source) == ALL_ROWS


def test_custom_field_ids_reach_the_walk_the_way_apollo_lists_them(tmp_path):
    fake = _workspace()
    fake.contacts["c1"]["typed_custom_fields"] = {"f1": PHONE}
    plain = _contents(tmp_path, apollo_source(service=fake, resource_name=TABLE))
    assert PHONE not in plain["contact:c1"]

    # `contact.f1` is how GET /fields lists the id; the record keys its value by `f1`
    listed = apollo_source(service=fake, resource_name=TABLE, custom_field_ids=["contact.f1"])
    chosen = _contents(tmp_path, listed)

    assert f"Messaging Angle: {PHONE}" in chosen["contact:c1"].splitlines()


def test_a_reused_source_forgets_the_quota_its_client_saw_last_run(tmp_path):
    fake = _workspace()
    source = apollo_source(service=fake, resource_name=TABLE)
    _run(tmp_path, source)

    fake.rate_limit = {"/contact_stages": {"minute": 0}}  # stale: the window has reset since
    fake.add_contact("c3", "Cara Hopper")
    assert "contact:c3" in _run(tmp_path, source)
    assert source.cognee_sync_stats["failed"] == 0


def test_two_accounts_with_their_own_resource_names_stay_apart(tmp_path):
    team_a, team_b = _workspace(), FakeApollo(team_id="team2")
    team_b.add_contact("b1", "Dev Hollerith")
    team_b.add_contact("b2", "Eve Shannon")

    def run(fake, name):
        return _run(tmp_path, apollo_source(service=fake, resource_name=name), name)

    assert run(team_a, "apollo_a") == ALL_ROWS
    assert run(team_b, "apollo_b") == ["contact:b1", "contact:b2"]

    del team_b.contacts["b2"]
    assert run(team_b, "apollo_b") == ["contact:b1"]
    assert run(team_a, "apollo_a") == ALL_ROWS


def test_a_second_account_under_the_same_resource_name_is_refused(tmp_path):
    from dlt.pipeline.exceptions import PipelineStepFailed

    _run(tmp_path, apollo_source(service=_workspace(), resource_name=TABLE))

    other = FakeApollo(team_id="team2")
    other.add_contact("b1", "Dev Hollerith")
    with pytest.raises(PipelineStepFailed, match="resource_name"):
        _run(tmp_path, apollo_source(service=other, resource_name=TABLE))

    assert _run(tmp_path, apollo_source(service=_workspace(), resource_name=TABLE)) == ALL_ROWS


def test_check_active_can_stop_a_run_through_the_resource(tmp_path):
    from dlt.pipeline.exceptions import PipelineStepFailed

    def revoked():
        raise PermissionError("access revoked")

    fake = _workspace()
    with pytest.raises(PipelineStepFailed, match="access revoked"):
        _run(tmp_path, apollo_source(service=fake, resource_name=TABLE, check_active=revoked))

    assert fake.calls == []


def test_the_same_filter_in_another_order_resumes_a_cut_listing(tmp_path):
    """The ids go into the search body, whose hash says whether a stored page still points
    into this listing, so their order must not matter."""
    fake = FakeApollo()
    for n in range(PAGE_SIZE * 2 + 50):
        fake.add_contact(f"c{n:03d}", f"Person {n}", contact_stage_id="cs1" if n % 2 else "cs2")

    def source(stage_ids, **kwargs):
        return apollo_source(
            service=fake,
            resource_name=TABLE,
            include=("contacts",),
            contact_stage_ids=stage_ids,
            **kwargs,
        )

    cut = source(["cs2", "cs1", "cs1"], max_requests=CONTACT_LOOKUPS + 1)
    _run(tmp_path, cut)
    assert cut.cognee_sync_stats["failed_budget"] == 1
    sent = [body for path, body in fake.bodies if path == "/contacts/search"]
    assert sent[0]["contact_stage_ids"] == ["cs1", "cs2"], "sorted, without the repeat"

    calls_before = len(fake.bodies)
    _run(tmp_path, source(["cs1", "cs2"]))

    pages = [b["page"] for path, b in fake.bodies[calls_before:] if path == "/contacts/search"]
    assert pages[0] == 2, "the listing resumed where the cut run stopped, not at page 1"


# --- what a host reads after a run ---


def test_a_held_back_record_is_a_count_and_not_a_failure(tmp_path):
    """A host marks a sync degraded when ``failed`` is not 0, as every connector's sync
    does, so records left alone by back-off must not count as failed."""
    fake = _workspace()
    _run(tmp_path, apollo_source(service=fake, resource_name=TABLE))
    fake.hidden.add("c2")
    fake.fail_paths["/contacts/c2"] = ApolloAPIError("Apollo request failed: HTTP 500")
    for _ in range(FAILURES_BEFORE_BACKOFF):
        _run(tmp_path, apollo_source(service=fake, resource_name=TABLE))
    source = apollo_source(service=fake, resource_name=TABLE)

    _run(tmp_path, source)
    counts = ingestion.source_counts(source)

    assert counts["held_back"] == 1
    assert counts["failed"] == 0
    assert "failed_deletion_check" not in counts


def test_a_host_reads_every_count_the_walk_reports(tmp_path):
    """A host sums ``source_counts`` into the connection's sync result, which keeps only
    non-negative ints, so a count of any other type would silently vanish from it."""
    fake = _workspace()
    _run(tmp_path, apollo_source(service=fake, resource_name=TABLE))
    # c2 leaves the search results and its record read is refused: kept, and counted
    fake.hidden.add("c2")
    fake.fail_paths["/contacts/c2"] = ApolloAccessError(
        "The Apollo API key or plan cannot call /contacts/{id} (HTTP 403)"
    )
    calls_before = len(fake.calls)
    source = apollo_source(service=fake, resource_name=TABLE)

    _run(tmp_path, source)
    counts = ingestion.source_counts(source)

    assert counts == source.cognee_sync_stats
    assert counts["requests"] == len(fake.calls) - calls_before
    assert counts["failed"] == counts["failed_access"] == 1
    assert counts["deleted"] == 0


# What ends a run by raising, and the one count key a host should find for it. A 5xx or a
# bad configuration has no key of its own among the ones a host knows, so it raises without one.
@pytest.mark.parametrize(
    ("path", "error", "reason"),
    [
        (
            "/contacts/search",
            ApolloAccessError("cannot call /contacts/search (HTTP 403)"),
            "access",
        ),
        ("/contact_stages", ApolloAuthError("Apollo rejected the API key (HTTP 401)"), "auth"),
        ("/contact_stages", ApolloAPIError("Apollo request failed: HTTP 500"), None),
        ("/contact_stages", ApolloSourceError("a stage filter names an unknown id"), None),
    ],
)
def test_a_run_that_raises_leaves_the_count_key_of_its_error_for_the_host(
    tmp_path, path, error, reason
):
    """dlt drops the state of a run that raises, not the stats object, and a host adds its
    counts in a ``finally`` (``sync_selected_resources``), so they must say why instead of
    ``failed: 0``."""
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _workspace()
    fake.fail_paths[path] = error
    source = apollo_source(service=fake, resource_name=TABLE)

    with pytest.raises(PipelineStepFailed):
        _run(tmp_path, source)

    counts = ingestion.source_counts(source)
    assert {key for key, count in counts.items() if key.startswith("failed_") and count} == (
        {f"failed_{reason}"} if reason else set()
    )
    assert counts["failed"] == 1, "every raise is one failure, with or without a reason key"
    if reason:
        assert counts[f"failed_{reason}"] == 1


def test_a_refused_listing_raises_because_a_cut_run_would_keep_its_state(tmp_path):
    """A listing the key cannot call raises, and dlt drops the state of that run: the accounts
    it had finished are listed again by the next run and nothing is forgotten. A cut run would
    keep them finished, never close the cycle and so never see a later delete."""
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _workspace()
    assert _run(tmp_path, apollo_source(service=fake, resource_name=TABLE)) == ALL_ROWS
    fake.fail_paths["/contacts/search"] = ApolloAccessError("cannot call /contacts/search")
    with pytest.raises(PipelineStepFailed, match="/contacts/search"):
        _run(tmp_path, apollo_source(service=fake, resource_name=TABLE))
    account_listings = fake.count("/accounts/search")
    del fake.fail_paths["/contacts/search"]

    clean = apollo_source(service=fake, resource_name=TABLE)

    assert _run(tmp_path, clean) == ALL_ROWS
    assert fake.count("/accounts/search") == account_listings + 1, "listed again, not skipped"
    assert clean.cognee_sync_stats["deleted"] == 0
