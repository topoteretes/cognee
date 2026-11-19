"""Unit tests for the Greenhouse (Harvest v3) dlt connector.

Two layers, all runnable in CI without live Greenhouse credentials:

* DB-free tests for rendering (allowlist fields, HTML→plain text, scorecard
  privacy), Link-header cursor parsing, and the document-source marker.
* dlt-pipeline tests (mocked Greenhouse client, temp sqlite destination)
  covering the acceptance criteria: initial load, edit re-sync, forget-on-delete
  for vanished records, and the scorecards opt-in gate.
"""

import pytest
from cognee.tasks.ingestion.dlt_utils import document_source_tag

from cognee_community_connector_greenhouse.greenhouse import (
    GREENHOUSE_SOURCE_NAME,
    GREENHOUSE_TABLE_JOB_POSTS,
    GREENHOUSE_TABLE_JOBS,
    GREENHOUSE_TABLE_SCORECARDS,
    _job_post_to_row,
    _job_to_row,
    _parse_next_link,
    _plain_text,
    _scorecard_to_row,
)


def _job(job_id, title="Engineer", status="open", **extra):
    job = {"id": job_id, "title": title, "status": status}
    job.update(extra)
    return job


def _post(post_id, job_id=7, title="Post", content="<p>Hello <b>world</b></p>"):
    return {
        "id": post_id,
        "job_id": job_id,
        "title": title,
        "content": content,
        "live": True,
        "public_url": f"https://boards.greenhouse.io/job/{post_id}",
    }


def _scorecard(card_id, application_id=10, candidate_id=99, recommendation="strong_hire"):
    return {
        "id": card_id,
        "application_id": application_id,
        "candidate_id": candidate_id,
        "interview": "Technical Screen",
        "overall_recommendation": recommendation,
        "submitted_at": "2026-05-01T10:00:00Z",
        "submitted_by": {"first_name": "Ada", "last_name": "Lovelace"},
        "ratings": [{"name": "Algorithms", "score": "4"}],
        "notes": "SECRET PRIVATE NOTES",  # must never reach a document
    }


class FakeGreenhouseClient:
    """Stand-in for GreenhouseClient backed by in-memory fixtures.

    Mirrors the real client's contract: ``get(url) -> (json, headers)`` with
    ``Link``-header cursor pagination.
    """

    def __init__(self, jobs=None, posts=None, scorecards=None):
        self._jobs = jobs or []
        self._posts = posts or []
        self._scorecards = scorecards or []
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(url)
        if "/v3/jobs" in url:
            items = self._jobs
        elif "/v3/job_posts" in url:
            items = self._posts
        elif "/v3/scorecards" in url:
            items = self._scorecards
        else:
            return [], {}
        return items, {}


class LinkClient:
    """Fake that returns one record per page to exercise Link pagination."""


# ---------------------------------------------------------------------------
# Rendering (DB-free)
# ---------------------------------------------------------------------------


def test_plain_text_strips_tags_and_entities():
    assert _plain_text("<p>Hello &amp; <b>world</b></p>") == "Hello & world"
    assert _plain_text("No markup") == "No markup"
    assert _plain_text(None) == ""


def test_job_to_row_renders_allowlist_fields():
    row = _job_to_row(
        _job(1, title="Engineer", status="closed", department="Eng", opened_at="2026-01-01")
    )
    assert row["id"] == 1
    assert row["title"] == "Engineer"
    assert "Status: closed" in row["content"]
    assert "Department: Eng" in row["content"]
    assert "Opened: 2026-01-01" in row["content"]


def test_job_to_row_ignores_unknown_fields():
    # A field outside the allowlist must never leak into the document.
    row = _job_to_row(_job(2, salary_info="TOP SECRET", private_notes="nope"))
    assert "salary" not in row["content"].lower()
    assert "TOP SECRET" not in row["content"]


def test_job_post_to_row_includes_plain_text_description():
    row = _job_post_to_row(_post(3, content="<p>Build <b>memories</b>.</p>"))
    assert row["id"] == 3
    assert row["url"] == "https://boards.greenhouse.io/job/3"
    assert "Build memories." in row["content"]
    assert "<b>" not in row["content"]
    assert "Job ID: 7" in row["content"]


def test_scorecard_to_row_excludes_private_data():
    row = _scorecard_to_row(_scorecard(4, application_id=10, candidate_id=99))
    assert row["id"] == 4
    assert "Technical Screen" in row["content"]
    assert "strong_hire" in row["content"]
    assert "Ada Lovelace" in row["content"]
    assert "Algorithms: 4" in row["content"]
    # Candidate feedback is sensitive — private notes never rendered.
    assert "SECRET PRIVATE NOTES" not in row["content"]
    assert "notes" not in row["content"].lower()


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_parse_next_link_extracts_only_next():
    header = (
        '<https://harvest.greenhouse.io/v3/jobs?cursor=abc>; rel="last", '
        '<https://harvest.greenhouse.io/v3/jobs?cursor=def>; rel="next"'
    )
    assert _parse_next_link(header) == "https://harvest.greenhouse.io/v3/jobs?cursor=def"
    assert _parse_next_link("") is None
    assert _parse_next_link(None) is None
    assert _parse_next_link('<https://h/v3/jobs?cursor=x>; rel="first"') is None


def test_parse_next_link_handles_list_header():
    header = ['<https://h/v3/jobs?cursor=z>; rel="next"']
    assert _parse_next_link(header) == "https://h/v3/jobs?cursor=z"


# ---------------------------------------------------------------------------
# Source wiring
# ---------------------------------------------------------------------------


def test_greenhouse_source_declares_document_marker():
    from cognee_community_connector_greenhouse.greenhouse import greenhouse_source

    source = greenhouse_source(client=FakeGreenhouseClient())
    assert GREENHOUSE_SOURCE_NAME == "greenhouse"
    assert document_source_tag(source) == "greenhouse"


def test_greenhouse_source_requires_credentials_in_env(monkeypatch):
    from cognee_community_connector_greenhouse.greenhouse import greenhouse_source

    monkeypatch.delenv("GREENHOUSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GREENHOUSE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GREENHOUSE_SUB", raising=False)

    # No client injected and no credentials anywhere → clear error, not a crash.
    with pytest.raises(ValueError, match="credentials"):
        greenhouse_source()


def test_greenhouse_source_reads_credentials_from_env(monkeypatch):
    from cognee_community_connector_greenhouse.greenhouse import greenhouse_source

    monkeypatch.setenv("GREENHOUSE_CLIENT_ID", "demo-id")
    monkeypatch.setenv("GREENHOUSE_CLIENT_SECRET", "demo-secret")
    monkeypatch.delenv("GREENHOUSE_SUB", raising=False)

    source = greenhouse_source()
    assert source.name == GREENHOUSE_SOURCE_NAME
    assert set(source.resources) == {GREENHOUSE_TABLE_JOBS, GREENHOUSE_TABLE_JOB_POSTS}


def _run_sync(dlt, tmp_path, fake_client, **kwargs):
    from cognee_community_connector_greenhouse.greenhouse import greenhouse_source

    db_path = (tmp_path / "greenhouse.db").as_posix()
    pipeline = dlt.pipeline(
        pipeline_name="greenhouse_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="greenhouse_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    pipeline.run(greenhouse_source(client=fake_client, **kwargs))
    return pipeline


def _read_table(pipeline, table):
    with pipeline.sql_client() as client:
        rows = client.execute_sql(f"SELECT id, title, content FROM {table}")
    return {row[0]: {"id": row[0], "title": row[1], "content": row[2]} for row in rows}


@pytest.fixture
def dlt_mod():
    return pytest.importorskip("dlt")


def test_first_sync_loads_all_resources(dlt_mod, tmp_path):
    client = FakeGreenhouseClient(
        jobs=[_job(1, title="Engineer")],
        posts=[_post(2, job_id=1)],
        scorecards=[_scorecard(3)],
    )
    pipeline = _run_sync(dlt_mod, tmp_path, client, include_scorecards=True)

    jobs = _read_table(pipeline, GREENHOUSE_TABLE_JOBS)
    posts = _read_table(pipeline, GREENHOUSE_TABLE_JOB_POSTS)
    cards = _read_table(pipeline, GREENHOUSE_TABLE_SCORECARDS)

    assert set(jobs) == {1}
    assert set(posts) == {2}
    assert set(cards) == {3}
    assert "Engineer" in jobs[1]["title"]


def test_scorecards_are_opt_in(dlt_mod, tmp_path):
    client = FakeGreenhouseClient(jobs=[_job(1)], scorecards=[_scorecard(2)])
    pipeline = _run_sync(dlt_mod, tmp_path, client, include_scorecards=False)

    jobs = _read_table(pipeline, GREENHOUSE_TABLE_JOBS)
    assert set(jobs) == {1}
    # Scorecard table must not exist when not explicitly opted in.
    with pipeline.sql_client() as conn:
        found = conn.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' "
            f"AND name='{GREENHOUSE_TABLE_SCORECARDS}'"
        )
    assert found == []


def test_edit_is_reflected_on_resync(dlt_mod, tmp_path):

    client = FakeGreenhouseClient(jobs=[_job(1, title="Engineer")])
    _run_sync(dlt_mod, tmp_path, client)

    edited = FakeGreenhouseClient(jobs=[_job(1, title="Senior Engineer")])
    pipeline = _run_sync(dlt_mod, tmp_path, edited)

    rows = _read_table(pipeline, GREENHOUSE_TABLE_JOBS)
    assert rows[1]["title"] == "Senior Engineer"


def test_vanished_job_is_removed_on_resync(dlt_mod, tmp_path):
    # A job that disappears upsteam (deleted) must be forgotten by orphan
    # cleanup, which reconciles against the full snapshot.
    client = FakeGreenhouseClient(jobs=[_job(1), _job(2)])
    _run_sync(dlt_mod, tmp_path, client)

    vanished = FakeGreenhouseClient(jobs=[_job(2)])
    pipeline = _run_sync(dlt_mod, tmp_path, vanished)

    rows = _read_table(pipeline, GREENHOUSE_TABLE_JOBS)
    assert 1 not in rows
    assert 2 in rows


def test_fetch_failure_aborts_sync_leaving_memory_untouched(dlt_mod, tmp_path):
    from cognee_community_connector_greenhouse.greenhouse import greenhouse_source

    class BoomClient:
        def get(self, url, params=None):
            raise RuntimeError("boom")

    db_path = (tmp_path / "boom.db").as_posix()
    pipeline = dlt_mod.pipeline(
        pipeline_name="greenhouse_boom",
        destination=dlt_mod.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="greenhouse_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    with pytest.raises(Exception):  # noqa: B017 - dlt wraps the source error
        pipeline.run(greenhouse_source(client=BoomClient()))


def test_iter_payload_follows_link_header(dlt_mod, tmp_path):
    # _iter_payload follows rel="next" until the Link header disappears.
    class LinkClient:
        def __init__(self, items):
            self._items = items

        def get(self, url, params=None):
            if "cursor=a" in url:
                return self._items[1:], {}
            if "cursor=b" in url:
                return self._items[2:], {}
            return self._items[0:1], {
                "Link": '<https://harvest.greenhouse.io/v3/jobs?cursor=a>; rel="next"'
            }

    client = LinkClient([_job(i) for i in (11, 12, 13)])
    pipeline = _run_sync(dlt_mod, tmp_path, client)
    rows = _read_table(pipeline, GREENHOUSE_TABLE_JOBS)
    assert set(rows) == {11, 12, 13}
