"""Linear DLT source: paging, cursors, resumption, rendering and token hygiene."""

import logging
from datetime import datetime, timezone
from time import sleep as _sleep
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.connectors import linear as linear_module
from cognee.tasks.ingestion.connectors.linear import (
    LinearAPIError,
    LinearAuthError,
    LinearClient,
    LinearEntityNotFoundError,
    LinearRateLimitedError,
    LinearTeamNotFoundError,
    _iter_rows,
    linear_source,
    render_issue,
    render_project,
)

TOKEN = "lin_oauth_SECRET_TOKEN_VALUE"
TEAM = "team-1"


def _ts(n: int) -> str:
    return f"2026-10-01T10:00:{n:02d}.000Z"


class FakeLinear:
    """An in-memory Linear team behind the GraphQL transport interface."""

    def __init__(self):
        self.issues: dict[str, dict] = {}
        self.comments: dict[str, dict] = {}
        self.projects: dict[str, dict] = {}
        self.calls: list[str] = []
        self.fail_after: int | None = None
        self.fail_with: Exception | None = None
        self.rate_limit: dict = {}

    # -- test helpers ------------------------------------------------------
    def add_issue(self, key: str, n: int, **extra):
        self.issues[key] = {
            "id": key,
            "identifier": f"ENG-{key}",
            "title": f"Issue {key}",
            "description": f"Body of {key}",
            "url": f"https://linear.app/x/issue/{key}",
            "trashed": False,
            "createdAt": _ts(0),
            "updatedAt": _ts(n),
            "state": {"name": "Todo"},
            "assignee": None,
            "priority": 2,
            "labels": {"nodes": [{"name": "b"}, {"name": "a"}]},
            "project": None,
            "parent": None,
            **extra,
        }

    def add_comment(self, key: str, issue: str, n: int, body="hello", bump_issue=False):
        self.comments[key] = {
            "id": key,
            "issue": {"id": issue},
            "body": body,
            "createdAt": _ts(n),
            "updatedAt": _ts(n),
            "user": {"name": "Ana"},
        }
        if bump_issue:
            self.issues[issue]["updatedAt"] = _ts(n)

    def add_project(self, key: str, n: int):
        self.projects[key] = {
            "id": key,
            "name": f"Project {key}",
            "description": "d",
            "content": "c",
            "url": f"https://linear.app/x/project/{key}",
            "startDate": None,
            "targetDate": None,
            "updatedAt": _ts(n),
            "status": {"name": "Planned"},
            "lead": None,
        }

    # -- transport ---------------------------------------------------------
    def execute(self, query, variables=None):
        variables = variables or {}
        self.calls.append(query)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.fail_with
        if "LinearTeamIssues" in query:
            return {"team": self._issues(variables)}
        if "LinearTeamComments" in query:
            return self._comments(variables)
        if "LinearTeamProjects" in query:
            return {"team": self._projects(variables)}
        if "LinearIssueComments" in query:
            return self._issue_comments(variables)
        raise AssertionError(query)

    @staticmethod
    def _in_window(item, window):
        stamp = item["updatedAt"]
        return stamp >= window.get("gte", "") and stamp <= window.get("lte", "~")

    @staticmethod
    def _page(items, variables):
        items = sorted(items, key=lambda i: (i["updatedAt"], i["id"]), reverse=True)
        start = int(variables.get("after") or 0)
        end = start + variables["first"]
        info = {"hasNextPage": end < len(items), "endCursor": str(end)}
        return items[start:end], info

    def _issues(self, variables):
        filter_ = variables.get("filter") or {}
        items = list(self.issues.values())
        if "id" in filter_:
            items = [i for i in items if i["id"] in filter_["id"]["in"]]
        if "updatedAt" in filter_:
            items = [i for i in items if self._in_window(i, filter_["updatedAt"])]
        nodes, info = self._page(items, variables)
        shaped = []
        for node in nodes:
            own = [c for c in self.comments.values() if c["issue"]["id"] == node["id"]]
            own.sort(key=lambda c: c["createdAt"])
            shaped.append(
                {
                    **node,
                    "comments": {
                        "nodes": own[:50],
                        "pageInfo": {"hasNextPage": len(own) > 50, "endCursor": "50"},
                    },
                }
            )
        return {"issues": {"nodes": shaped, "pageInfo": info}}

    def _comments(self, variables):
        window = (variables["filter"] or {}).get("updatedAt", {})
        items = [c for c in self.comments.values() if self._in_window(c, window)]
        nodes, info = self._page(items, variables)
        return {"comments": {"nodes": nodes, "pageInfo": info}}

    def _projects(self, variables):
        window = (variables.get("filter") or {}).get("updatedAt", {})
        items = [p for p in self.projects.values() if self._in_window(p, window)]
        nodes, info = self._page(items, variables)
        return {"projects": {"nodes": nodes, "pageInfo": info}}

    def _issue_comments(self, variables):
        own = sorted(
            (c for c in self.comments.values() if c["issue"]["id"] == variables["id"]),
            key=lambda c: c["createdAt"],
        )
        start = int(variables["after"])
        end = start + variables["first"]
        page = {"hasNextPage": end < len(own), "endCursor": str(end)}
        return {"issue": {"comments": {"nodes": own[start:end], "pageInfo": page}}}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    # The fake's timestamps are on 2026-10-01; the source seeds its comment floor
    # from the wall clock, so pin it just before them.
    start = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(linear_module, "time", SimpleNamespace(time=lambda: start, sleep=_sleep))


def _stats():
    return {"scanned": 0, "skipped": 0, "failed": 0, "deleted": 0}


def _run(client, state, **kwargs):
    stats = _stats()
    rows = list(_iter_rows(client, TEAM, state, stats, **kwargs))
    return rows, stats


@pytest.fixture
def team():
    fake = FakeLinear()
    fake.add_issue("i1", 1)
    fake.add_issue("i2", 2)
    fake.add_issue("i3", 3)
    fake.add_comment("c1", "i1", 4)
    fake.add_project("p1", 5)
    return fake


def test_first_run_pages_through_everything_and_second_run_yields_nothing(team):
    state: dict = {}
    rows, stats = _run(team, state, page_size=2)

    assert sorted(row["id"] for row in rows) == ["issue:i1", "issue:i2", "issue:i3", "project:p1"]
    assert stats["failed"] == 0
    assert all(row["_deleted"] is False for row in rows)

    again, again_stats = _run(team, state, page_size=2)
    assert again == []
    assert again_stats["failed"] == 0


def test_empty_team_yields_nothing_and_settles(team):
    empty = FakeLinear()
    state: dict = {}
    rows, stats = _run(empty, state)
    assert rows == []
    assert stats["failed"] == 0
    assert _run(empty, state)[0] == []


def test_a_new_edit_after_a_settled_run_is_the_only_row(team):
    state: dict = {}
    _run(team, state)
    team.issues["i2"]["updatedAt"] = _ts(9)
    team.issues["i2"]["title"] = "Renamed"

    rows, _ = _run(team, state)
    assert [row["id"] for row in rows] == ["issue:i2"]
    assert rows[0]["title"] == "ENG-i2 Renamed"


def test_comment_edit_that_does_not_bump_the_issue_rerenders_the_issue(team):
    state: dict = {}
    _run(team, state)
    team.add_comment("c2", "i3", 20, body="a late remark")  # issue i3 keeps updatedAt 3

    rows, _ = _run(team, state)
    assert [row["id"] for row in rows] == ["issue:i3"]
    assert "a late remark" in rows[0]["content"]


def test_comment_on_an_issue_changed_in_the_same_run_renders_it_once(team):
    state: dict = {}
    _run(team, state)
    team.add_comment("c2", "i2", 20, body="both", bump_issue=True)

    rows, _ = _run(team, state)
    assert [row["id"] for row in rows] == ["issue:i2"]


def test_issues_sharing_an_updated_at_across_a_page_boundary_are_not_lost_or_repeated(team):
    team.add_issue("i4", 3)  # same stamp as i3
    state: dict = {}
    rows, _ = _run(team, state, page_size=1)
    assert sorted(row["id"] for row in rows if row["id"].startswith("issue")) == [
        "issue:i1",
        "issue:i2",
        "issue:i3",
        "issue:i4",
    ]
    assert _run(team, state, page_size=1)[0] == []


def test_a_rate_limit_stops_cleanly_and_the_next_run_finishes_the_walk(team):
    state: dict = {}
    team.fail_after = 2
    team.fail_with = LinearRateLimitedError()
    first, first_stats = _run(team, state, page_size=1)
    assert first_stats["failed_rate_limit"] == 1 and first_stats["failed"] == 1
    assert len(first) < 4

    team.fail_after = None
    second, second_stats = _run(team, state, page_size=1)
    assert second_stats["failed"] == 0
    assert {row["id"] for row in first + second} == {
        "issue:i1",
        "issue:i2",
        "issue:i3",
        "project:p1",
    }
    assert _run(team, state, page_size=1)[0] == []


def test_low_remaining_quota_stops_before_the_next_request(team):
    state: dict = {}
    original = team.execute

    def report_low_quota_after_answering(query, variables=None):
        data = original(query, variables)
        team.rate_limit = {"requests": 5000, "complexity": 1_000}  # known once Linear answers
        return data

    team.execute = report_low_quota_after_answering
    _, stats = _run(team, state)
    assert stats["failed_rate_limit"] == 1
    assert len(team.calls) == 1  # the first request ran; the budget check stopped the second


def test_request_budget_bounds_a_run_and_it_resumes(team):
    state: dict = {}
    rows, stats = _run(team, state, page_size=1, max_requests=2)
    assert stats["failed_budget"] == 1
    more, more_stats = _run(team, state, page_size=1, max_requests=50)
    assert more_stats["failed"] == 0
    assert len({row["id"] for row in rows + more}) == 4


def test_auth_error_with_no_progress_raises_and_with_progress_stops_cleanly(team):
    state: dict = {}
    team.fail_after = 0
    team.fail_with = LinearAuthError("rejected")
    with pytest.raises(LinearAuthError):
        _run(team, state)

    team.fail_after = 2
    rows, stats = _run(team, {}, page_size=1)
    assert rows and stats["failed_auth"] == 1


def test_missing_team_raises_a_typed_error():
    class Gone:
        rate_limit: dict = {}

        def execute(self, query, variables=None):
            return {"team": None}

    with pytest.raises(LinearTeamNotFoundError):
        _run(Gone(), {})


def test_trashed_issues_are_skipped(team):
    team.issues["i2"]["trashed"] = True
    rows, stats = _run(team, {})
    assert "issue:i2" not in {row["id"] for row in rows}
    assert stats["skipped"] >= 1


def test_comments_beyond_the_first_page_are_all_rendered(team):
    for n in range(60):
        team.add_comment(f"x{n:02d}", "i1", 10 + (n % 40), body=f"remark {n:02d}")
    rows, _ = _run(team, {})
    content = next(row for row in rows if row["id"] == "issue:i1")["content"]
    assert all(f"remark {n:02d}" in content for n in range(60))


def test_rendering_is_stable_and_carries_no_volatile_fields(team):
    issue = team.issues["i1"]
    comments = [team.comments["c1"]]
    first = render_issue(issue, comments)
    assert render_issue({**issue, "updatedAt": _ts(59)}, list(reversed(comments))) == first
    assert _ts(1) not in first["content"] and _ts(59) not in first["content"]
    assert first["content"].index("Labels: a, b") >= 0
    assert set(first) == {"id", "title", "content", "url", "_deleted"}

    project = render_project(team.projects["p1"])
    assert set(project) == {"id", "title", "content", "url", "_deleted"}


def test_rows_always_have_a_title():
    row = render_issue({"id": "z", "title": "", "identifier": ""}, [])
    assert row["title"] == "Untitled issue" and row["content"]


# -- the client ---------------------------------------------------------------
class _Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _client_with(monkeypatch, responses):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(headers)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("requests.post", post)
    return LinearClient(TOKEN, sleep=lambda _: None), calls


def test_rate_limit_is_a_distinct_error_even_though_linear_answers_400(monkeypatch):
    body = {"errors": [{"message": "x", "extensions": {"code": "RATELIMITED"}}]}
    client, _ = _client_with(
        monkeypatch, [_Response(400, body, {"X-RateLimit-Requests-Reset": "1760000000000"})]
    )
    with pytest.raises(LinearRateLimitedError) as caught:
        client.execute("query A { x }")
    assert caught.value.reset_at_ms == 1760000000000


def test_unauthorized_and_other_failures_map_to_typed_errors(monkeypatch):
    client, _ = _client_with(monkeypatch, [_Response(401, {}), _Response(403, {})])
    with pytest.raises(LinearAuthError):
        client.execute("query A { x }")
    with pytest.raises(LinearAPIError, match="HTTP 403"):
        client.execute("query A { x }")


def test_server_errors_are_retried_a_bounded_number_of_times(monkeypatch):
    client, _ = _client_with(
        monkeypatch, [_Response(502), _Response(502), _Response(200, {"data": {"ok": 1}})]
    )
    assert client.execute("query A { x }") == {"ok": 1}
    client, _ = _client_with(monkeypatch, [_Response(502)] * 3)
    with pytest.raises(LinearAPIError, match="HTTP 502"):
        client.execute("query A { x }")


def test_remaining_quota_headers_are_recorded(monkeypatch):
    client, _ = _client_with(
        monkeypatch,
        [
            _Response(
                200,
                {"data": {}},
                {
                    "X-RateLimit-Requests-Remaining": "4000",
                    "X-RateLimit-Complexity-Remaining": "1500000",
                },
            )
        ],
    )
    client.execute("query A { x }")
    assert client.rate_limit == {"requests": 4000, "complexity": 1_500_000}


def test_oauth_tokens_use_bearer_and_personal_keys_do_not(monkeypatch):
    client, calls = _client_with(monkeypatch, [_Response(200, {"data": {}})])
    client.execute("query A { x }")
    assert calls[0]["Authorization"] == f"Bearer {TOKEN}"

    personal = LinearClient("lin_api_abc")
    assert personal._headers()["Authorization"] == "lin_api_abc"


def test_the_token_never_reaches_errors_reprs_logs_or_rows(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    leaky = _Response(
        500,
        {"errors": [{"message": f"bad header Bearer {TOKEN}", "extensions": {"code": "X"}}]},
    )
    client, _ = _client_with(monkeypatch, [leaky] * 3)
    with pytest.raises(LinearAPIError) as caught:
        client.execute("query A { x }")
    assert TOKEN not in str(caught.value) and TOKEN not in repr(caught.value)
    assert TOKEN not in repr(client) and TOKEN not in str(vars(client).get("rate_limit"))

    import requests

    client, _ = _client_with(
        monkeypatch, [requests.ConnectionError(f"Authorization: Bearer {TOKEN}")] * 3
    )
    with pytest.raises(LinearAPIError) as caught:
        client.execute("query A { x }")
    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__

    fake = FakeLinear()
    fake.add_issue("i1", 1)
    rows, stats = _run(fake, {})
    assert TOKEN not in repr(rows) and TOKEN not in repr(stats)
    assert TOKEN not in caplog.text


def test_the_source_needs_a_client_or_a_token():
    with pytest.raises(ValueError):
        linear_source(team_id=TEAM)


def test_resource_name_defaults_to_a_linear_prefixed_table_name():
    resource = linear_source(team_id="0A1B-2C3D", access_token=TOKEN)
    assert resource.name == "linear_0a1b_2c3d"
    assert resource.cognee_sync_stats == {}


def test_check_active_runs_before_every_request(team):
    seen = []
    _run(team, {}, check_active=lambda: seen.append(1))
    assert len(seen) == len(team.calls)


def test_importing_the_connectors_package_does_not_load_the_linear_integration():
    import subprocess
    import sys

    code = (
        "import sys, cognee.tasks.ingestion.connectors.linear as m;"
        "bad = [n for n in sys.modules if n.startswith('cognee.modules.integrations.linear')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -- through a real dlt pipeline --------------------------------------------
def test_cursors_survive_between_real_dlt_runs(team, tmp_path):
    import dlt

    def run(source):
        pipeline = dlt.pipeline(
            pipeline_name="linear_test",
            destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'l.db'}"),
            dataset_name="linear",
            pipelines_dir=str(tmp_path / "pipelines"),
        )
        pipeline.run(source)
        query = 'SELECT id FROM "linear_team_1" ORDER BY id'
        with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
            return [row[0] for row in cursor.fetchall()]

    source = linear_source(team_id=TEAM, service=team, resource_name="linear_team_1")
    assert run(source) == ["issue:i1", "issue:i2", "issue:i3", "project:p1"]
    assert source.cognee_sync_stats["scanned"] > 0

    quiet = linear_source(team_id=TEAM, service=team, resource_name="linear_team_1")
    run(quiet)
    assert quiet.cognee_sync_stats["scanned"] == 0 or quiet.cognee_sync_stats["skipped"] > 0


# -- regressions found by the review swarm ----------------------------------
def test_a_comment_edited_after_its_issue_was_rendered_is_not_lost(team):
    state: dict = {}
    _run(team, state)
    original = team.execute
    edited = {"done": False}

    def edit_after_the_issues_page(query, variables=None):
        data = original(query, variables)
        if "LinearTeamIssues" in query and not edited["done"]:
            edited["done"] = True
            import copy

            data = copy.deepcopy(data)  # the page the source holds stays stale
            team.comments["c1"]["body"] = "EDITED"
            team.comments["c1"]["updatedAt"] = _ts(30)
        return data

    team.issues["i1"]["updatedAt"] = _ts(8)  # i1 changes, so its page is fetched
    team.execute = edit_after_the_issues_page
    first, _ = _run(team, state)
    team.execute = original
    second, _ = _run(team, state)

    contents = [row["content"] for row in first + second if row["id"] == "issue:i1"]
    assert any("EDITED" in content for content in contents)


def test_every_dirty_issue_is_rerendered_whatever_the_page_size(team):
    for n in range(4, 34):
        team.add_issue(f"d{n}", 1)
    state: dict = {}
    _run(team, state, page_size=50)
    for n in range(4, 34):
        team.add_comment(f"dc{n}", f"d{n}", 20 + n % 5, body=f"late {n}")

    rows, _ = _run(team, state, page_size=10)
    assert len([row for row in rows if row["id"].startswith("issue:d")]) == 30
    assert state["pending_issue_ids"] == []


def test_a_walk_that_cannot_advance_raises_instead_of_looking_finished(team):
    class NoCursor(FakeLinear):
        def _page(self, items, variables):
            nodes, _ = super()._page(items, variables)
            return nodes, {"hasNextPage": bool(nodes), "endCursor": None}

    broken = NoCursor()
    broken.add_issue("i1", 1)
    broken.add_issue("i2", 2)
    state: dict = {}
    with pytest.raises(LinearAPIError, match="did not advance"):
        _run(broken, state, page_size=1)
    assert "floor" not in state["streams"]["issues"]


def test_a_repeated_cursor_raises(team):
    class Stuck(FakeLinear):
        def _page(self, items, variables):
            nodes, _ = super()._page(items, variables)
            return nodes, {"hasNextPage": bool(nodes), "endCursor": "1"}

    stuck = Stuck()
    stuck.add_issue("i1", 1)
    stuck.add_issue("i2", 2)
    with pytest.raises(LinearAPIError, match="did not advance"):
        _run(stuck, {}, page_size=1)


def test_rows_sharing_one_timestamp_beyond_a_runs_reach_are_still_all_read():
    tied = FakeLinear()
    for n in range(6):
        tied.add_issue(f"t{n}", 5)
    state: dict = {}
    seen: set[str] = set()
    for _ in range(6):
        rows, stats = _run(tied, state, page_size=2, max_requests=3)
        seen |= {row["id"] for row in rows}
        if not stats["failed"]:
            break
    assert seen == {f"issue:t{n}" for n in range(6)}
    assert "resume_after" not in state["streams"]["issues"]


def test_a_refused_resume_cursor_falls_back_to_the_ceiling_window(team):
    state: dict = {}
    _run(team, state, page_size=1, max_requests=2)
    assert state["streams"]["issues"].get("resume_after")

    original = team.execute

    def refuse_cursors(query, variables=None):
        if variables and variables.get("after"):
            raise LinearAPIError("Linear request failed: GraphQL codes INVALID")
        return original(query, variables)

    team.execute = refuse_cursors
    rows, stats = _run(team, state, page_size=50)
    assert stats["failed"] == 0
    assert {row["id"] for row in rows} >= {"issue:i1", "issue:i2"}


def test_projects_are_read_even_when_the_issue_walk_uses_the_whole_budget(team):
    rows, stats = _run(team, {}, page_size=1, max_requests=2)
    assert "project:p1" in {row["id"] for row in rows}
    assert stats["failed_budget"] == 1


def test_archived_projects_and_comments_follow_include_archived(team):
    seen = []
    original = team.execute

    def record(query, variables=None):
        seen.append((query.split("(")[0].split()[-1], (variables or {}).get("archived")))
        return original(query, variables)

    team.execute = record
    _run(team, {}, include_archived=True)
    _run(team, {}, include_archived=False)
    by_query = {name: {flag for n, flag in seen if n == name} for name, _ in seen}
    assert by_query["LinearTeamProjects"] == {True, False}
    assert by_query["LinearTeamIssues"] == {True, False}


def test_a_missing_team_from_linear_becomes_the_typed_error():
    class Gone:
        rate_limit: dict = {}

        def execute(self, query, variables=None):
            raise LinearEntityNotFoundError("Linear entity not found or not accessible")

    with pytest.raises(LinearTeamNotFoundError):
        _run(Gone(), {})


def test_entity_not_found_from_the_graphql_error_is_recognised_without_echoing_it(monkeypatch):
    body = {"errors": [{"message": "Entity not found: Team - Could not find referenced Team."}]}
    client, _ = _client_with(monkeypatch, [_Response(200, body)])
    with pytest.raises(LinearEntityNotFoundError) as caught:
        client.execute("query A { x }")
    assert "Could not find" not in str(caught.value)


def test_a_deleted_issue_during_comment_paging_keeps_the_comments_read(team):
    for n in range(60):
        team.add_comment(f"x{n:02d}", "i1", 10 + (n % 40), body=f"remark {n:02d}")
    original = team.execute

    def vanish(query, variables=None):
        if "LinearIssueComments" in query:
            raise LinearEntityNotFoundError("gone")
        return original(query, variables)

    team.execute = vanish
    rows, stats = _run(team, {})
    assert stats["failed"] == 0
    assert any(row["id"] == "issue:i1" for row in rows)


def test_a_gateway_429_is_a_rate_limit_not_a_failure(monkeypatch):
    client, _ = _client_with(monkeypatch, [_Response(429, None)])
    with pytest.raises(LinearRateLimitedError):
        client.execute("query A { x }")


def test_a_response_without_data_raises(monkeypatch):
    client, _ = _client_with(monkeypatch, [_Response(200, {})])
    with pytest.raises(LinearAPIError, match="no data"):
        client.execute("query A { x }")


@pytest.mark.parametrize("bad", ["", "   ", "tok\nen", "tok en", "tökën"])
def test_a_malformed_token_is_refused_without_echoing_it(bad):
    with pytest.raises(ValueError) as caught:
        LinearClient(bad)
    assert bad.strip() not in str(caught.value) or not bad.strip()


def test_a_token_pasted_with_a_trailing_newline_is_cleaned():
    assert LinearClient(f"{TOKEN}\n")._headers()["Authorization"] == f"Bearer {TOKEN}"


def test_the_source_rejects_a_malformed_token_when_it_is_built():
    with pytest.raises(ValueError):
        linear_source(team_id=TEAM, access_token="bad token")


def test_page_size_is_clamped_to_what_fits_the_complexity_limit(team, tmp_path):
    import dlt

    sizes = []
    original = team.execute

    def record(query, variables=None):
        sizes.append((variables or {}).get("first"))
        return original(query, variables)

    team.execute = record
    pipeline = dlt.pipeline(
        pipeline_name="linear_clamp",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'l.db'}"),
        dataset_name="linear",
        pipelines_dir=str(tmp_path / "pipelines"),
    )
    pipeline.run(linear_source(team_id=TEAM, service=team, resource_name="linear_t", page_size=500))
    assert sizes and max(sizes) == linear_module.MAX_PAGE_SIZE == 50


def test_comment_queries_stay_scoped_to_the_team_whatever_the_filter_holds(team):
    seen = []
    original = team.execute

    def record(query, variables=None):
        if "LinearTeamComments" in query:
            seen.append(variables["filter"])
        return original(query, variables)

    team.execute = record
    walker = linear_module._Walker(
        team,
        TEAM,
        {},
        _stats(),
        check_active=None,
        include_archived=True,
        page_size=50,
        max_requests=10,
    )
    walker._fetch_comments({"issue": {"team": {"id": {"eq": "someone-elses-team"}}}}, None)

    assert seen == [{"issue": {"team": {"id": {"eq": TEAM}}}}]
