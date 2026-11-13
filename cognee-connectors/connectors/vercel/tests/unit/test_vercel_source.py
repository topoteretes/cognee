# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Vercel source: rendering, the sync model, deletion on evidence only, and the cuts.

The walker runs on a plain dict as its state; the dlt behaviour (merge, hard delete, state kept
or rolled back, one table per resource name, host retire) runs through a real ``dlt.pipeline`` on
SQLite. The fake answers like Vercel's documented API (``fake_vercel.py``). Test names follow the
user scenario of the plan (first sync, repeat, edit, move, deselect, access lost, delete) and its
threat list (token leak, account mixing, wrong deletion, lost changes, provider text in prompts,
personal data).
"""

import json
import math
import random
import time
from types import SimpleNamespace

import pytest
from fake_vercel import (
    CANARIES,
    CANARY_EMAIL,
    DAY_MS,
    MINUTE_MS,
    NOW_MS,
    FakeVercel,
    sha,
)

from cognee_connector_vercel import source as vercel_module
from cognee_connector_vercel.source import (
    DEPLOYMENT_ROUTE,
    PAGE_SIZE,
    VercelAPIError,
    VercelResponseError,
    VercelSourceError,
    _iter_rows,
    build_log_row,
    deployment_row,
    event_text,
    new_stats,
    project_row,
    vercel_source,
)
from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    STRUCTURE_COLUMN,
)

TABLE = "vercel_web"


@pytest.fixture(autouse=True)
def _no_active_dlt_pipeline():
    """A source iterated outside a pipeline keeps its state in the active pipeline, and a
    test that ran a real pipeline leaves it active, so the next test read that state. A CI
    shard orders tests differently from a local run, which hid it."""
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    context = Container()[PipelineContext]
    if context.is_active():
        context.deactivate()


def _team(*, failed: int = 1, page_size: int | None = None) -> FakeVercel:
    """Team acme, project web: a READY production deployment, a preview and ``failed`` ERRORs."""
    fake = FakeVercel(page_size=page_size)
    fake.add_project("prj_web", "web")
    fake.add_deployment("dpl_ready", minutes_ago=30, commit=1)
    fake.add_deployment("dpl_preview", minutes_ago=20, target=None, commit=2)
    for n in range(failed):
        uid = f"dpl_fail{n}"
        fake.add_deployment(
            uid,
            state="ERROR",
            minutes_ago=10 - n,
            commit=10 + n,
            errorCode="BUILD_FAILED",
            errorMessage="Command npm run build exited with 1",
        )
        fake.set_log(uid, "Running build", "\x1b[31mError: Cannot find module 'x'\x1b[0m")
    return fake


def _walk(fake, state=None, **kwargs):
    """One run of the walker; returns (rows, stats, state)."""
    state = {} if state is None else state
    stats = new_stats()
    kwargs.setdefault("project_id", "prj_web")
    kwargs.setdefault("team_id", fake.team_id)
    kwargs.setdefault("account_id", fake.account_id)
    rows = list(_iter_rows(fake, state, stats, **kwargs))
    return rows, stats, state


def _ids(rows) -> list[str]:
    return [row["id"] for row in rows]


def _tombstones(rows) -> list[str]:
    return [row["id"] for row in rows if row["_deleted"]]


# --- rendering ------------------------------------------------------------------------------


def test_the_project_row_reads_named_fields_only():
    fake = _team()
    row = project_row(fake.projects["prj_web"])

    assert row["title"] == "Project web"
    assert row["content"].splitlines() == [
        "Project: web",
        "Framework: nextjs",
        "Git repository: github: acme/web",
        "Production branch: main",
        "Root directory: apps/web",
        "Node.js version: 22.x",
        "Created: 2026-08-12 12:00 UTC",
    ]
    for canary in CANARIES:
        assert canary not in json.dumps(row)


def test_the_project_row_ignores_what_moves_without_a_change():
    fake = _team()
    before = project_row(fake.projects["prj_web"])
    fake.projects["prj_web"].update(
        updatedAt=NOW_MS + 5, targets={}, latestDeployments=[], live=True
    )

    assert project_row(fake.projects["prj_web"]) == before


def test_the_deployment_row_puts_the_facts_a_question_names_first():
    fake = _team()
    row = deployment_row(fake.deployments["dpl_fail0"], "prj_web")

    assert row["title"] == "web: production deployment 2026-11-10 11:50 UTC (dpl_fail0)"
    assert row["content"].splitlines() == [
        "Project: web",
        "Environment: production",
        "State: ERROR",
        "Branch: main",
        f"Commit: {sha(10)}",
        "Commit message: Commit 10",
        "Commit author: Ada Lovelace (ada)",
        "Created by: ada",
        "Error code: BUILD_FAILED",
        "Error message: Command npm run build exited with 1",
        "Created: 2026-11-10 11:50 UTC",
        "Deployment URL: https://web-dpl_fail0.vercel.app",
    ]
    assert len(sha(10)) == 40
    assert row["url"] == "https://vercel.com/acme/web/dpl_fail0"


def test_the_deployment_row_never_reads_an_email_the_raw_meta_or_a_build_time():
    fake = _team()
    row = deployment_row(fake.deployments["dpl_ready"], "prj_web")
    text = json.dumps(row)

    for canary in CANARIES:
        assert canary not in text
    assert "@" not in text
    fake.update("dpl_ready", buildingAt=1, ready=2, updatedAt=3)
    assert deployment_row(fake.deployments["dpl_ready"], "prj_web") == row


@pytest.mark.parametrize(
    ("name", "login", "line"),
    [
        ("Ada Lovelace", "ada", "Commit author: Ada Lovelace (ada)"),
        (None, "ada", "Commit author: ada"),
        ("Ada Lovelace", None, "Commit author: Ada Lovelace"),
        ("ada", "ada", "Commit author: ada"),
        ("Ada\nLovelace\x1b[31m", "ada", "Commit author: Ada Lovelace (ada)"),
        (None, None, None),
    ],
)
@pytest.mark.parametrize("git", ["github", "gitlab", "bitbucket"])
def test_the_commit_author_is_the_name_a_person_has_elsewhere_and_the_login(git, name, login, line):
    """A sibling connector writes the name, a Git source the login: both merge."""
    fake = _team()
    meta = {f"{git}CommitSha": sha(1)}
    if name is not None:
        meta[f"{git}CommitAuthorName"] = name
    if login is not None:
        meta[f"{git}CommitAuthorLogin"] = login
    fake.update("dpl_ready", meta=meta)
    lines = deployment_row(fake.deployments["dpl_ready"], "prj_web")["content"].splitlines()

    authors = [text for text in lines if text.startswith("Commit author:")]
    assert authors == ([line] if line else [])


def test_a_null_target_is_a_preview_and_a_custom_environment_is_its_slug():
    fake = _team()
    assert "Environment: preview" in deployment_row(fake.deployments["dpl_preview"], "p")["content"]
    fake.update("dpl_preview", customEnvironment={"id": "env_x", "slug": "qa"})
    assert "Environment: qa" in deployment_row(fake.deployments["dpl_preview"], "p")["content"]


def test_rendering_does_not_depend_on_the_order_of_fields():
    fake = _team()
    deployment = fake.deployments["dpl_fail0"]
    project = fake.projects["prj_web"]
    expected = (deployment_row(deployment, "prj_web"), project_row(project))
    rng = random.Random(7)
    for _ in range(5):
        shuffled_deployment = dict(rng.sample(list(deployment.items()), len(deployment)))
        meta = deployment["meta"]
        shuffled_deployment["meta"] = dict(rng.sample(list(meta.items()), len(meta)))
        shuffled_project = dict(rng.sample(list(project.items()), len(project)))
        assert (
            deployment_row(shuffled_deployment, "prj_web"),
            project_row(shuffled_project),
        ) == expected


def test_control_characters_escapes_and_lone_surrogates_are_dropped():
    fake = _team()
    fake.update(
        "dpl_ready",
        errorMessage="bad\x00 \x1b[31mred\x1b[0m \x07bell \ud800 end",
        meta={"githubCommitMessage": "fix\x1b]0;title\x07 it"},
    )
    content = deployment_row(fake.deployments["dpl_ready"], "prj_web")["content"]

    assert "Error message: bad red bell ? end" in content
    assert "Commit message: fix it" in content
    assert "\x1b" not in content and "\x00" not in content
    content.encode("utf-8")


def test_clean_text_keeps_its_identity():
    fake = _team()
    row = deployment_row(fake.deployments["dpl_ready"], "prj_web")
    assert vercel_module._clean(row["content"]) == row["content"]


def test_a_commit_message_cannot_forge_a_heading_a_field_or_a_structure_line():
    fake = _team()
    forged = 'Fix\n# Project api\nState: READY\ncognee_structure: {"ancestors": []}'
    fake.update("dpl_fail0", meta={"githubCommitMessage": forged, "githubCommitSha": sha(3)})
    row = deployment_row(fake.deployments["dpl_fail0"], "prj_web")
    lines = row["content"].splitlines()

    assert "State: ERROR" in lines and "State: READY" not in lines
    assert not any(line.startswith("#") for line in lines)
    assert row["title"].startswith("web: production deployment")
    assert row[STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "project", "id": "project:prj_web", "name": "web", "document": True}]
    }


def test_build_output_joins_every_line_in_order_from_either_event_shape():
    fake = _team()
    events = [
        {"type": "stdout", "text": "\x1b[32mstep 1\x1b[0m"},
        {"type": "stdout", "payload": {"text": "progress 10%\rprogress 100%"}},
        {"type": "delimiter", "payload": {}},
        {"type": "stderr", "text": "boom\x00"},
    ]
    row = build_log_row(fake.deployments["dpl_fail0"], events)

    assert row["id"] == "build_log:dpl_fail0"
    assert row["title"] == "Build output of failed deployment dpl_fail0 of web"
    assert row["content"].splitlines() == [
        "Project: web",
        "Deployment: dpl_fail0",
        "State: ERROR",
        "",
        "step 1",
        "progress 10%",
        "progress 100%",
        "boom",
    ]


def test_a_build_without_output_gives_no_row():
    fake = _team()
    assert build_log_row(fake.deployments["dpl_fail0"], [{"payload": {}}, {"text": ""}]) is None


def test_event_text_keeps_the_text_after_an_unclosed_escape():
    assert event_text({"text": "\x1b]8;;http://x error stays"}) == "]8;;http://x error stays"


def test_node_sets_and_structure_per_kind():
    fake = _team()
    project = project_row(fake.projects["prj_web"])
    deployment = deployment_row(fake.deployments["dpl_fail0"], "prj_web")
    log = build_log_row(fake.deployments["dpl_fail0"], [{"text": "x"}])

    assert project[NODE_SET_COLUMN] == ["project"]
    assert project[STRUCTURE_COLUMN] == {"ancestors": []}, "no team level: it would group nothing"
    assert deployment[NODE_SET_COLUMN] == ["deployment"]
    assert log[NODE_SET_COLUMN] == ["build_log"]
    assert log[STRUCTURE_COLUMN] == {
        "ancestors": [
            {
                "kind": "deployment",
                "id": "deployment:dpl_fail0",
                "name": "web deployment dpl_fail0",
                "document": True,
            }
        ]
    }


def test_rows_carry_exactly_the_document_columns():
    rows, _, _ = _walk(_team(), include_build_logs=True)
    columns = {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN, STRUCTURE_COLUMN}
    assert all(set(row) == columns for row in rows)


# --- 1, 2, 3, 4a: first sync, repeat, edit, promotion -----------------------------------------


def test_first_sync_yields_the_project_and_every_deployment():
    rows, stats, state = _walk(_team())

    assert _ids(rows) == [
        "project:prj_web",
        "deployment:dpl_fail0",
        "deployment:dpl_preview",
        "deployment:dpl_ready",
    ]
    assert stats == {**new_stats(), "scanned": 4}
    assert state["v"] == 1


def test_an_unchanged_sync_yields_nothing():
    fake = _team()
    _, _, state = _walk(fake)

    rows, stats, _ = _walk(fake, state)

    assert rows == []
    assert stats == {**new_stats(), "scanned": 4, "skipped": 4}


def test_a_state_change_re_renders_only_that_deployment():
    fake = _team()
    fake.set_state("dpl_ready", "BUILDING")
    _, _, state = _walk(fake)
    fake.set_state("dpl_ready", "READY")

    rows, _, _ = _walk(fake, state)

    assert _ids(rows) == ["deployment:dpl_ready"]
    assert "State: READY" in rows[0]["content"]


def test_a_project_rename_re_renders_the_project_row_only():
    fake = _team()
    _, _, state = _walk(fake)
    fake.rename_project("prj_web", "website")

    rows, _, _ = _walk(fake, state)

    assert _ids(rows) == ["project:prj_web"]
    assert rows[0]["title"] == "Project website"


def test_a_promotion_re_renders_the_deployment_with_its_substate():
    fake = _team()
    _, _, state = _walk(fake)
    fake.update("dpl_ready", readySubstate="PROMOTED")

    rows, _, _ = _walk(fake, state)

    assert _ids(rows) == ["deployment:dpl_ready"]
    assert "State: READY (PROMOTED)" in rows[0]["content"]


# --- 4b, 6: moved out of the account, access lost -------------------------------------------


def test_a_project_of_another_account_stops_the_run_before_any_row():
    fake = _team()
    _, _, state = _walk(fake)
    before = json.loads(json.dumps(state))
    fake.projects["prj_web"]["accountId"] = "team_other"  # transferred to another team

    rows, stats, state = _walk(fake, state)

    assert rows == []
    assert stats == {**new_stats(), "failed": 1, "failed_auth": 1}
    assert state == before
    assert fake.paths("/v7") == fake.paths("/v7")[:1], "the listing is not read again"


def test_a_token_of_another_team_raises_a_typed_auth_error():
    fake = _team()
    stats = new_stats()

    with pytest.raises(vercel_module.VercelAuthError, match="HTTP 403"):
        list(
            _iter_rows(
                fake, {}, stats, project_id="prj_web", team_id="team_other", account_id="team_other"
            )
        )
    assert stats["failed_auth"] == 1 and stats["failed"] == 1


def test_a_personal_account_sends_no_team_id_and_checks_the_user_id():
    fake = FakeVercel(team_id=None)
    fake.add_project("prj_web", "web")
    fake.add_deployment("dpl_ready")

    rows, stats, _ = _walk(fake, team_id=None, account_id="user_ada")

    assert _ids(rows) == ["project:prj_web", "deployment:dpl_ready"]
    assert all("teamId" not in params for _, params in fake.calls)
    assert rows[0][STRUCTURE_COLUMN] == {"ancestors": []}
    rows, stats, _ = _walk(fake, team_id=None, account_id="user_bob")
    assert rows == [] and stats["failed_auth"] == 1


def test_a_project_named_instead_of_its_id_is_refused():
    fake = _team()
    fake.projects["web"] = fake.projects["prj_web"]

    with pytest.raises(VercelResponseError, match="pass the project id"):
        _walk(fake, project_id="web")


@pytest.mark.parametrize("status", [401, 403])
def test_access_lost_after_rows_keeps_them_and_forgets_nothing(status):
    fake = _team(failed=3, page_size=2)
    _, _, state = _walk(fake)
    fake.delete("dpl_ready")  # a real deletion this run never gets to confirm
    fake.set_state("dpl_fail2", "CANCELED")
    answer = fake.answer

    def second_page_fails(path, params):
        if path == "/v7/deployments" and "until" in params:
            return status, {"error": {"code": "forbidden"}}
        return answer(path, params)

    fake.answer = second_page_fails

    rows, stats, state = _walk(fake, state)

    assert _ids(rows) == ["deployment:dpl_fail2"], "the first page's change is kept"
    assert _tombstones(rows) == []
    assert stats["failed_auth"] == 1 and stats["failed"] == 1
    assert "dpl_ready" in state["created"]


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_token_before_any_row_raises_so_the_caller_sees_it(status):
    """A cut here returned an empty success: an SDK user with an expired token saw nothing."""
    fake = _team()
    _, _, state = _walk(fake)
    fake.fail("/v9/projects", status)
    stats = new_stats()

    with pytest.raises(vercel_module.VercelAuthError, match=f"HTTP {status}"):
        list(
            _iter_rows(
                fake,
                state,
                stats,
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
            )
        )
    assert stats["failed_auth"] == 1 and stats["failed"] == 1


# --- 7: deletion on evidence only -------------------------------------------------------------


def test_a_deployment_listed_as_deleted_is_forgotten_with_its_build_output():
    fake = _team()
    _, _, state = _walk(fake, include_build_logs=True)
    fake.set_state("dpl_fail0", "DELETED")

    rows, stats, state = _walk(fake, state, include_build_logs=True)

    assert _tombstones(rows) == ["deployment:dpl_fail0", "build_log:dpl_fail0"]
    assert stats["deleted"] == 2
    assert "dpl_fail0" not in state["created"]
    rows, _, _ = _walk(fake, state, include_build_logs=True)
    assert rows == [], "a deleted deployment still listed is forgotten once"


def test_a_known_deployment_missing_from_a_complete_listing_is_forgotten_on_a_404():
    fake = _team()
    _, _, state = _walk(fake)
    fake.delete("dpl_preview")

    rows, stats, _ = _walk(fake, state)

    assert _tombstones(rows) == ["deployment:dpl_preview"]
    assert fake.paths("/v13/deployments/") == ["/v13/deployments/dpl_preview"]
    assert stats["deleted"] == 1 and stats["failed"] == 0


def test_a_deployment_that_retention_removed_is_forgotten_on_a_410():
    fake = _team()
    _, _, state = _walk(fake)
    fake.expire("dpl_preview")

    rows, _, _ = _walk(fake, state)

    assert _tombstones(rows) == ["deployment:dpl_preview"]


@pytest.mark.parametrize("variant", ["full", "minimal"])
def test_a_known_deployment_the_listing_missed_is_kept_on_either_v13_shape(variant):
    """The single read answers ``id``/``createdAt`` (and the minimal variant no ``projectId``),
    not the list's ``uid``/``created``: reading it with the list's shape raised every run."""
    fake = _team()
    _, _, state = _walk(fake)
    fake.unlisted.add("dpl_preview")
    if variant == "minimal":
        fake.minimal_reads.add("dpl_preview")

    rows, stats, state = _walk(fake, state)

    assert rows == [], "kept, and not rendered from another endpoint's shape"
    assert stats["failed"] == 0 and stats["check_errors"] == 0, "a live answer, not an error"
    assert fake.paths("/v13/") == ["/v13/deployments/dpl_preview"]
    assert "dpl_preview" in state["created"]


@pytest.mark.parametrize(
    "answer",
    [
        {"uid": "dpl_preview", "readyState": "READY", "created": 1},
        {"id": "dpl_other", "readyState": "READY"},
        {"id": "dpl_preview", "readyState": "READY", "projectId": "prj_api"},
        {"id": "dpl_preview"},
        ["dpl_preview"],
    ],
)
def test_a_single_read_of_no_documented_shape_keeps_the_deployment_and_counts_it(answer):
    fake = _team()
    _, _, state = _walk(fake)
    fake.unlisted.add("dpl_preview")
    fake.answers["/v13/deployments/dpl_preview"] = answer

    rows, stats, state = _walk(fake, state)

    assert _tombstones(rows) == []
    assert stats["failed"] == 0 and stats["check_errors"] == 1
    assert "dpl_preview" in state["created"]
    fake.calls.clear()
    _walk(fake, state)
    assert fake.paths("/v13/") == ["/v13/deployments/dpl_preview"], "asked again next pass"


def test_an_empty_listing_forgets_nothing_vercel_still_shows():
    fake = _team()
    _, _, state = _walk(fake)
    fake.unlisted.update(fake.deployments)

    rows, _, _ = _walk(fake, state)

    assert _tombstones(rows) == []
    assert len(fake.paths("/v13/deployments/")) == 3


def test_a_deployment_that_left_the_window_is_kept_and_never_checked():
    fake = _team()
    fake.add_deployment("dpl_old", minutes_ago=40 * 24 * 60)
    _, _, state = _walk(fake)
    assert "dpl_old" in state["created"]
    fake.calls.clear()

    rows, _, state = _walk(fake, state, since_ms=NOW_MS - 30 * DAY_MS)

    assert _tombstones(rows) == []
    assert fake.paths("/v13/") == [], "a deployment outside the window is not read"
    assert "dpl_old" not in state["created"], "the state holds the window only"
    assert all(params.get("since") == NOW_MS - 30 * DAY_MS for _, params in fake.calls[1:])


def test_a_deployment_deleted_after_it_left_the_window_keeps_its_document():
    fake = _team()
    fake.add_deployment("dpl_old", minutes_ago=40 * 24 * 60)
    _, _, state = _walk(fake, since_ms=NOW_MS - 60 * DAY_MS)
    fake.delete("dpl_old")

    rows, _, _ = _walk(fake, state, since_ms=NOW_MS - 30 * DAY_MS)

    assert _tombstones(rows) == []


def test_a_project_that_answers_404_forgets_nothing():
    fake = _team()
    _, _, state = _walk(fake)
    before = json.loads(json.dumps(state))
    del fake.projects["prj_web"]
    fake.calls.clear()

    rows, stats, state = _walk(fake, state)

    assert rows == []
    assert stats == {**new_stats(), "unresolved": 1}
    assert state == before
    assert fake.paths("/v7") == []


@pytest.mark.parametrize(
    "body",
    [
        {"pagination": {"count": 0, "next": None, "prev": None}},
        {"deployments": [], "pagination": {"count": 0}},
        {"deployments": []},
        [],
        {"deployments": "none", "pagination": {"next": None}},
    ],
)
def test_a_listing_of_the_wrong_shape_raises_and_forgets_nothing(body):
    fake = _team()
    _, _, state = _walk(fake)
    fake.delete("dpl_ready")
    fake.answers["/v7/deployments"] = body

    with pytest.raises(VercelResponseError, match="/v7/deployments"):
        _walk(fake, state)
    assert fake.paths("/v13/") == []


@pytest.mark.timeout(30)  # without the guard the walk pages forever
def test_a_cursor_that_does_not_move_raises():
    fake = _team(failed=3, page_size=2)
    fake.stuck_cursor = True

    with pytest.raises(VercelResponseError, match="cursor did not move"):
        _walk(fake)


@pytest.mark.parametrize(
    "item",
    [
        {"uid": "dpl_x", "name": "web", "projectId": "prj_web", "created": 1},
        {"uid": "../x", "name": "web", "projectId": "prj_web", "created": 1, "readyState": "READY"},
        {
            "uid": "dpl_x",
            "name": "web",
            "projectId": "prj_api",
            "created": 1,
            "readyState": "READY",
        },
        "dpl_x",
    ],
)
def test_a_deployment_of_the_wrong_shape_or_project_raises(item):
    fake = _team()
    fake.answers["/v7/deployments"] = {
        "deployments": [item],
        "pagination": {"count": 1, "next": None, "prev": None},
    }
    stats = new_stats()

    with pytest.raises(VercelResponseError, match="/v7/deployments: a deployment"):
        list(
            _iter_rows(
                fake, {}, stats, project_id="prj_web", team_id="team_acme", account_id="team_acme"
            )
        )
    assert stats["failed"] == 1


def test_a_rate_limit_is_a_cut_that_forgets_nothing_and_is_not_retried():
    fake = _team(failed=3, page_size=2)
    _, _, state = _walk(fake)
    fake.delete("dpl_ready")
    fake.calls.clear()
    fake.quota = 2  # the project and the first page

    rows, stats, _ = _walk(fake, state)

    assert _tombstones(rows) == []
    assert stats["failed_rate_limit"] == 1 and stats["failed"] == 1
    assert len(fake.calls) == 3, "the 429 is not asked again"


def test_a_cut_during_the_checks_keeps_the_tombstones_already_settled():
    fake = _team()
    _, _, state = _walk(fake)
    fake.delete("dpl_preview")
    fake.delete("dpl_ready")
    fake.fail("/v13/deployments/dpl_ready", 429)

    rows, stats, state = _walk(fake, state)

    assert _tombstones(rows) == ["deployment:dpl_preview"]
    assert stats["failed_rate_limit"] == 1
    assert "dpl_ready" in state["created"], "the unconfirmed one is asked again next run"


def test_server_errors_cut_with_failed_unavailable():
    fake = _team()
    fake.fail("/v7/deployments", 503)

    rows, stats, _ = _walk(fake)

    assert _ids(rows) == ["project:prj_web"]
    assert stats["failed_unavailable"] == 1 and stats["failed"] == 1


def test_another_error_answer_raises_and_counts_a_failure():
    fake = _team()
    fake.fail("/v7/deployments", 400)
    stats = new_stats()

    with pytest.raises(VercelAPIError, match="HTTP 400"):
        list(
            _iter_rows(
                fake, {}, stats, project_id="prj_web", team_id="team_acme", account_id="team_acme"
            )
        )
    assert stats["failed"] == 1 and stats["failed_error"] == 1 and stats["failed_response"] == 0


def test_two_deployments_of_one_millisecond_at_a_page_boundary_are_both_read():
    """``until`` is exclusive: asking for ``next`` itself would skip ``dpl_b``."""
    fake = FakeVercel(page_size=3)
    fake.add_project("prj_web", "web")
    for uid, created in (
        ("dpl_x", NOW_MS + 2),
        ("dpl_y", NOW_MS + 1),
        ("dpl_c", NOW_MS),
        ("dpl_b", NOW_MS),  # the same millisecond as dpl_c, on the next page
        ("dpl_d", NOW_MS - 1),
    ):
        fake.add_deployment(uid, created=created, createdAt=created)

    rows, _, _ = _walk(fake)

    assert sorted(_ids(rows)[1:]) == [
        "deployment:dpl_b",
        "deployment:dpl_c",
        "deployment:dpl_d",
        "deployment:dpl_x",
        "deployment:dpl_y",
    ]


def test_pages_are_read_with_until_one_millisecond_after_next():
    fake = _team(failed=3, page_size=2)

    _walk(fake)

    listings = [params for path, params in fake.calls if path == "/v7/deployments"]
    assert [p.get("until") for p in listings] == [
        None,
        NOW_MS - 9 * MINUTE_MS + 1,
        NOW_MS - 10 * MINUTE_MS + 1,
        NOW_MS - 20 * MINUTE_MS + 1,
    ]
    assert all(p["limit"] == PAGE_SIZE and p["projectId"] == "prj_web" for p in listings)


def test_a_budget_cut_keeps_rows_and_the_next_run_reaches_the_full_state():
    fake = _team(failed=3, page_size=2)
    full_rows, _, full_state = _walk(_team(failed=3, page_size=2), include_build_logs=True)

    rows, stats, state = _walk(fake, max_requests=3, include_build_logs=True)
    assert stats["failed_budget"] == 1 and stats["failed"] == 1
    assert rows, "what was reached is kept"
    more, stats, state = _walk(fake, state, include_build_logs=True)

    assert stats["failed"] == 0
    assert sorted(_ids(rows + more)) == sorted(_ids(full_rows))
    assert state == full_state


def test_check_active_runs_before_every_request_and_stops_the_walk():
    fake = _team()
    checks = []

    class Paused(Exception):
        pass

    def check_active():
        checks.append(len(fake.calls))
        if len(checks) == 2:
            raise Paused

    stats = new_stats()
    with pytest.raises(Paused):
        list(
            _iter_rows(
                fake,
                {},
                stats,
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
                check_active=check_active,
            )
        )
    assert checks == [0, 1], "the second request never went out"
    assert set(stats) == set(new_stats()), "the counts are complete on a stop"


def test_stats_are_non_negative_ints_present_from_the_start():
    stats = new_stats()
    assert set(stats) == {
        "scanned",
        "skipped",
        "deleted",
        "failed",
        "unresolved",
        "failed_rate_limit",
        "failed_budget",
        "failed_auth",
        "failed_unavailable",
        "failed_response",
        "failed_error",
        "failed_state",
        "check_errors",
        "build_log_errors",
    }
    assert all(type(value) is int and value == 0 for value in stats.values())


def test_a_state_of_another_version_is_refused():
    fake = _team()
    with pytest.raises(VercelSourceError, match="state version 2"):
        _walk(fake, {"v": 2})


# --- a pass that spans runs ----------------------------------------------------------------


def _many(count: int) -> FakeVercel:
    fake = FakeVercel()
    fake.add_project("prj_web", "web")
    for n in range(count):
        fake.add_deployment(f"dpl_{n:03d}", minutes_ago=n + 1)
    return fake


def _until_done(fake, state, **kwargs):
    """Run until a pass completes (a budget may still cut the head read); return every row."""
    every = []
    for _ in range(20):
        rows, _, state = _walk(fake, state, **kwargs)
        every += rows
        if "pass" not in state:
            return every, state
    raise AssertionError("the pass never completed")


@pytest.mark.parametrize("budget", [2, 3])
def test_a_budget_smaller_than_the_listing_gets_through_it_over_several_runs(budget):
    """With the listing re-read from the top, budget 2 emitted 101 of 251 rows on every run."""
    fake = _many(251)
    full, _, _ = _walk(_many(251))

    rows, state = _until_done(fake, {}, max_requests=budget)

    assert sorted(set(_ids(rows))) == sorted(_ids(full))
    assert "pass" not in state, "the pass completed"


def test_a_partial_pass_forgets_nothing_and_the_completed_pass_does():
    fake = _many(251)
    _, _, state = _walk(fake)
    fake.delete("dpl_250")  # on the last page

    rows, stats, state = _walk(fake, state, max_requests=2)
    assert stats["failed_budget"] == 1 and _tombstones(rows) == []
    assert state["pass"]["until"] is not None, "the position is kept"

    rows, state = _until_done(fake, state, max_requests=2)
    assert _tombstones(rows) == ["deployment:dpl_250"]


def test_a_run_continues_the_pass_and_reads_the_head_again():
    fake = _many(251)
    _, _, state = _walk(fake)
    _, _, state = _walk(fake, state, max_requests=2)  # a pass cut after its first page
    fake.add_deployment("dpl_new", minutes_ago=0)
    fake.set_state("dpl_000", "ERROR", errorCode="OOM")
    fake.calls.clear()

    rows, stats, state = _walk(fake, state)

    assert stats["failed"] == 0 and "pass" not in state
    assert {"deployment:dpl_new", "deployment:dpl_000"} <= set(_ids(rows))
    listings = [params.get("until") for path, params in fake.calls if path == "/v7/deployments"]
    assert listings[0] is None, "the head is read first"
    assert listings[1] is not None, "then the pass continues where it stopped"


@pytest.mark.parametrize("trouble", [503, 400, None, "shape"])
def test_a_deletion_check_that_cannot_be_read_never_holds_the_pass_open(trouble):
    """One known deployment the listing lacks, whose own read always fails: it used to cut (5xx)
    or raise (400) every run before the head, so no edit and no new deployment ever landed."""
    fake = _team()
    _, _, state = _walk(fake)
    fake.unlisted.add("dpl_preview")
    if trouble == "shape":
        fake.answers["/v13/deployments/dpl_preview"] = {"uid": "dpl_preview"}
    else:
        fake.fail("/v13/deployments/dpl_preview", trouble)

    rows, stats, state = _walk(fake, state)
    assert stats["failed"] == 0 and stats["check_errors"] == 1
    assert "pass" not in state, "the pass completed"
    assert "dpl_preview" in state["created"], "kept, asked again next pass"
    fake.add_deployment("dpl_new", minutes_ago=1)
    fake.set_state("dpl_ready", "ERROR")

    rows, stats, state = _walk(fake, state)

    assert {"deployment:dpl_new", "deployment:dpl_ready"} <= set(_ids(rows))
    assert _tombstones(rows) == []
    assert fake.paths("/v13/").count("/v13/deployments/dpl_preview") == 2


@pytest.mark.parametrize("read", ["check", "build_log"])
@pytest.mark.parametrize("body", ["not json", "wrong shape"])
def test_a_200_that_is_not_json_or_of_the_wrong_shape_on_one_item_never_stops_the_sync(read, body):
    """The client raises VercelResponseError for a 200 that is not JSON; on a per-item read it
    raised the run on every sync, so nothing new ever loaded."""
    fake = _team()
    _, _, state = _walk(fake, include_build_logs=read == "build_log")
    if read == "check":
        fake.unlisted.add("dpl_preview")
        path = "/v13/deployments/dpl_preview"
    else:
        fake.set_state("dpl_ready", "ERROR")
        path = "/v3/deployments/dpl_ready/events"
    if body == "not json":
        fake.answers[path] = ValueError("not json")
    else:
        fake.answers[path] = {"unexpected": True}
    reason = "check_errors" if read == "check" else "build_log_errors"
    fake.add_deployment("dpl_new", minutes_ago=1)

    rows, stats, state = _walk(fake, state, include_build_logs=read == "build_log")

    assert "deployment:dpl_new" in _ids(rows), "the rest of the sync still loads"
    assert stats["failed"] == 0 and stats[reason] == 1
    assert _tombstones(rows) == []
    fake.calls.clear()
    _walk(fake, state, include_build_logs=read == "build_log")
    assert fake.paths(path) == [path], "asked again in the next pass"


@pytest.mark.parametrize("path", ["/v9/projects/prj_web", "/v7/deployments"])
def test_a_200_that_is_not_json_on_the_project_or_the_listing_still_raises(path):
    fake = _team()
    fake.answers[path] = ValueError("not json")

    with pytest.raises(VercelResponseError, match="without a JSON body"):
        _walk(fake)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_or_a_rate_limit_on_a_deletion_check_still_stops_the_run(status):
    fake = _team()
    _, _, state = _walk(fake)
    fake.unlisted.add("dpl_preview")
    fake.fail("/v13/deployments/dpl_preview", status)

    fake.rename_project("prj_web", "website")  # a row goes out first, so a 401/403 is a cut

    _, stats, state = _walk(fake, state)

    reason = "failed_rate_limit" if status == 429 else "failed_auth"
    assert stats[reason] == 1 and stats["check_errors"] == 0
    assert state["pass"]["pending"] == ["dpl_preview"], "checked again on the next run"


def test_a_new_deployment_arrives_on_the_next_run_of_a_long_pass():
    """Head first: with the head read after the listing, it landed only when the pass ended."""
    fake = _many(1000)
    _, _, state = _walk(fake, max_requests=3)
    fake.add_deployment("dpl_new", minutes_ago=0)
    fake.set_state("dpl_005", "ERROR")

    rows, stats, state = _walk(fake, state, max_requests=3)

    assert {"deployment:dpl_new", "deployment:dpl_005"} <= set(_ids(rows))
    assert stats["failed_budget"] == 1 and "pass" in state, "the pass still moved on"


def test_with_a_budget_of_two_a_continued_run_reads_no_head_and_still_moves_on():
    fake = _many(251)
    _, _, state = _walk(fake, max_requests=2)
    reached = state["pass"]["until"]
    fake.calls.clear()

    _walk(fake, state, max_requests=2)

    listings = [params.get("until") for path, params in fake.calls if path == "/v7/deployments"]
    assert listings == [reached]


@pytest.mark.parametrize(
    "setup",
    ["cut", "raise", "check", "build_log", "check_and_build_log"],
)
def test_failed_is_the_sum_of_the_reason_keys_and_item_errors_are_never_in_it(setup):
    fake = _team(failed=2)
    _, _, state = _walk(fake)
    if setup in ("check", "check_and_build_log"):
        fake.unlisted.update({"dpl_preview", "dpl_ready"})
        fake.fail("/v13/deployments/dpl_preview", 503)
        fake.fail("/v13/deployments/dpl_ready", 400)
    if setup in ("build_log", "check_and_build_log"):
        fake.fail("/v3/deployments/dpl_fail0/events", 500)
        fake.fail("/v3/deployments/dpl_fail1/events", 400)
    if setup == "cut":
        fake.fail("/v7/deployments", 429)
    stats = new_stats()
    kwargs = {"project_id": "prj_web", "team_id": "team_acme", "account_id": "team_acme"}
    if setup == "raise":
        fake.answers["/v7/deployments"] = {"deployments": []}
        with pytest.raises(VercelResponseError, match="without deployments or pagination"):
            list(_iter_rows(fake, state, stats, **kwargs))
    else:
        list(_iter_rows(fake, state, stats, include_build_logs=True, **kwargs))

    reasons = sum(value for key, value in stats.items() if key.startswith("failed_"))
    assert stats["failed"] == reasons
    items = stats["check_errors"] + stats["build_log_errors"]
    if setup in ("cut", "raise"):
        assert stats["failed"] == 1 and items == 0
    else:
        # An item that cannot be read is not a failed run: a host blocks retiring the
        # connection's other tables on ``failed``.
        assert stats["failed"] == 0 and items == {"check": 2, "build_log": 2}.get(setup, 4)


def test_build_output_entries_leave_with_their_deployment():
    """A retry entry was read every pass, and a no-output entry kept, after the deployment left
    the window."""
    fake = _team(failed=2)
    fake.add_deployment("dpl_old", state="ERROR", minutes_ago=2 * 24 * 60)
    fake.fail("/v3/deployments/dpl_old/events", 500)
    fake.add_deployment("dpl_quiet", state="ERROR", minutes_ago=2 * 24 * 60 + 5)
    fake.set_log("dpl_quiet")
    fake.set_log("dpl_fail1")
    _, _, state = _walk(fake, since_ms=NOW_MS - 3 * DAY_MS, include_build_logs=True)
    assert state["logs"]["dpl_old"]["status"] == "retry"
    assert state["logs"]["dpl_quiet"]["status"] == "none"
    assert state["logs"]["dpl_fail1"]["status"] == "none"
    fake.calls.clear()

    _, _, state = _walk(fake, state, since_ms=NOW_MS - 1 * DAY_MS, include_build_logs=True)
    fake.delete("dpl_fail1")
    _, _, state = _walk(fake, state, since_ms=NOW_MS - 1 * DAY_MS, include_build_logs=True)

    assert fake.paths("/v3/deployments/dpl_old") == [], "not asked once it left the window"
    assert set(state["logs"]) == set(), "retry, none-left-the-window and none-deleted all gone"


# --- build output -----------------------------------------------------------------------------


def test_build_output_is_off_by_default_and_never_requested():
    fake = _team()

    rows, _, _ = _walk(fake)

    assert fake.paths("/v3/") == []
    assert not any(row["id"].startswith("build_log:") for row in rows)


def test_build_output_is_read_once_per_failed_deployment():
    fake = _team(failed=2)

    rows, _, state = _walk(fake, include_build_logs=True)
    assert sorted(fake.paths("/v3/")) == [
        "/v3/deployments/dpl_fail0/events",
        "/v3/deployments/dpl_fail1/events",
    ]
    assert all(
        params["limit"] == -1 and params["follow"] == 0
        for path, params in fake.calls
        if path.startswith("/v3/")
    ), "every line, read once, never the live stream"
    log = next(row for row in rows if row["id"] == "build_log:dpl_fail0")
    assert "Error: Cannot find module 'x'" in log["content"]
    fake.calls.clear()

    _walk(fake, state, include_build_logs=True)

    assert fake.paths("/v3/") == []


def test_build_output_vercel_no_longer_serves_gives_no_row_and_keeps_the_deployment():
    fake = _team()
    fake.fail("/v3/deployments/dpl_fail0/events", 410)

    rows, stats, _ = _walk(fake, include_build_logs=True)

    assert "deployment:dpl_fail0" in _ids(rows)
    assert "build_log:dpl_fail0" not in _ids(rows)
    assert stats["failed"] == 0


@pytest.mark.parametrize("trouble", [500, 400, None, {"events": []}, ["a line"]])
def test_build_output_that_cannot_be_read_is_counted_and_the_walk_goes_on(trouble):
    """A poison build (5xx, a timeout, a 4xx, a wrong body) must not hold back the rows and the
    deletions after it: it is counted, and asked again in the next pass only."""
    fake = _team(failed=2)
    _, _, state = _walk(fake)
    fake.delete("dpl_ready")
    if isinstance(trouble, int) or trouble is None:
        fake.fail("/v3/deployments/dpl_fail1/events", trouble)
    else:
        fake.answers["/v3/deployments/dpl_fail1/events"] = trouble

    rows, stats, state = _walk(fake, state, include_build_logs=True)

    assert "build_log:dpl_fail0" in _ids(rows)
    assert _tombstones(rows) == ["deployment:dpl_ready"], "the deletion below still lands"
    assert stats["failed"] == 0 and stats["build_log_errors"] == 1, "an item, not the run"
    assert state["logs"]["dpl_fail1"]["status"] == "retry"
    fake.heal()
    fake.answers.clear()
    rows, stats, _ = _walk(fake, state, include_build_logs=True)
    assert _ids(rows) == ["build_log:dpl_fail1"] and stats["failed"] == 0


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_or_a_rate_limit_on_build_output_still_stops_the_run(status):
    fake = _team()
    fake.fail("/v3/deployments", status)

    _, stats, state = _walk(fake, include_build_logs=True)

    reason = "failed_rate_limit" if status == 429 else "failed_auth"
    assert stats[reason] == 1 and stats["build_log_errors"] == 0
    assert state["logs"]["dpl_fail0"]["status"] == "queued", "read on the next run"


@pytest.mark.parametrize("answer", ["404", "410", "empty"])
def test_build_output_that_is_empty_or_gone_is_not_asked_again(answer):
    fake = _team()
    if answer == "empty":
        fake.set_log("dpl_fail0")
    else:
        fake.fail("/v3/deployments/dpl_fail0/events", int(answer))

    rows, stats, state = _walk(fake, include_build_logs=True)
    assert "build_log:dpl_fail0" not in _ids(rows) and stats["failed"] == 0
    fake.heal()
    fake.calls.clear()
    _walk(fake, state, include_build_logs=True)

    assert fake.paths("/v3/") == []


def test_switching_build_output_off_forgets_what_was_synced():
    fake = _team()
    _, _, state = _walk(fake, include_build_logs=True)

    rows, stats, _ = _walk(fake, state)

    assert _tombstones(rows) == ["build_log:dpl_fail0"]
    assert stats["deleted"] == 1


# --- the factory -------------------------------------------------------------------------------


def test_the_resource_is_a_merge_table_named_resource_name():
    source = vercel_source(
        resource_name=TABLE, project_id="prj_web", team_id="team_acme", service=_team()
    )

    assert source.name == TABLE
    assert source.write_disposition == "merge"
    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "vercel"
    assert getattr(source, dlt_utils.PIPELINE_SCOPE_ATTR) == TABLE
    columns = source.columns
    assert columns["_deleted"]["hard_delete"] is True
    assert columns["url"]["nullable"] is True
    assert source.cognee_sync_stats == new_stats(), "every key before the first row"


def test_a_service_alone_is_enough_and_no_environment_is_read(monkeypatch):
    monkeypatch.setenv("VERCEL_TOKEN", "from-the-environment")
    with pytest.raises(ValueError, match="access token is empty"):
        vercel_source(resource_name=TABLE, project_id="prj_web", team_id="team_acme")
    vercel_source(resource_name=TABLE, project_id="prj_web", team_id="team_acme", service=_team())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resource_name": ""}, "resource_name"),
        ({"resource_name": "_dlt_loads"}, "_dlt_"),
        ({"project_id": ""}, "project_id"),
        ({"project_id": ["prj_web"]}, "project_id"),
        ({"project_id": "prj_web/../x"}, "project_id"),
        ({"team_id": None}, "account_id"),
        ({"team_id": "team acme"}, "team_id"),
        ({"max_requests": -1}, "max_requests"),
        ({"max_requests": 1}, "never moves the pass|the project and a page"),
        ({"max_requests": True}, "max_requests"),
        ({"lookback_days": 0}, "lookback_days"),
        ({"lookback_days": 1.5}, "lookback_days"),
        ({"include_build_logs": "yes"}, "include_build_logs"),
    ],
)
def test_the_factory_refuses_a_bad_argument_before_any_network(kwargs, match):
    fake = _team()
    arguments = {"resource_name": TABLE, "project_id": "prj_web", "team_id": "team_acme"}
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        vercel_source(service=fake, **arguments)
    assert fake.calls == []


def test_the_factory_takes_keyword_arguments_only():
    with pytest.raises(TypeError, match="positional"):
        vercel_source(TABLE, "prj_web")  # type: ignore[misc]


def test_an_older_cognee_is_refused(monkeypatch):
    monkeypatch.setattr(dlt_utils, "host_document_sync_version", lambda: 2)
    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION >= 3"):
        vercel_source(
            resource_name=TABLE, project_id="prj_web", team_id="team_acme", service=_team()
        )


def test_a_window_is_worked_out_on_each_run(monkeypatch):
    fake = _team()
    clock = [NOW_MS / 1000]
    # The module's own clock only: time.time stays real for the rest of the process.
    monkeypatch.setattr(vercel_module, "time", SimpleNamespace(time=lambda: clock[0]))
    source = vercel_source(
        resource_name=TABLE,
        project_id="prj_web",
        team_id="team_acme",
        service=fake,
        lookback_days=30,
    )

    list(source)
    clock[0] += 86_400
    list(source)

    sinces = [params["since"] for path, params in fake.calls if path == "/v7/deployments"]
    assert sinces == [NOW_MS - 30 * DAY_MS, NOW_MS - 29 * DAY_MS]


# --- through a real dlt pipeline ------------------------------------------------------------


def _pipeline(tmp_path):
    import dlt

    return dlt.pipeline(
        pipeline_name="vercel_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'v.db'}"),
        dataset_name="vercel",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


def _source(fake, name: str = TABLE, **kwargs):
    kwargs.setdefault("project_id", "prj_web")
    kwargs.setdefault("team_id", "team_acme")
    return vercel_source(service=fake, resource_name=name, **kwargs)


def _staged(tmp_path, table: str = TABLE) -> dict[str, str]:
    pipeline = _pipeline(tmp_path)
    with (
        pipeline.sql_client() as sql,
        sql.execute_query(f'SELECT id, content FROM "{table}" ORDER BY id') as cursor,
    ):
        return {row[0]: row[1] for row in cursor.fetchall()}


def _run(tmp_path, source) -> dict[str, str]:
    """Run ``source`` the way ingest_dlt_source does; return the staged ``id -> content``."""
    _pipeline(tmp_path).run(source, columns=DOCUMENT_COLUMN_HINTS)
    return _staged(tmp_path, source.name)


def test_edits_and_deletions_survive_between_real_dlt_runs(tmp_path):
    fake = _team()
    first = _source(fake)
    assert list(_run(tmp_path, first)) == [
        "deployment:dpl_fail0",
        "deployment:dpl_preview",
        "deployment:dpl_ready",
        "project:prj_web",
    ]
    assert first.cognee_sync_stats["failed"] == 0

    fake.set_state("dpl_ready", "DELETED")
    fake.delete("dpl_preview")
    fake.update("dpl_fail0", errorCode="OOM")
    staged = _run(tmp_path, _source(fake))

    assert list(staged) == ["deployment:dpl_fail0", "project:prj_web"]
    assert "Error code: OOM" in staged["deployment:dpl_fail0"]


def test_a_cut_run_keeps_its_rows_and_state_and_the_next_one_completes(tmp_path):
    fake = _team(failed=3, page_size=2)
    cut = _source(fake, max_requests=2)
    staged = _run(tmp_path, cut)
    assert cut.cognee_sync_stats["failed_budget"] == 1
    assert len(staged) == 3, "the project and the first page are kept"

    rest = _source(fake)
    staged = _run(tmp_path, rest)

    assert len(staged) == 6
    assert rest.cognee_sync_stats["scanned"] == 6
    assert rest.cognee_sync_stats["skipped"] == 3, "what the cut run emitted is not emitted again"


def test_a_failed_run_keeps_the_staging_and_state_it_found(tmp_path):
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _team()
    before = _run(tmp_path, _source(fake))
    fake.delete("dpl_ready")
    fake.answers["/v7/deployments"] = {"deployments": []}

    failing = _source(fake)
    with pytest.raises(PipelineStepFailed, match="without deployments or pagination"):
        _run(tmp_path, failing)
    assert failing.cognee_sync_stats["failed"] == 1
    assert _staged(tmp_path) == before

    del fake.answers["/v7/deployments"]
    staged = _run(tmp_path, _source(fake))
    assert "deployment:dpl_ready" not in staged, "the deletion is confirmed on the next run"


def test_two_projects_keep_their_tables_and_deselecting_one_empties_only_it(tmp_path):
    empty_resource = pytest.importorskip("cognee.modules.integrations.ingestion").empty_resource

    fake = _team()
    fake.add_project("prj_api", "api")
    fake.add_deployment("dpl_api", "prj_api")
    _run(tmp_path, _source(fake, "vercel_web"))
    _run(tmp_path, _source(fake, "vercel_api", project_id="prj_api"))
    assert list(_staged(tmp_path, "vercel_api")) == ["deployment:dpl_api", "project:prj_api"]

    _pipeline(tmp_path).run(empty_resource("vercel", "vercel_api"), write_disposition="replace")

    assert _staged(tmp_path, "vercel_api") == {}
    assert len(_staged(tmp_path, "vercel_web")) == 4
    again = _source(fake, "vercel_api", project_id="prj_api")
    assert list(_run(tmp_path, again)) == ["deployment:dpl_api", "project:prj_api"]
    assert again.cognee_sync_stats["skipped"] == 0, "a re-ticked project is read from Vercel"


@pytest.mark.parametrize("include_build_logs", [False, True])
def test_no_secret_and_no_email_reaches_staging(tmp_path, include_build_logs):
    fake = _team()
    _run(tmp_path, _source(fake, include_build_logs=include_build_logs))

    staged = _staged(tmp_path)
    for content in staged.values():
        for canary in CANARIES:
            assert canary not in content
        assert "@" not in content
    raw = (tmp_path / "v.db").read_bytes()
    for canary in CANARIES:
        assert canary.encode() not in raw
    assert CANARY_EMAIL.encode() not in raw


def test_a_build_secret_reaches_staging_only_when_build_output_is_asked_for(tmp_path):
    fake = _team()
    fake.set_log("dpl_fail0", "export SECRET=short-secret-canary")

    _run(tmp_path, _source(fake))
    assert b"short-secret-canary" not in (tmp_path / "v.db").read_bytes()

    staged = _run(tmp_path, _source(fake, include_build_logs=True))
    assert "short-secret-canary" in staged["build_log:dpl_fail0"]

    staged = _run(tmp_path, _source(fake))
    assert "build_log:dpl_fail0" not in staged, "switching it off forgets it"


def test_rows_without_a_url_keep_their_columns_when_a_later_load_has_one(tmp_path):
    fake = FakeVercel()
    fake.add_project("prj_web", "web")
    _run(tmp_path, _source(fake))  # the project row alone, url None

    assert "url" in _pipeline(tmp_path).default_schema.get_table_columns(TABLE)


def test_a_dlt_run_fills_the_stats_the_host_reads(tmp_path):
    fake = _team()
    source = _source(fake)
    _run(tmp_path, source)
    assert source.cognee_sync_stats == {**new_stats(), "scanned": 4}

    fake.delete("dpl_ready")
    _run(tmp_path, source)
    assert source.cognee_sync_stats == {**new_stats(), "scanned": 3, "skipped": 3, "deleted": 1}


def test_a_route_named_in_a_message_carries_no_id():
    assert "{deployment}" in DEPLOYMENT_ROUTE


# --- review round: odd values, state binding, logs, head, wiring ------------------------------


@pytest.mark.parametrize(
    ("label", "change"),
    [
        ("created 1e20", lambda f: f.update("dpl_ready", created=1e20)),
        ("created NaN", lambda f: f.update("dpl_ready", created=float("nan"))),
        ("created infinity", lambda f: f.update("dpl_ready", created=float("inf"))),
        ("created a bool", lambda f: f.update("dpl_ready", created=True)),
        ("inspectorUrl a number", lambda f: f.update("dpl_ready", inspectorUrl=5)),
        ("errorCode a list", lambda f: f.update("dpl_ready", errorCode=["x"])),
        ("meta a string", lambda f: f.update("dpl_ready", meta="sha")),
        ("commit sha a number", lambda f: f.update("dpl_ready", meta={"githubCommitSha": 1})),
        ("target a number", lambda f: f.update("dpl_ready", target=1)),
        ("project createdAt 1e20", lambda f: f.projects["prj_web"].update(createdAt=1e20)),
        ("project link a list", lambda f: f.projects["prj_web"].update(link=["x"])),
        ("project framework a dict", lambda f: f.projects["prj_web"].update(framework={})),
    ],
)
def test_an_odd_value_in_a_200_ends_in_a_typed_error_that_is_counted(label, change):
    """These raised untyped (TypeError, OverflowError, ValueError) with nothing counted."""
    fake = _team()
    change(fake)
    stats = new_stats()

    with pytest.raises(VercelResponseError, match="Vercel"):
        list(
            _iter_rows(
                fake, {}, stats, project_id="prj_web", team_id="team_acme", account_id="team_acme"
            )
        )
    assert stats["failed"] == 1 and stats["failed_response"] == 1, label


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 1e20, -1, 1.5, "1", True])
def test_a_next_that_is_no_time_ends_in_a_typed_error(value):
    fake = _team(failed=3, page_size=2)
    listing = fake._listing

    def odd_next(params):
        body = listing(params)
        body["pagination"]["next"] = value
        return body

    fake._listing = odd_next

    with pytest.raises(VercelResponseError, match="next that is no time"):
        _walk(fake)


def test_is_ms_takes_only_finite_whole_milliseconds_datetime_can_render():
    assert vercel_module._is_ms(NOW_MS) and vercel_module._is_ms(float(NOW_MS))
    for bad in (float("nan"), float("inf"), -float("inf"), 1e20, -1, 0.5, True, "1", None):
        assert not vercel_module._is_ms(bad), bad


def test_an_error_from_the_hosts_check_is_not_counted_as_a_failure():
    """No catch-all: a revoked connection is the host's, not a failed response."""
    fake = _team()
    stats = new_stats()

    class Revoked(Exception):
        pass

    def revoked():
        raise Revoked

    with pytest.raises(Revoked):
        list(
            _iter_rows(
                fake,
                {},
                stats,
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
                check_active=revoked,
            )
        )
    assert stats == new_stats()


def _record_logs(monkeypatch) -> list[str]:
    lines: list[str] = []

    def record(message, *args, **_kwargs):
        lines.append(message % args if args else message)

    monkeypatch.setattr(vercel_module, "logger", SimpleNamespace(warning=record, info=record))
    return lines


def test_a_project_of_another_account_says_what_to_check_and_not_that_it_resumes(monkeypatch):
    lines = _record_logs(monkeypatch)
    fake = _team()
    fake.projects["prj_web"]["accountId"] = "team_other"

    _, stats, _ = _walk(fake)

    assert stats["failed_auth"] == 1
    assert any("team_id and account_id" in line for line in lines), lines
    assert not any("resumes" in line for line in lines), lines
    lines.clear()
    fake.projects["prj_web"]["accountId"] = "team_acme"
    fake.fail("/v7/deployments", 401)  # after the project row: a cut that resumes
    _walk(fake)
    assert any("resumes on the next run" in line for line in lines), "a revoked token resumes"


def test_no_log_line_of_a_walk_names_the_token_the_team_or_an_id(monkeypatch):
    from fake_vercel import serve_over_http

    from cognee_connector_vercel.source import VercelClient

    lines = _record_logs(monkeypatch)
    token = "SECRET-vercel-token-123"
    for trouble in ("/v7/deployments", "/v9/projects"):
        for status in (401, 404, 429, 503):
            fake = _team()
            fake.fail(trouble, status)
            serve_over_http(monkeypatch, fake, token)
            client = VercelClient(token, sleep=lambda _: None)  # the real client, faked transport
            rows = _iter_rows(
                client,
                {},
                new_stats(),
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
            )
            # A fresh walk emits the project row first: a 401 on the listing is a cut there.
            raises = (trouble, status) in (("/v9/projects", 401), ("/v7/deployments", 404))
            if raises:
                with pytest.raises(VercelAPIError, match=f"HTTP {status}"):
                    list(rows)
            else:
                list(rows)
    fake = _team()
    fake.projects["prj_web"]["accountId"] = "team_other"
    _walk(fake)

    assert lines
    for line in lines:
        for secret in (token, "team_acme", "team_other", "prj_web", "dpl_"):
            assert secret not in line, line


def test_state_written_for_another_project_is_refused_before_any_request(tmp_path):
    """Reusing a resource_name for another project checked the first project's deployments
    against the second on every pass, forever."""
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _team()
    fake.add_project("prj_api", "api")
    fake.add_deployment("dpl_api", "prj_api")
    _run(tmp_path, _source(fake, "vercel_shared"))
    fake.calls.clear()

    other = _source(fake, "vercel_shared", project_id="prj_api")
    with pytest.raises(PipelineStepFailed, match="another project or team"):
        _run(tmp_path, other)

    assert fake.calls == [], "refused before any request"
    assert other.cognee_sync_stats["failed"] == 1
    assert other.cognee_sync_stats["failed_state"] == 1
    assert len(_staged(tmp_path, "vercel_shared")) == 4, "nothing written"


@pytest.mark.parametrize(
    "state",
    [
        {"v": 1, "project_id": "prj_web", "team_id": "team_other"},
        {"v": 1, "project_id": "prj_api", "team_id": "team_acme"},
        {"v": 2},
    ],
)
def test_a_state_of_another_project_team_or_version_is_a_state_failure(state):
    fake = _team()
    stats = new_stats()

    with pytest.raises(vercel_module.VercelStateError, match="state"):
        list(
            _iter_rows(
                fake,
                dict(state),
                stats,
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
            )
        )
    assert fake.calls == []
    assert stats["failed"] == 1 and stats["failed_state"] == 1


def test_a_state_without_a_bound_project_is_bound_on_its_first_run():
    fake = _team()
    _, _, state = _walk(fake, {"v": 1})
    assert (state["project_id"], state["team_id"]) == ("prj_web", "team_acme")


@pytest.mark.parametrize("status", [302, 400, 404])
def test_an_error_answer_on_the_listing_is_failed_error_and_a_bad_body_failed_response(status):
    fake = _team()
    fake.fail("/v7/deployments", status)
    stats = new_stats()

    with pytest.raises(VercelAPIError, match=f"HTTP {status}"):
        list(
            _iter_rows(
                fake, {}, stats, project_id="prj_web", team_id="team_acme", account_id="team_acme"
            )
        )
    assert stats["failed_error"] == 1 and stats["failed_response"] == 0
    assert stats["failed"] == 1


def test_check_active_reaches_every_request_and_every_row_through_the_factory():
    fake = _team()
    calls = []
    source = vercel_source(
        resource_name=TABLE,
        project_id="prj_web",
        team_id="team_acme",
        service=fake,
        check_active=lambda: calls.append(1),
    )

    rows = list(source)

    # The client asks once per request; guarded_rows asks before each step and each row.
    assert len(calls) == len(fake.calls) + 2 * (len(rows) + 1)


def test_a_head_of_several_pages_is_read_to_the_top():
    fake = _team(failed=0, page_size=2)
    for n in range(6):
        fake.add_deployment(f"dpl_old{n}", minutes_ago=100 + n)
    _, _, state = _walk(fake, max_requests=3)
    assert "pass" in state
    for n in range(5):
        fake.add_deployment(f"dpl_new{n}", created=NOW_MS + n, createdAt=NOW_MS + n)

    rows, _, state = _walk(fake, state)

    assert {f"deployment:dpl_new{n}" for n in range(5)} <= set(_ids(rows))


def test_a_head_read_to_its_end_is_not_read_again_on_the_next_run():
    """``top`` moves to the newest deployment the head saw: the same new block was read again
    on every continued run."""
    fake = _many(1000)
    _, _, state = _walk(fake, max_requests=3)
    for n in range(150):
        fake.add_deployment(f"dpl_n{n:03d}", created=NOW_MS + 1 + n, createdAt=NOW_MS + 1 + n)

    def head_pages(state):
        saved = state["pass"]["until"]
        fake.calls.clear()
        _, _, state = _walk(fake, state, max_requests=6)
        untils = [params.get("until") for path, params in fake.calls if path == "/v7/deployments"]
        return untils.index(saved), state

    first, state = head_pages(state)
    second, state = head_pages(state)

    assert first == 2, "the new block is two pages"
    assert second == 1, "then only the first page"


def test_the_fingerprint_covers_the_structure():
    """A row that only moved is emitted again, so staging holds its new position."""
    fake = _team()
    row = deployment_row(fake.deployments["dpl_ready"], "prj_web")
    moved = {**row, STRUCTURE_COLUMN: {"ancestors": []}}

    assert vercel_module._fingerprint(moved) != vercel_module._fingerprint(row)


def test_a_personal_account_runs_through_the_factory():
    fake = FakeVercel(team_id=None)
    fake.add_project("prj_web", "web")
    fake.add_deployment("dpl_ready")
    source = vercel_source(
        resource_name=TABLE, project_id="prj_web", account_id="user_ada", service=fake
    )

    assert _ids(list(source)) == ["project:prj_web", "deployment:dpl_ready"]
    assert source.cognee_sync_stats["failed"] == 0


def test_an_account_id_given_with_a_team_is_the_one_checked():
    fake = FakeVercel(team_id="team_acme")
    fake.add_project("prj_web", "web", accountId="user_x")
    fake.add_deployment("dpl_ready")
    source = vercel_source(
        resource_name=TABLE,
        project_id="prj_web",
        team_id="team_acme",
        account_id="user_x",
        service=fake,
    )

    assert _ids(list(source)) == ["project:prj_web", "deployment:dpl_ready"]
    assert source.cognee_sync_stats["failed_auth"] == 0


def test_a_personal_account_without_account_id_is_refused():
    with pytest.raises(ValueError, match="account_id \\(the user id\\) for a personal account"):
        vercel_source(resource_name=TABLE, project_id="prj_web", service=FakeVercel(team_id=None))


_LONE_SURROGATE = json.loads('"\\ud800"')


@pytest.mark.parametrize(
    ("field", "trouble"),
    [
        ("name", "events 500"),
        ("name", "empty output"),
        ("name", "events ok"),
        ("inspectorUrl", "events 500"),
        ("inspectorUrl", "empty output"),
    ],
)
def test_a_lone_surrogate_never_reaches_the_resource_state(tmp_path, field, trouble):
    """The state kept the raw name and url of a failed deployment; dlt cannot store a lone
    surrogate, so the run failed on every sync (with failed 0 and no reason)."""
    fake = FakeVercel()
    fake.add_project("prj_web", "web")
    value = f"web{_LONE_SURROGATE}x" if field == "name" else f"https://v/{_LONE_SURROGATE}"
    fake.add_deployment("dpl_s", state="ERROR", **{field: value})
    fake.set_log("dpl_s", "boom")
    if trouble == "events 500":
        fake.fail("/v3/deployments/dpl_s/events", 500)
    elif trouble == "empty output":
        fake.set_log("dpl_s")

    for _ in range(2):
        source = _source(fake, include_build_logs=True)
        _run(tmp_path, source)
        stats = source.cognee_sync_stats
        reasons = sum(value for key, value in stats.items() if key.startswith("failed_"))
        assert stats["failed"] == reasons
        assert stats["failed"] == 0
        assert stats["build_log_errors"] == (1 if trouble == "events 500" else 0)
    state = _pipeline(tmp_path).state["sources"]
    assert _LONE_SURROGATE not in json.dumps(state, ensure_ascii=False, default=str)


def test_a_state_naming_a_project_but_no_team_is_a_counted_state_failure():
    fake = _team()
    stats = new_stats()

    with pytest.raises(vercel_module.VercelStateError, match="another project or team"):
        list(
            _iter_rows(
                fake,
                {"v": 1, "project_id": "prj_web"},
                stats,
                project_id="prj_web",
                team_id="team_acme",
                account_id="team_acme",
            )
        )
    assert fake.calls == []
    assert stats["failed"] == 1 and stats["failed_state"] == 1


def test_a_state_version_of_another_type_is_named_with_its_type():
    with pytest.raises(vercel_module.VercelStateError, match="state version '1'"):
        _walk(_team(), {"v": "1"})


# --- the error matrix: every call site x every outcome ----------------------------------------
#
# The single place where the error behaviour is pinned; vercel.md's error table is this matrix.
# Outcomes are answered by the fake's ``service=`` surface, so "transport" is the no-answer
# error the client raises after its retries (the retries themselves: test_vercel_client.py).

_SITES = ("project", "listing_fresh", "listing_continued", "head", "check", "build_output")
_OUTCOMES = (
    "transport",
    "302",
    "400",
    "401",
    "403",
    "404",
    "409",
    "410",
    "429",
    "500",
    "502",
    "503",
    "504",
    "not_json",
    "wrong_shape",
    "odd_value",
)
_RAISE, _CUT, _COUNT = "raise", "cut", "count"


def _expected(site: str, outcome: str, after_a_row: bool):
    """(kind, detail) the matrix pins for one cell."""
    page = site in ("project", "listing_fresh", "listing_continued", "head")
    item_key = {"check": "check_errors", "build_output": "build_log_errors"}.get(site)
    if outcome in ("401", "403"):
        return (_CUT, "failed_auth") if after_a_row else (_RAISE, vercel_module.VercelAuthError)
    if outcome == "429":
        return _CUT, "failed_rate_limit"
    if outcome in ("transport", "500", "502", "503", "504"):
        return (_CUT, "failed_unavailable") if page else (_COUNT, item_key)
    if outcome in ("302", "400", "409"):
        return (_RAISE, VercelAPIError) if page else (_COUNT, item_key)
    if outcome in ("404", "410"):
        return {
            "project": ("unresolved", None),
            "check": ("tombstone", "deployment:dpl_preview"),
            "build_output": ("no_output", None),
        }.get(site, (_RAISE, vercel_module.VercelNotFoundError))
    return (_RAISE, VercelResponseError) if page else (_COUNT, item_key)


def _matrix_cells():
    for site in _SITES:
        for outcome in _OUTCOMES:
            variants = ("before_any_row", "after_a_row")
            for variant in variants:
                unreachable = site == "project" and variant == "after_a_row"
                cell_id = f"{site}-{outcome}-{variant}"
                if unreachable:
                    cell_id += "-UNREACHABLE_project_read_is_the_first_call"
                yield pytest.param(site, outcome, variant, unreachable, id=cell_id)


def _matrix_fake() -> FakeVercel:
    fake = _team(failed=1, page_size=2)  # dpl_fail0, dpl_preview, dpl_ready: two pages
    return fake


def _matrix_state(site: str, fake: FakeVercel) -> dict:
    """The state the failing run starts from, built by clean runs of the same fake."""
    kwargs = {"include_build_logs": True}
    if site == "build_output":
        # A build output refused in the last pass is asked again in this one.
        fake.fail("/v3/deployments/dpl_fail0/events", 500)
        _, _, state = _walk(fake, **kwargs)
        fake.heal()
        return state
    _, _, state = _walk(fake, **kwargs)
    if site in ("listing_continued", "head"):
        _, _, state = _walk(fake, state, max_requests=2, **kwargs)  # a pass cut after page 1
        assert state["pass"]["until"] is not None
    if site == "check":
        fake.unlisted.add("dpl_preview")
    return state


def _matrix_body(site: str, outcome: str, original):
    if outcome == "not_json":
        return ValueError("not json")
    if site == "project":
        if outcome == "wrong_shape":
            return {"id": "prj_web"}
        return {**original, "createdAt": 1e20}
    if site in ("listing_fresh", "listing_continued", "head"):
        if outcome == "wrong_shape":
            return {"deployments": []}
        items = [dict(item) for item in original["deployments"]]
        items[0]["created"] = float("nan")
        return {**original, "deployments": items}
    if site == "check":
        if outcome == "wrong_shape":
            return {"uid": "dpl_preview", "readyState": "READY"}
        return {"id": "dpl_preview", "readyState": 5}
    if outcome == "wrong_shape":
        return {"events": []}
    return [{"type": "stdout", "text": 5}]


def _matrix_inject(fake: FakeVercel, site: str, outcome: str, state: dict) -> None:
    saved_until = (state.get("pass") or {}).get("until")

    def hit(path: str, params: dict) -> bool:
        if site == "project":
            return path.startswith("/v9/projects/")
        if site == "listing_fresh" or site == "head":
            return path == "/v7/deployments" and "until" not in params
        if site == "listing_continued":
            return path == "/v7/deployments" and params.get("until") == saved_until
        if site == "check":
            return path.startswith("/v13/")
        return path.startswith("/v3/")

    answer = fake.answer

    def failing(path, params):
        if not hit(path, params):
            return answer(path, params)
        if outcome == "transport":
            fake.calls.append((path, dict(params)))
            return None, None
        if outcome in ("not_json", "wrong_shape", "odd_value"):
            return 200, _matrix_body(site, outcome, answer(path, params)[1])
        fake.calls.append((path, dict(params)))
        return int(outcome), {"error": {"code": "x"}}

    fake.answer = failing


@pytest.mark.parametrize(("site", "outcome", "variant", "unreachable"), list(_matrix_cells()))
def test_error_matrix(site, outcome, variant, unreachable):
    fake = _matrix_fake()
    if unreachable:
        # Nothing can be emitted before the project read: it is the first request of every run.
        state = _matrix_state(site, fake)
        fake.calls.clear()
        fake.rename_project("prj_web", "website")
        _walk(fake, state, include_build_logs=True)
        assert fake.calls[0][0] == "/v9/projects/prj_web"
        return
    state = _matrix_state(site, fake)
    after_a_row = variant == "after_a_row"
    if after_a_row:
        fake.rename_project("prj_web", "website")  # the project row is emitted first
    _matrix_inject(fake, site, outcome, state)
    stats = new_stats()
    rows: list[dict] = []
    raised = None
    walk = _iter_rows(
        fake,
        state,
        stats,
        project_id="prj_web",
        team_id="team_acme",
        account_id="team_acme",
        include_build_logs=True,
    )
    try:
        rows.extend(walk)  # keeps the rows that came out before a raise
    except VercelSourceError as error:
        raised = error

    kind, detail = _expected(site, outcome, after_a_row)
    reasons = {key: value for key, value in stats.items() if key.startswith("failed_")}
    assert all(type(value) is int and value >= 0 for value in stats.values()), stats
    assert stats["failed"] == sum(reasons.values()), stats
    tombstones = _tombstones(rows)
    if kind == "tombstone":
        assert raised is None and tombstones == [detail]
        assert stats["failed"] == 0
        return
    assert tombstones == [], "nothing is forgotten on a failure"
    if after_a_row:
        assert "project:prj_web" in _ids(rows), "rows emitted before the failure are kept"
    if kind == _RAISE:
        assert type(raised) is detail, (raised, detail)
        expected_reason = vercel_module._raised_reason(raised)
        assert stats["failed"] == 1 and stats[expected_reason] == 1, stats
    elif kind == _CUT:
        assert raised is None, raised
        assert stats["failed"] == 1 and stats[detail] == 1, stats
    elif kind == _COUNT:
        assert raised is None, raised
        assert stats["failed"] == 0 and stats[detail] == 1, "an item error, not a failed run"
        assert "pass" not in state, "the run went on and finished its pass"
    elif kind == "unresolved":
        assert raised is None and stats["unresolved"] == 1 and stats["failed"] == 0
        assert rows == []
    else:  # no_output: a finished build without output, recorded and not a failure
        assert raised is None and stats["failed"] == 0
        assert state["logs"]["dpl_fail0"]["status"] == "none"


def test_a_zero_budget_defers_the_run_without_a_request_or_a_row(monkeypatch):
    """The Cloud pod gives dispatch units past 40 a budget of 0 and must settle every one; a
    ValueError at construction failed the whole dispatch."""

    def no_call(*_args, **_kwargs):
        raise AssertionError("a deferred run makes no request")

    monkeypatch.setattr("requests.request", no_call)
    source = vercel_source(
        resource_name=TABLE,
        project_id="prj_web",
        team_id="team_acme",
        access_token="SECRET-token",
        max_requests=0,
    )

    assert list(source) == []
    assert source.cognee_sync_stats == {**new_stats(), "failed": 1, "failed_budget": 1}


def test_a_deferred_run_leaves_a_pass_in_progress_untouched(tmp_path):
    fake = _many(251)
    _run(tmp_path, _source(fake, max_requests=2))  # a pass cut after its first page
    before = json.dumps(_pipeline(tmp_path).state["sources"], sort_keys=True, default=str)
    staged_before = _staged(tmp_path)
    fake.calls.clear()

    deferred = _source(fake, max_requests=0)
    _run(tmp_path, deferred)

    assert fake.calls == []
    assert json.dumps(_pipeline(tmp_path).state["sources"], sort_keys=True, default=str) == before
    assert _staged(tmp_path) == staged_before
    assert deferred.cognee_sync_stats == {**new_stats(), "failed": 1, "failed_budget": 1}
    finished = _source(fake)
    staged = _run(tmp_path, finished)
    assert len(staged) == 252 and finished.cognee_sync_stats["failed"] == 0
