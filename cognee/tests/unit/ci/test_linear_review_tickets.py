"""Tests for .github/scripts/linear_review_tickets.py (the PR review tickets workflow).

GitHub and Linear are faked at the HTTP-call level (``github_get`` /
``github_paged`` / ``Linear.query``), so the selection, de-duplication, dry-run
and ticket content run exactly as in the workflow, with no network.
"""

import importlib.util
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / ".github" / "scripts" / "linear_review_tickets.py"
_spec = importlib.util.spec_from_file_location("linear_review_tickets", SCRIPT)
lrt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lrt)

REPO = "topoteretes/cognee"
AUTHOR = {"login": "author", "id": 1, "type": "User"}
ALICE = {"login": "alice", "id": 101, "type": "User"}
BOB = {"login": "bob", "id": 102, "type": "User"}
OUTSIDER = {"login": "outsider", "id": 999, "type": "User"}
BOT = {"login": "coderabbitai[bot]", "id": 500, "type": "Bot"}


def _pr(number, title="Fix a thing", merged_at="2026-10-20T10:00:00Z", base="dev"):
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "user": AUTHOR,
        "base": {"ref": base},
        "merged_at": merged_at,
        "additions": 10,
        "deletions": 2,
        "changed_files": 3,
    }


def _review(user, state, at):
    return {"user": user, "state": state, "submitted_at": at}


# --------------------------------------------------------------------------- #
# Approval selection
# --------------------------------------------------------------------------- #


def test_approvals_keep_each_human_approver_once_at_their_first_approval():
    reviews = [
        _review(ALICE, "COMMENTED", "2026-10-20T08:00:00Z"),
        _review(ALICE, "APPROVED", "2026-10-20T09:30:00Z"),
        _review(ALICE, "APPROVED", "2026-10-20T09:00:00Z"),
        _review(BOB, "CHANGES_REQUESTED", "2026-10-20T08:30:00Z"),
        _review(BOB, "DISMISSED", "2026-10-20T08:45:00Z"),  # a dismissed approval
        _review(BOT, "APPROVED", "2026-10-20T08:50:00Z"),
        _review(AUTHOR, "APPROVED", "2026-10-20T08:55:00Z"),
    ]

    approvals = lrt.approvals_of(_pr(7), reviews)

    assert [(a.reviewer_login, a.approved_at) for a in approvals] == [
        ("alice", "2026-10-20T09:00:00Z")
    ]
    approval = approvals[0]
    assert (approval.pr_number, approval.base_ref, approval.reviewer_github_id) == (7, "dev", 101)


def test_a_pull_request_without_approvals_yields_nothing():
    reviews = [_review(ALICE, "COMMENTED", "2026-10-20T08:00:00Z")]
    assert lrt.approvals_of(_pr(7), reviews) == []


# --------------------------------------------------------------------------- #
# Ticket ids
# --------------------------------------------------------------------------- #


def test_ticket_id_is_deterministic_per_pull_request_and_reviewer_and_uuid_v4():
    alice_on_7 = lrt.approvals_of(_pr(7), [_review(ALICE, "APPROVED", "2026-10-20T09:00:00Z")])[0]
    bob_on_7 = lrt.approvals_of(_pr(7), [_review(BOB, "APPROVED", "2026-10-20T09:00:00Z")])[0]
    alice_on_8 = lrt.approvals_of(_pr(8), [_review(ALICE, "APPROVED", "2026-10-20T09:00:00Z")])[0]

    first = lrt.ticket_uuid(REPO, alice_on_7)
    assert first == lrt.ticket_uuid(REPO, alice_on_7)
    assert uuid.UUID(first).version == 4
    assert len({first, lrt.ticket_uuid(REPO, bob_on_7), lrt.ticket_uuid(REPO, alice_on_8)}) == 3
    assert first != lrt.ticket_uuid("someone/fork", alice_on_7)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc)


def test_window_uses_lookback_hours_when_set(monkeypatch):
    monkeypatch.setenv("LOOKBACK_HOURS", "6")
    assert lrt.window_start(NOW) == NOW - timedelta(hours=6)


@pytest.mark.parametrize("value", ["0", "-3"])
def test_window_rejects_a_non_positive_lookback(monkeypatch, value):
    monkeypatch.setenv("LOOKBACK_HOURS", value)
    with pytest.raises(ValueError, match="LOOKBACK_HOURS"):
        lrt.window_start(NOW)


def test_window_defaults_to_72_hours(monkeypatch):
    monkeypatch.delenv("LOOKBACK_HOURS", raising=False)
    assert lrt.window_start(NOW) == NOW - timedelta(hours=72)


def test_window_is_capped(monkeypatch):
    monkeypatch.setenv("LOOKBACK_HOURS", "10000")
    assert lrt.window_start(NOW) == NOW - lrt.MAX_LOOKBACK


# --------------------------------------------------------------------------- #
# Linear fake
# --------------------------------------------------------------------------- #


class FakeLinear:
    """Answers the script's queries by matching their operation; records mutations."""

    # SDK sits under Engineering, which sits under Company (as in the Linear workspace).
    PARENTS = {"team-sdk": "team-eng", "team-eng": "team-company", "team-company": None}

    def __init__(self, existing_ids=(), labels=None, teams=None, readable_teams=None):
        # None: every team readable. A set: only those (a key restricted to some teams).
        self.readable_teams = readable_teams
        self.team_reads = []
        self.existing_ids = set(existing_ids)
        self.labels = (
            labels if labels is not None else [{"id": "lbl-ws", "name": "PR review", "team": None}]
        )

        self.teams = teams
        self.created = []
        self.attached = []

    def __call__(self, query, variables=None):
        variables = variables or {}
        if "teams(" in query:
            nodes = (
                self.teams
                if self.teams is not None
                else [
                    {
                        "id": "team-sdk",
                        "key": variables["key"],
                        "name": "SDK",
                        "parent": {"id": "team-eng"},
                        "states": {
                            "nodes": [
                                {
                                    "id": "s-todo",
                                    "name": "Todo",
                                    "type": "unstarted",
                                    "position": 1,
                                },
                                {
                                    "id": "s-merged",
                                    "name": "Merged",
                                    "type": "completed",
                                    "position": 6,
                                },
                                {
                                    "id": "s-done",
                                    "name": "Done",
                                    "type": "completed",
                                    "position": 5,
                                },
                            ]
                        },
                    }
                ]
            )
            return {"teams": {"nodes": nodes}}
        if "team(id:" in query:
            self.team_reads.append(variables["id"])
            if self.readable_teams is not None and variables["id"] not in self.readable_teams:
                raise RuntimeError('Linear API error: [{"message": "Entity not found: Team"}]')
            parent = self.PARENTS[variables["id"]]
            return {"team": {"parent": {"id": parent} if parent else None}}
        if "issueLabels(" in query:
            return {"issueLabels": {"nodes": self.labels}}
        if "users(" in query:
            if variables.get("after") is None:
                return {
                    "users": {
                        "nodes": [
                            {
                                "id": "u-alice",
                                "name": "Alice A",
                                "active": True,
                                "gitHubUserId": "101",
                            },
                            {
                                "id": "u-gone",
                                "name": "Gone",
                                "active": False,
                                "gitHubUserId": "103",
                            },
                            {
                                "id": "u-nogh",
                                "name": "No GitHub",
                                "active": True,
                                "gitHubUserId": None,
                            },
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                    }
                }
            return {
                "users": {
                    "nodes": [
                        {"id": "u-bob", "name": "Bob B", "active": True, "gitHubUserId": "102"}
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        if "issues(" in query:
            assert "includeArchived: true" in query
            found = variables["id"] in self.existing_ids
            return {"issues": {"nodes": [{"id": variables["id"]}] if found else []}}
        if "issueCreate" in query:
            self.created.append(variables["input"])
            return {"issueCreate": {"success": True, "issue": {"id": "x", "identifier": "SDK-1"}}}
        if "attachmentCreate" in query:
            self.attached.append(variables["input"])
            return {"attachmentCreate": {"success": True}}
        raise AssertionError(f"unexpected Linear query: {query[:80]}")


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Two merged PRs: #7 approved by alice and bob, #8 approved by an outsider and a bot."""
    prs = {7: _pr(7, "Add a feature"), 8: _pr(8, "Fix docs", merged_at="2026-10-20T11:00:00Z")}
    reviews = {
        7: [
            _review(ALICE, "APPROVED", "2026-10-20T09:00:00Z"),
            _review(BOB, "APPROVED", "2026-10-20T09:10:00Z"),
        ],
        8: [
            _review(OUTSIDER, "APPROVED", "2026-10-20T10:00:00Z"),
            _review(BOT, "APPROVED", "2026-10-20T10:05:00Z"),
        ],
    }
    searches = []

    def github_get(path):
        if path.startswith("/search/issues"):
            searches.append(path)
            return {"items": [{"number": n} for n in prs]}
        number = int(path.rsplit("/", 1)[1])
        return prs[number]

    def github_paged(path):
        return reviews[int(path.split("/pulls/")[1].split("/")[0])]

    monkeypatch.setattr(lrt, "github_get", github_get)
    monkeypatch.setattr(lrt, "github_paged", github_paged)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")
    monkeypatch.setenv("LINEAR_REVIEW_TEAM_KEY", "SDK")
    monkeypatch.setenv("LOOKBACK_HOURS", "24")
    monkeypatch.setenv("LINEAR_REVIEW_LABEL", "PR review")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("PR_NUMBER", raising=False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    return {"summary": summary, "searches": searches, "prs": prs}


def _install(monkeypatch, fake):
    monkeypatch.setattr(lrt.Linear, "query", lambda self, q, v=None: fake(q, v))


# --------------------------------------------------------------------------- #
# End to end over the fakes
# --------------------------------------------------------------------------- #


def test_creates_one_completed_ticket_per_matched_approver(monkeypatch, world):
    fake = FakeLinear()
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")

    lrt.main()

    assert [t["assigneeId"] for t in fake.created] == ["u-alice", "u-bob"]
    ticket = fake.created[0]
    assert ticket["teamId"] == "team-sdk"
    assert ticket["stateId"] == "s-done"  # the completed state with the lowest position
    assert ticket["title"] == "PR review: cognee#7 Add a feature"
    assert "https://github.com/topoteretes/cognee/pull/7" in ticket["description"]
    assert "Approved: 2026-10-20T09:00:00Z" in ticket["description"]
    assert ticket["labelIds"] == ["lbl-ws"]
    assert uuid.UUID(ticket["id"]).version == 4
    assert [a["url"] for a in fake.attached] == [
        "https://github.com/topoteretes/cognee/pull/7",
        "https://github.com/topoteretes/cognee/pull/7",
    ]
    assert all(a["issueId"] in {t["id"] for t in fake.created} for a in fake.attached)
    assert "is:pr is:merged" in json.dumps(world["searches"]) or "is%3Apr" in world["searches"][0]

    summary = world["summary"].read_text()
    assert "Created 2 ticket(s)" in summary
    assert "1 approval(s) with no Linear user" in summary
    assert "@outsider" in summary
    assert "coderabbitai" not in summary  # bots are not even reported


def test_dry_run_reads_everything_and_writes_nothing(monkeypatch, world):
    fake = FakeLinear()
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "true")

    lrt.main()

    assert fake.created == [] and fake.attached == []
    summary = world["summary"].read_text()
    assert "(dry run)" in summary and "Would create 2 ticket(s)" in summary
    assert "@alice" in summary and "Alice A" in summary


def test_rerun_creates_nothing_for_tickets_that_already_exist(monkeypatch, world):
    first = FakeLinear()
    _install(monkeypatch, first)
    monkeypatch.setenv("DRY_RUN", "false")
    lrt.main()

    second = FakeLinear(existing_ids={t["id"] for t in first.created})
    _install(monkeypatch, second)
    lrt.main()

    assert second.created == []
    assert "2 already existed" in world["summary"].read_text()


def test_an_existing_team_or_workspace_label_is_used_not_another_teams(monkeypatch, world):
    fake = FakeLinear(
        labels=[
            {"id": "lbl-other-team", "name": "PR review", "team": {"id": "team-x"}},
            {"id": "lbl-ws", "name": "PR review", "team": None},
        ]
    )
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")

    lrt.main()

    assert {tuple(t["labelIds"]) for t in fake.created} == {("lbl-ws",)}


def test_a_label_inherited_from_a_parent_team_is_used(monkeypatch, world):
    fake = FakeLinear(
        labels=[
            {"id": "lbl-unrelated", "name": "PR review", "team": {"id": "team-x"}},
            {"id": "lbl-company", "name": "PR Review", "team": {"id": "team-company"}},
        ]
    )
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")

    lrt.main()

    assert {tuple(t["labelIds"]) for t in fake.created} == {("lbl-company",)}


def test_a_parent_team_label_needs_no_access_to_the_parent_team(monkeypatch, world):
    """The real setup: the label is in Engineering, the key can read SDK only."""
    fake = FakeLinear(
        labels=[{"id": "lbl-eng", "name": "PR review", "team": {"id": "team-eng"}}],
        readable_teams={"team-sdk"},
    )
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")

    lrt.main()

    assert {tuple(t["labelIds"]) for t in fake.created} == {("lbl-eng",)}
    assert fake.team_reads == []  # the parent id came with SDK's own record


def test_an_unreadable_ancestor_fails_with_the_fix(monkeypatch, world):
    """A same-named label two levels up, and a key that cannot read the parent."""
    fake = FakeLinear(
        labels=[{"id": "lbl-company", "name": "PR review", "team": {"id": "team-company"}}],
        readable_teams={"team-sdk"},
    )
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "true")

    with pytest.raises(RuntimeError, match="cannot read team team-eng"):
        lrt.main()
    assert fake.created == []


@pytest.mark.parametrize("dry_run", ["false", "true"])
def test_a_missing_label_fails_the_run_before_any_ticket(monkeypatch, world, dry_run):
    """Only another team's label exists: the run stops, creating neither a label nor a ticket."""
    fake = FakeLinear(labels=[{"id": "lbl-other", "name": "PR review", "team": {"id": "team-x"}}])
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", dry_run)

    with pytest.raises(RuntimeError, match="No Linear label 'PR review' usable in team SDK"):
        lrt.main()
    assert fake.created == [] and fake.attached == []


def test_an_empty_label_name_fails_fast(monkeypatch, world):
    _install(monkeypatch, FakeLinear())
    monkeypatch.setenv("LINEAR_REVIEW_LABEL", "")

    with pytest.raises(ValueError, match="LINEAR_REVIEW_LABEL"):
        lrt.main()


def test_an_unknown_team_fails_the_run(monkeypatch, world):
    _install(monkeypatch, FakeLinear(teams=[]))
    monkeypatch.setenv("DRY_RUN", "true")

    with pytest.raises(RuntimeError, match="No Linear team with key 'SDK'"):
        lrt.main()


def test_pr_number_handles_that_pull_request_only(monkeypatch, world):
    fake = FakeLinear()
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("PR_NUMBER", "8")

    lrt.main()

    assert world["searches"] == []  # no window search at all
    assert fake.created == []  # #8 was approved by an outsider and a bot only
    summary = world["summary"].read_text()
    assert "Pull request #8 of `topoteretes/cognee` only." in summary
    assert "@outsider" in summary and "#7" not in summary


def test_pr_number_creates_one_ticket_per_approver_of_that_pull_request(monkeypatch, world):
    fake = FakeLinear()
    _install(monkeypatch, fake)
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("PR_NUMBER", "7")

    lrt.main()

    assert [t["title"] for t in fake.created] == ["PR review: cognee#7 Add a feature"] * 2
    assert world["searches"] == []


def test_pr_number_of_an_unmerged_pull_request_fails(monkeypatch, world):
    _install(monkeypatch, FakeLinear())
    world["prs"][7]["merged_at"] = None
    monkeypatch.setenv("PR_NUMBER", "7")

    with pytest.raises(ValueError, match="#7 of topoteretes/cognee is not merged"):
        lrt.main()


def test_users_are_matched_by_github_id_across_pages_active_only(monkeypatch):
    _install(monkeypatch, FakeLinear())
    users = lrt.Linear("lin_api_test").users_by_github_id()
    assert sorted(users) == ["101", "102"]


def test_an_empty_api_key_fails_fast():
    with pytest.raises(ValueError, match="LINEAR_API_KEY"):
        lrt.Linear("")
