"""Create a Linear ticket for every approval on a merged pull request.

Once a day this finds the pull requests of this repository merged in the last
72 hours (LOOKBACK_HOURS overrides it), and for every reviewer who approved one
of them
creates a ticket assigned to that reviewer in the review team, already in a
completed state, so time spent on reviews shows up in Linear.

Reviewers are matched to Linear users by GitHub account: a Linear user who has
connected GitHub carries the numeric GitHub user id (``gitHubUserId``), which is
the id GitHub reports for the reviewer. Approvals by bots, and by people with no
Linear user (external contributors, or members who never connected GitHub in
Linear), get no ticket; they are listed in the run summary.

Every ticket's id is derived from (repository, pull request, reviewer), so a
pull request is ticketed once per reviewer however often it is seen. That is
what lets the window be a plain 72 hours for a daily run: consecutive windows
overlap by two days, which absorbs GitHub's varying start time for scheduled
runs and up to two failed or skipped days, and every pull request seen again is
skipped. A longer outage needs one manual run with a larger LOOKBACK_HOURS.

Input (env):
  GITHUB_TOKEN, GITHUB_REPOSITORY            the repository and a token to read it.
  GITHUB_RUN_ID                              this run, linked from every ticket.
  LINEAR_API_KEY                             Linear personal API key.
  LINEAR_REVIEW_TEAM_KEY                     team the tickets go to, e.g. SDK.
  LINEAR_REVIEW_LABEL                        label put on every ticket. It must exist
                                             in the workspace, the team or a team above
                                             it (the run fails otherwise).
  LOOKBACK_HOURS                             optional window in hours instead of 72
                                             (manual runs, backfills); at most 336.
  PR_NUMBER                                  optional: handle this one merged pull
                                             request only, ignoring the window (tests).
  DRY_RUN                                    "true": read everything, write nothing,
                                             print the tickets that would be created.
  GITHUB_STEP_SUMMARY                        where the run summary is appended.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

GITHUB_API = "https://api.github.com"
LINEAR_API = "https://api.linear.app/graphql"
HTTP_TIMEOUT = 30  # seconds
DEFAULT_LOOKBACK = timedelta(hours=72)
# A window never reaches further back than this: GitHub search caps results at
# 1000 per query.
MAX_LOOKBACK = timedelta(days=14)
# Namespace for the deterministic ticket ids. NEVER CHANGE IT: every ticket id
# is derived from it, so a new value gives every past review a new id and the
# next run over an already-ticketed period creates a duplicate for each one.
TICKET_NAMESPACE = uuid.UUID("6f1c2b7e-2d4a-4e0b-9a51-8c3f6f0d5a11")
MAX_TITLE = 200


@dataclass(frozen=True)
class Approval:
    pr_number: int
    pr_title: str
    pr_url: str
    pr_author: str
    base_ref: str
    merged_at: str
    additions: int
    deletions: int
    changed_files: int
    reviewer_login: str
    reviewer_github_id: int
    approved_at: str


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    pr_number = os.environ.get("PR_NUMBER")
    if pr_number:
        approvals = pull_approvals(repo, int(pr_number))
        scope = f"Pull request #{pr_number} of `{repo}` only."
    else:
        until = datetime.now(timezone.utc)
        since = window_start(until)
        approvals = find_approvals(repo, since, until)
        scope = (
            f"Pull requests of `{repo}` merged "
            f"{since:%Y-%m-%d %H:%M} .. {until:%Y-%m-%d %H:%M} UTC."
        )
    print(f"{len(approvals)} approval(s). {scope}")

    linear = Linear(os.environ["LINEAR_API_KEY"])
    team = linear.team(os.environ["LINEAR_REVIEW_TEAM_KEY"])
    label_name = os.environ.get("LINEAR_REVIEW_LABEL", "")
    if not label_name:
        raise ValueError("LINEAR_REVIEW_LABEL is empty")
    label_id = linear.label_id(team, label_name)
    users = linear.users_by_github_id()

    created, existing, unmatched = [], [], []
    for approval in approvals:
        user = users.get(str(approval.reviewer_github_id))
        if user is None:
            unmatched.append(approval)
            continue
        ticket_id = ticket_uuid(repo, approval)
        if linear.issue_exists(ticket_id):
            existing.append(approval)
            continue
        if not dry_run:
            linear.create_ticket(ticket_id, team, user["id"], label_id, repo, approval, run_url())
        created.append((approval, user))

    write_summary(scope, dry_run, created, existing, unmatched)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #


def window_start(until: datetime) -> datetime:
    """LOOKBACK_HOURS back when set, else 72 h back; never more than MAX_LOOKBACK."""
    lookback = os.environ.get("LOOKBACK_HOURS")
    if lookback:
        hours = float(lookback)
        if hours <= 0:
            raise ValueError(f"LOOKBACK_HOURS must be positive, got {lookback!r}")
        since = until - timedelta(hours=hours)
    else:
        since = until - DEFAULT_LOOKBACK
    return max(since, until - MAX_LOOKBACK)


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


def find_approvals(repo: str, since: datetime, until: datetime) -> list[Approval]:
    """Every (merged pull request, approving human reviewer) pair in the window, oldest first."""
    approvals = []
    for number in merged_pull_numbers(repo, since, until):
        approvals.extend(pull_approvals(repo, number))
    return sorted(approvals, key=lambda a: (a.merged_at, a.pr_number, a.reviewer_login))


def pull_approvals(repo: str, number: int) -> list[Approval]:
    """The approvals of one merged pull request; an unmerged one is an error."""
    pr = github_get(f"/repos/{repo}/pulls/{number}")
    if not pr.get("merged_at"):
        raise ValueError(f"Pull request #{number} of {repo} is not merged")
    return approvals_of(pr, github_paged(f"/repos/{repo}/pulls/{number}/reviews"))


def merged_pull_numbers(repo: str, since: datetime, until: datetime) -> list[int]:
    query = (
        f"repo:{repo} is:pr is:merged merged:{since:%Y-%m-%dT%H:%M:%SZ}..{until:%Y-%m-%dT%H:%M:%SZ}"
    )
    numbers, page = [], 1
    while True:
        result = github_get(
            "/search/issues?" + urllib.parse.urlencode({"q": query, "per_page": 100, "page": page})
        )
        numbers.extend(item["number"] for item in result["items"])
        if len(result["items"]) < 100:
            return numbers
        page += 1


def approvals_of(pr: dict, reviews: list[dict]) -> list[Approval]:
    """One Approval per human reviewer whose review of ``pr`` is APPROVED, at their first approval.

    A dismissed approval has state DISMISSED and does not count. Bots and the
    pull request's author are left out.
    """
    first_approval: dict[int, dict] = {}
    for review in reviews:
        user = review.get("user") or {}
        if review.get("state") != "APPROVED" or user.get("type") == "Bot":
            continue
        if user.get("id") == pr["user"]["id"]:
            continue
        current = first_approval.get(user["id"])
        if current is None or review["submitted_at"] < current["submitted_at"]:
            first_approval[user["id"]] = review
    return [
        Approval(
            pr_number=pr["number"],
            pr_title=pr["title"],
            pr_url=pr["html_url"],
            pr_author=pr["user"]["login"],
            base_ref=pr["base"]["ref"],
            merged_at=pr["merged_at"],
            additions=pr.get("additions", 0),
            deletions=pr.get("deletions", 0),
            changed_files=pr.get("changed_files", 0),
            reviewer_login=review["user"]["login"],
            reviewer_github_id=review["user"]["id"],
            approved_at=review["submitted_at"],
        )
        for review in first_approval.values()
    ]


def github_get(path: str):
    request = urllib.request.Request(
        GITHUB_API + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def github_paged(path: str) -> list:
    items, page = [], 1
    separator = "&" if "?" in path else "?"
    while True:
        batch = github_get(f"{path}{separator}per_page=100&page={page}")
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #


class Linear:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("LINEAR_API_KEY is empty")
        self.api_key = api_key

    def query(self, query: str, variables: dict | None = None) -> dict:
        request = urllib.request.Request(
            LINEAR_API,
            data=json.dumps({"query": query, "variables": variables or {}}).encode(),
            headers={"Content-Type": "application/json", "Authorization": self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Linear API HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"
            ) from error
        if payload.get("errors"):
            raise RuntimeError(f"Linear API error: {json.dumps(payload['errors'])[:500]}")
        return payload["data"]

    def team(self, key: str) -> dict:
        """The team with this key, its completed workflow state and its parent team's id.

        The parent id comes with the team itself, so it needs no access to the
        parent team (a key restricted to the review team cannot read its parent).
        """
        if not key:
            raise ValueError("LINEAR_REVIEW_TEAM_KEY is empty")
        data = self.query(
            """query($key: String!) {
                 teams(filter: {key: {eq: $key}}) {
                   nodes { id key name parent { id } states { nodes { id name type position } } }
                 }
               }""",
            {"key": key},
        )
        teams = data["teams"]["nodes"]
        if not teams:
            raise RuntimeError(f"No Linear team with key {key!r}")
        team = teams[0]
        completed = sorted(
            (s for s in team["states"]["nodes"] if s["type"] == "completed"),
            key=lambda s: s["position"],
        )
        if not completed:
            raise RuntimeError(f"Linear team {key!r} has no completed workflow state")
        return {
            "id": team["id"],
            "key": team["key"],
            "state_id": completed[0]["id"],
            "parent_id": team["parent"]["id"] if team["parent"] else None,
        }

    def label_id(self, team: dict, name: str) -> str:
        """The id of the label with this name usable in the team.

        Usable: a workspace label, or a label of the team or of a team above it
        (a sub-team inherits its parents' labels). A same-named label of an
        unrelated team is not. The team and its parent are known from ``team``;
        higher ancestors are only read when a candidate label belongs to
        neither, which needs the API key to read those teams.
        """
        data = self.query(
            """query($name: String!) {
                 issueLabels(filter: {name: {eqIgnoreCase: $name}}) {
                   nodes { id name team { id } }
                 }
               }""",
            {"name": name},
        )
        labels = data["issueLabels"]["nodes"]
        known = {team["id"], team["parent_id"]} - {None}
        for label in labels:
            if label["team"] is None or label["team"]["id"] in known:
                return label["id"]
        others = {label["team"]["id"]: label["id"] for label in labels}
        if others and team["parent_id"]:
            for ancestor_id in self.ancestors_above(team["parent_id"]):
                if ancestor_id in others:
                    return others[ancestor_id]
        raise RuntimeError(
            f"No Linear label {name!r} usable in team {team['key']}: create it in the "
            "workspace, in the team or in a team above it, and make sure the API key "
            "can read the team it is in"
        )

    def ancestors_above(self, team_id: str) -> list[str]:
        """The ids of the teams above ``team_id``, nearest first."""
        ancestors = []
        while True:
            try:
                parent = self.query(
                    """query($id: String!) { team(id: $id) { parent { id } } }""",
                    {"id": team_id},
                )["team"]["parent"]
            except RuntimeError as error:
                raise RuntimeError(
                    f"The Linear API key cannot read team {team_id}, needed to tell whether "
                    "a same-named label is inherited: give the key access to that team, "
                    "or put the label in the review team, its parent or the workspace"
                ) from error
            if parent is None:
                return ancestors
            ancestors.append(parent["id"])
            team_id = parent["id"]

    def users_by_github_id(self) -> dict[str, dict]:
        """Active, assignable Linear users who connected GitHub, by GitHub user id."""
        users, after = {}, None
        while True:
            data = self.query(
                """query($after: String) {
                     users(first: 100, after: $after) {
                       nodes { id name active gitHubUserId }
                       pageInfo { hasNextPage endCursor }
                     }
                   }""",
                {"after": after},
            )
            for user in data["users"]["nodes"]:
                if user["active"] and user["gitHubUserId"]:
                    users[str(user["gitHubUserId"])] = user
            page = data["users"]["pageInfo"]
            if not page["hasNextPage"]:
                return users
            after = page["endCursor"]

    def issue_exists(self, issue_id: str) -> bool:
        # Archived (and trashed) tickets count: their id is taken, and creating
        # it again would fail every later run.
        data = self.query(
            """query($id: ID!) {
                 issues(filter: {id: {eq: $id}}, includeArchived: true) { nodes { id } }
               }""",
            {"id": issue_id},
        )
        return bool(data["issues"]["nodes"])

    def create_ticket(
        self,
        issue_id: str,
        team: dict,
        assignee_id: str,
        label_id: str,
        repo: str,
        approval: Approval,
        source_url: str | None,
    ) -> None:
        issue = {
            "id": issue_id,
            "teamId": team["id"],
            "assigneeId": assignee_id,
            "stateId": team["state_id"],
            "title": ticket_title(repo, approval),
            "description": ticket_description(approval, source_url),
            "labelIds": [label_id],
        }
        data = self.query(
            """mutation($input: IssueCreateInput!) {
                 issueCreate(input: $input) { success issue { id identifier } }
               }""",
            {"input": issue},
        )
        if not data["issueCreate"]["success"]:
            raise RuntimeError(f"Linear refused to create the ticket for {approval.pr_url}")
        attached = self.query(
            """mutation($input: AttachmentCreateInput!) {
                 attachmentCreate(input: $input) { success }
               }""",
            {
                "input": {
                    "issueId": issue_id,
                    "url": approval.pr_url,
                    "title": f"#{approval.pr_number} {approval.pr_title}"[:MAX_TITLE],
                }
            },
        )
        if not attached["attachmentCreate"]["success"]:
            raise RuntimeError(f"Linear refused to attach {approval.pr_url}")
        print(
            f"Created {data['issueCreate']['issue']['identifier']} for "
            f"{approval.reviewer_login} on #{approval.pr_number}"
        )


# --------------------------------------------------------------------------- #
# Ticket content
# --------------------------------------------------------------------------- #


def ticket_uuid(repo: str, approval: Approval) -> str:
    """Deterministic per (repository, pull request, reviewer), shaped as UUID v4 as Linear requires."""
    digest = uuid.uuid5(
        TICKET_NAMESPACE, f"{repo}#{approval.pr_number}:{approval.reviewer_github_id}"
    )
    return str(uuid.UUID(bytes=digest.bytes, version=4))


def ticket_title(repo: str, approval: Approval) -> str:
    return f"PR review: {repo.split('/')[-1]}#{approval.pr_number} {approval.pr_title}"[:MAX_TITLE]


def ticket_description(approval: Approval, source_url: str | None) -> str:
    lines = [
        f"Approved pull request [#{approval.pr_number} {approval.pr_title}]({approval.pr_url}).",
        "",
        f"- Author: @{approval.pr_author}",
        f"- Reviewer: @{approval.reviewer_login}",
        f"- Approved: {approval.approved_at}",
        f"- Merged into `{approval.base_ref}`: {approval.merged_at}",
        f"- Size: +{approval.additions} −{approval.deletions} in {approval.changed_files} file(s)",
    ]
    if source_url:
        lines += ["", f"_Created by the [PR review tickets workflow]({source_url})._"]
    return "\n".join(lines)


def run_url() -> str | None:
    if not os.environ.get("GITHUB_RUN_ID"):
        return None
    return (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    )


def write_summary(scope, dry_run, created, existing, unmatched) -> None:
    verb = "Would create" if dry_run else "Created"
    lines = [
        f"## PR review tickets{' (dry run)' if dry_run else ''}",
        "",
        scope,
        "",
        (
            f"**{verb} {len(created)} ticket(s)**, {len(existing)} already existed, "
            f"{len(unmatched)} approval(s) with no Linear user."
        ),
        "",
    ]
    if created:
        lines += ["| PR | Reviewer | Linear user |", "|---|---|---|"]
        lines += [
            f"| [#{a.pr_number}]({a.pr_url}) {a.pr_title[:60]} | @{a.reviewer_login} | {u['name']} |"
            for a, u in created
        ]
        lines.append("")
    if unmatched:
        lines += [
            "No Linear user has connected these GitHub accounts, so their approvals got no ticket:",
            "",
        ]
        lines += [f"- @{a.reviewer_login} on [#{a.pr_number}]({a.pr_url})" for a in unmatched]
    text = "\n".join(lines) + "\n"
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(text)


if __name__ == "__main__":
    main()
