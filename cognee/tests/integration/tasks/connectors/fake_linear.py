"""An in-memory Linear team behind the source's GraphQL transport interface."""


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
