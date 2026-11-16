"""In-memory stand-in for the Mixpanel API, shaped like its real responses.

An endpoint the connector does not use is not answered: a call to it raises
``AssertionError`` and fails the test that made it.
"""

from cognee_connector_mixpanel.source import MixpanelAPIError

PROJECT_ID = "12345"


class FakeMixpanel:
    def __init__(self, project_id: str = PROJECT_ID):
        self.project_id = project_id
        # A second project the credential could see, to test discovery refusing a guess.
        self.extra_projects: list[str] = []
        self.annotations: dict[str, dict] = {}
        self.cohorts: dict[str, dict] = {}
        self.funnels: dict[str, dict] = {}
        self.schemas: dict[str, dict] = {}
        # (from_date, to_date) -> list of {event, properties: {time}} raw export rows
        self.exports: dict[tuple[str, str], list[dict]] = {}
        self.fail_paths: dict[str, Exception] = {}
        self.calls: list[tuple[str, str]] = []

    # -- seeding -----------------------------------------------------------
    def add_annotation(self, annotation_id: str, description: str, *, day="2026-11-01"):
        self.annotations[annotation_id] = {
            "id": annotation_id,
            "date": day,
            "project_id": int(self.project_id),
            "description": description,
        }
        return self.annotations[annotation_id]

    def add_cohort(self, cohort_id: str, name: str, **fields):
        self.cohorts[cohort_id] = {
            "id": cohort_id,
            "name": name,
            "count": 42,  # volatile: must never be rendered
            "created": "2026-01-15",
            **fields,
        }
        return self.cohorts[cohort_id]

    def add_funnel(self, funnel_id: str, name: str, steps: list[str]):
        self.funnels[funnel_id] = {
            "funnel_id": funnel_id,
            "name": name,
            "steps": [{"event": step} for step in steps],
        }
        return self.funnels[funnel_id]

    def add_schema(self, name: str, description: str = "", properties: list[str] | None = None):
        self.schemas[name] = {
            "name": name,
            "description": description,
            "properties": [{"name": p} for p in (properties or [])],
        }
        return self.schemas[name]

    def add_event(self, from_date: str, to_date: str, name: str, *, epoch_seconds: int):
        self.exports.setdefault((from_date, to_date), []).append(
            {"event": name, "properties": {"time": epoch_seconds}}
        )

    # -- the api -----------------------------------------------------------
    def count(self, path: str) -> int:
        return sum(1 for _, called in self.calls if called == path)

    def request(
        self,
        method: str,
        path: str,
        *,
        base: str = "query",
        body=None,
        params=None,
        check_active=None,
    ):
        self.calls.append((method, path))
        if check_active is not None:
            check_active()
        if path in self.fail_paths:
            raise self.fail_paths[path]
        if path == "/projects":
            assert base == "query"
            projects = {self.project_id: {"id": self.project_id, "name": "Product"}}
            projects.update({pid: {"id": pid} for pid in self.extra_projects})
            return {"results": {"projects": projects}}
        if method == "GET" and path == "/annotations":
            assert base == "query"
            return [dict(a) for a in self.annotations.values()]
        if method == "GET" and path == "/cohorts/list":
            assert base == "query"
            return [dict(c) for c in self.cohorts.values()]
        if method == "GET" and path == "/funnels/list":
            assert base == "query"
            assert params == {"project_id": self.project_id}
            return [dict(f) for f in self.funnels.values()]
        prefix = f"/projects/{self.project_id}/schemas"
        if method == "GET" and path == prefix:
            assert base == "app"
            return {"schemas": [dict(s) for s in self.schemas.values()]}
        raise AssertionError(f"unexpected call {method} {path} (base={base})")

    def export_events(self, from_date, to_date, *, check_active=None):
        self.calls.append(("GET", "/export"))
        if check_active is not None:
            check_active()
        key = (from_date.isoformat(), to_date.isoformat())
        if key not in self.exports:
            raise MixpanelAPIError(f"unexpected export window {key}")
        return [dict(event, properties=dict(event["properties"])) for event in self.exports[key]]
