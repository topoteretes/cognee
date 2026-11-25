# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Offline Microsoft Graph fake: notebooks, sections, pages and HTML served through ``httpx``."""

import json
from collections import Counter, deque
from copy import deepcopy
from urllib.parse import urlencode

import httpx

from cognee.tasks.ingestion.connectors import onenote_source


class Graph:
    def __init__(self, token="offline-token"):
        self.token = token
        self.notebooks = {"a": "Project Alpha", "b": "Excluded"}
        self.sections = {"a": "Planning", "b": "Private"}
        self.pages = {"a": [self.page("p1")], "b": [self.page("secret")]}
        self.html = {
            "p1": "<html><body><h1>Plan</h1><p>Original body</p>"
            '<img alt="Architecture diagram" src="https://graph.microsoft.com/v1.0/'
            'me/onenote/resources/image/content"></body></html>',
            "secret": "<p>Excluded content</p>",
        }
        self.errors = {}
        self.requests = []
        self.hidden = set()
        self.deleted_sections = set()
        # Page ids a listing has served. A page that is no longer listed anywhere answers
        # 20113 (it was deleted) if it was served once, and 20102 (never existed) if not.
        self.served = set()
        # Container path ("notebooks/a", "sectionGroups/g1") to its section groups, and
        # section group id to its sections, as [(id, display name)].
        self.groups = {}
        self.group_sections = {}
        self.root_sections_hidden = False
        # When set, listings answer this many rows plus an @odata.nextLink instead of
        # relying on $skip, as Graph does for its own paging.
        self.next_link_size = None
        # Path to a queue of scripted answers: an httpx.Response, an Exception to raise, or a
        # (status, headers) tuple. Consumed one per request before normal handling.
        self.scripted = {}
        self.content_types = {}
        # Echo the bearer token and the request URL into error bodies, like a hostile or
        # careless server would.
        self.echo_token = False
        self.echoed = 0

    def script(self, path, *answers):
        """Answer the next requests for ``path`` with these, then fall back to normal handling."""
        self.scripted.setdefault(path, deque()).extend(answers)

    def nest_sections(self, group="group", name="Research", container="notebooks/a"):
        """Move the notebook's section into a section group, as a nested notebook has it."""
        self.groups.setdefault(container, []).append((group, name))
        self.group_sections[group] = [("s-a", self.sections["a"])]
        self.root_sections_hidden = True

    def mark_deleted(self, *page_ids):
        """These pages existed once and are gone: Graph answers 20113 for them, not 20102."""
        self.served.update(page_ids)

    def hide_page(self, page_id):
        """The page answers 404 + 20102 ("does not exist"), as for a page this token cannot see."""
        self.hidden.add(page_id)

    def delete_section(self, notebook="a"):
        """Delete the section: its pages vanish and Graph answers 20113 for each of them."""
        self.pages[notebook] = []
        self.deleted_sections.add(f"s-{notebook}")
        self.errors[f"sections/s-{notebook}/pages"] = (404, "20113")

    @staticmethod
    def page(identity, timestamp="2026-10-01T00:00:00Z"):
        return {
            "id": identity,
            "title": identity,
            "createdDateTime": "2026-01-01T00:00:00Z",
            "lastModifiedDateTime": timestamp,
            "links": {"oneNoteWebUrl": {"href": f"https://onenote.com/{identity}"}},
        }

    def handle(self, request):
        assert request.headers["Authorization"] == f"Bearer {self.token}"
        path = request.url.path.removeprefix("/v1.0/me/onenote/")
        self.requests.append((path, dict(request.url.params)))
        if self.scripted.get(path):
            answer = self.scripted[path].popleft()
            if isinstance(answer, Exception):
                raise answer
            if isinstance(answer, tuple):
                status, headers = answer
                return httpx.Response(status, headers=headers, json={"error": {"code": "10001"}})
            return answer
        if path in self.errors:
            status, code = self.errors[path]
            return self.error(request, status, code)
        parts = path.split("/")
        if path == "notebooks":
            values = [
                {"id": identity, "displayName": title} for identity, title in self.notebooks.items()
            ]
        elif parts[0] in ("sections", "sectionGroups") and len(parts) == 2:
            if parts[1] in self.deleted_sections:
                return self.error(request, 404, "20113")
            return httpx.Response(200, json={"id": parts[1], "displayName": parts[1]})
        elif parts[0] == "notebooks" and len(parts) == 2:
            return httpx.Response(
                200, json={"id": parts[1], "displayName": self.notebooks[parts[1]]}
            )
        elif parts[0] == "notebooks" and parts[-1] == "sections":
            values = [] if self.root_sections_hidden else self.notebook_section(parts[1])
        elif parts[0] == "sectionGroups" and parts[-1] == "sections":
            values = [
                {"id": identity, "displayName": title}
                for identity, title in self.group_sections.get(parts[1], [])
            ]
        elif parts[-1] == "sectionGroups":
            values = [
                {"id": identity, "displayName": title}
                for identity, title in self.groups.get("/".join(parts[:-1]), [])
            ]
        elif parts[0] == "sections" and parts[-1] == "pages":
            values = self.pages[parts[1].removeprefix("s-")]
            self.served.update(page["id"] for page in values)
        elif parts[0] == "pages" and parts[-1] == "content":
            return httpx.Response(
                200,
                text=f"<html><body>{self.html[parts[1]]}</body></html>",
                headers={"content-type": self.content_types.get(parts[1], "text/html")},
            )
        elif parts[0] == "pages":
            return self.page_metadata(request, parts[1])
        else:
            raise AssertionError(f"Unexpected Graph request: {request.url}")
        return self.listing(request, values)

    def notebook_section(self, notebook):
        return [{"id": f"s-{notebook}", "displayName": self.sections[notebook]}]

    def error(self, request, status, code):
        message = "Fixture error"
        if self.echo_token:
            message = f"Rejected {self.token} for {request.url}"
            self.echoed += 1
        return httpx.Response(status, json={"error": {"code": code, "message": message}})

    def listing(self, request, values):
        """One page of a collection: ordered by creation time, paged by $skip or nextLink."""
        params = request.url.params
        assert params.get("$orderby") == "createdDateTime asc"
        values = sorted(values, key=lambda row: (row.get("createdDateTime", ""), row["id"]))
        top = int(params.get("$top", "100"))
        if self.next_link_size is None:
            skip = int(params.get("$skip", "0"))
            return httpx.Response(200, json={"value": values[skip : skip + top]})
        skip = int(params.get("$skiptoken", "0"))
        body = {"value": values[skip : skip + self.next_link_size]}
        if skip + self.next_link_size < len(values):
            query = urlencode(
                {
                    "$orderby": "createdDateTime asc",
                    "$top": top,
                    "$skiptoken": skip + self.next_link_size,
                }
            )
            body["@odata.nextLink"] = f"https://graph.microsoft.com{request.url.path}?{query}"
        return httpx.Response(200, json=body)

    def page_metadata(self, request, page_id):
        """GET /pages/{id}: the parents come back only when asked for with $expand."""
        expand = set(request.url.params.get("$expand", "").split(","))
        for notebook, pages in self.pages.items():
            for page in pages:
                if page["id"] == page_id:
                    metadata = deepcopy(page)
                    if "parentNotebook" in expand:
                        metadata["parentNotebook"] = {
                            "id": notebook,
                            "displayName": self.notebooks[notebook],
                        }
                    if "parentSection" in expand:
                        metadata["parentSection"] = {
                            "id": f"s-{notebook}",
                            "displayName": self.sections[notebook],
                        }
                    return httpx.Response(200, json=metadata)
        if page_id in self.served and page_id not in self.hidden:
            return self.error(request, 404, "20113")
        return self.error(request, 404, "20102")

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def content_requests(self):
        return Counter(path for path, _ in self.requests if path.endswith("/content"))


def sync(pipeline, graph, selection=("a",), account="tenant:user", **kwargs):
    tables = []
    for notebook in selection:
        with graph.client() as client:
            resource = onenote_source(
                "offline-token",
                notebook_id=notebook,
                account_id=account,
                http_client=client,
                **kwargs,
            )
            tables.append(resource.name)
            pipeline.run(resource, write_disposition="merge")
            graph.last_stats = resource.cognee_sync_stats
    return tables


def rows(pipeline, table):
    # Table names are generated by the connector, never supplied by a provider.
    assert table.replace("_", "").isalnum()
    with (
        pipeline.sql_client() as client,
        client.execute_query(f'SELECT id, title, content FROM "{table}"') as cursor,
    ):
        return list(cursor.fetchall())


def structures(pipeline, table):
    """``{page id: ancestors}`` as staged in the ``cognee_structure`` column, nearest parent first."""
    assert table.replace("_", "").isalnum()
    with (
        pipeline.sql_client() as client,
        client.execute_query(f'SELECT id, cognee_structure FROM "{table}"') as cursor,
    ):
        return {page: json.loads(value)["ancestors"] for page, value in cursor.fetchall()}


def loaded_rows(pipeline, table):
    """How many rows the last run of ``pipeline`` extracted for ``table`` (tombstones included)."""
    return pipeline.last_trace.last_normalize_info.row_counts[table]


def resource_cache(pipeline, table):
    for source in pipeline.state["sources"].values():
        resource = source.get("resources", {}).get(table)
        if resource is not None:
            return resource["onenote_cache"]
    raise AssertionError(f"Missing persisted resource state for {table}")


def deletion_graph(count=12):
    """Twelve pages whose section is deleted later: Graph answers 20113 for each of them."""
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"deleted-{index:02d}") for index in range(count)]
    graph.html.update({page["id"]: f"<p>Body {page['id']}</p>" for page in graph.pages["a"]})
    return graph
