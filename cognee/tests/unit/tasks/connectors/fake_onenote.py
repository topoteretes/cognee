# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Offline Microsoft Graph fake: notebooks, sections, pages and HTML served through ``httpx``."""

from collections import Counter
from copy import deepcopy

import httpx

from cognee.tasks.ingestion.connectors import onenote_source


class Graph:
    def __init__(self):
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
        self.gone_code = "20113"
        self.hidden = set()

    def hide_page(self, page_id):
        """The page answers 404 + 20102 ("does not exist"), as for a page this token cannot see."""
        self.hidden.add(page_id)

    def delete_section(self, notebook="a"):
        """Delete the section: its pages vanish and Graph answers 20113 for each of them."""
        self.pages[notebook] = []
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
        assert request.headers["Authorization"] == "Bearer offline-token"
        path = request.url.path.removeprefix("/v1.0/me/onenote/")
        self.requests.append((path, dict(request.url.params)))
        if path in self.errors:
            status, code = self.errors[path]
            return httpx.Response(
                status, json={"error": {"code": code, "message": "Fixture error"}}
            )
        parts = path.split("/")
        if path == "notebooks":
            values = [
                {"id": identity, "displayName": title} for identity, title in self.notebooks.items()
            ]
        elif parts[0] == "notebooks" and len(parts) == 2:
            return httpx.Response(
                200, json={"id": parts[1], "displayName": self.notebooks[parts[1]]}
            )
        elif parts[0] == "notebooks" and parts[-1] == "sections":
            values = [{"id": f"s-{parts[1]}", "displayName": self.sections[parts[1]]}]
        elif parts[-1] == "sectionGroups":
            values = []
        elif parts[0] == "sections" and parts[-1] == "pages":
            values = self.pages[parts[1].removeprefix("s-")]
        elif parts[0] == "pages" and parts[-1] == "content":
            return httpx.Response(
                200,
                text=f"<html><body>{self.html[parts[1]]}</body></html>",
                headers={"content-type": "text/html"},
            )
        elif parts[0] == "pages":
            for notebook, pages in self.pages.items():
                for page in pages:
                    if page["id"] == parts[1]:
                        metadata = deepcopy(page)
                        metadata["parentNotebook"] = {
                            "id": notebook,
                            "displayName": self.notebooks[notebook],
                        }
                        metadata["parentSection"] = {
                            "id": f"s-{notebook}",
                            "displayName": self.sections[notebook],
                        }
                        return httpx.Response(200, json=metadata)
            code = "20102" if parts[1] in self.hidden else self.gone_code
            return httpx.Response(404, json={"error": {"code": code, "message": "Page gone"}})
        else:
            raise AssertionError(f"Unexpected Graph request: {request.url}")
        assert request.url.params.get("$orderby") == "createdDateTime asc"
        values = sorted(values, key=lambda row: (row.get("createdDateTime", ""), row["id"]))
        skip = int(request.url.params.get("$skip", "0"))
        top = int(request.url.params.get("$top", "100"))
        return httpx.Response(200, json={"value": values[skip : skip + top]})

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
