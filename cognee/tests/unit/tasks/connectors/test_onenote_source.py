# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""One OneNote notebook is one merge resource, and an error never forgets a page.

The walk runs through a real ``dlt`` pipeline on SQLite against the Graph fake: incremental HTML
fetching, the body cache that survives a failed run, deletion only on confirmed evidence, the
request budget and the arguments checked when the source is built. No network, no LLM.
"""

import json
import subprocess
import sys
from copy import deepcopy

import dlt
import fake_onenote
import httpx
import pytest
from dlt.extract.exceptions import InvalidResourceDataTypeMultiplePipes
from dlt.pipeline.exceptions import PipelineStepFailed
from fake_onenote import (
    Graph,
    deletion_graph,
    loaded_rows,
    resource_cache,
    rows,
    structures,
    sync,
)

from cognee.tasks.ingestion.connectors import onenote_source
from cognee.tasks.ingestion.connectors.onenote import OneNoteError


@pytest.fixture(autouse=True)
def _no_dlt_telemetry(monkeypatch):
    # DLT's telemetry executor otherwise waits for network retries at interpreter shutdown.
    monkeypatch.setenv("RUNTIME__DLTHUB_TELEMETRY", "false")


@pytest.fixture
def pipeline(tmp_path):
    return dlt.pipeline(
        pipeline_name="onenote_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path}/data.db"),
        dataset_name="notes",
        pipelines_dir=str(tmp_path / "state"),
    )


def test_initial_selection_html_and_unchanged_replay(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    first = rows(pipeline, table)
    assert len(first) == 1
    assert "Original body" in first[0][2]
    assert "Architecture diagram" in first[0][2]
    assert "resources/image/content" in first[0][2]
    # The row text is the page body only: where the page sits travels as its structure.
    assert "Project Alpha" not in first[0][2] and "Planning" not in first[0][2]
    assert "Source:" not in first[0][2] and "onenote.com" not in first[0][2]
    assert structures(pipeline, table) == {
        "p1": [
            {"kind": "section", "id": "s-a", "name": "Planning"},
            {"kind": "notebook", "id": "a", "name": "Project Alpha"},
        ]
    }
    assert loaded_rows(pipeline, table) == 1
    calls = graph.content_requests()
    sync(pipeline, graph)
    assert rows(pipeline, table) == first
    assert graph.content_requests() == calls
    # Nothing changed, so nothing is staged again.
    assert loaded_rows(pipeline, table) == 0
    assert graph.last_stats["complete"] == 1 and graph.last_stats["skipped"] == 1
    assert not any("secret" in path for path, _ in graph.requests)


def test_rename_updates_the_structure_and_keeps_the_text(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.notebooks["a"] = "Project Beta"
    graph.sections["a"] = "Execution"
    sync(pipeline, graph)
    # The row is published again so the structure is written, and its text is the same, so
    # cognee derives the same content hash and does not extract it again.
    assert loaded_rows(pipeline, table) == 1
    assert rows(pipeline, table) == before
    assert structures(pipeline, table)["p1"] == [
        {"kind": "section", "id": "s-a", "name": "Execution"},
        {"kind": "notebook", "id": "a", "name": "Project Beta"},
    ]
    assert graph.last_stats["complete"] == 1


def test_edits_timestamp_only_and_older_new_page(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    first = rows(pipeline, table)
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-02T00:00:00Z"
    sync(pipeline, graph)
    assert rows(pipeline, table) == first
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = "<p>Updated body</p>"
    graph.pages["a"].append(Graph.page("older", "2020-01-01T00:00:00Z"))
    graph.html["older"] = "<p>Older new page</p>"
    sync(pipeline, graph)
    content = "\n".join(row[2] for row in rows(pipeline, table))
    assert "Updated body" in content and "Older new page" in content
    assert "Original body" not in content


@pytest.mark.parametrize(
    "status,code",
    [
        (403, "40003"),
        (404, "20112"),
        (404, "unknown"),
        (401, "20113"),
        (403, "20102"),
    ],
)
def test_missing_page_access_or_ambiguous_failure_preserves_table(pipeline, status, code):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    graph.errors["pages/p1"] = (status, code)
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_failed_notebook_preserves_its_snapshot(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.errors["notebooks/a/sections"] = (403, "40003")
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert rows(pipeline, table) == before
    del graph.errors["notebooks/a/sections"]
    sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1


def test_deselection_preserves_table_and_reselection_cache(pipeline):
    graph = Graph()
    tables = sync(pipeline, graph, ("a", "b"))
    before = {table: rows(pipeline, table) for table in tables}
    calls = graph.content_requests()
    sync(pipeline, graph, ("a",))
    assert {table: rows(pipeline, table) for table in tables} == before
    sync(pipeline, graph, ("a", "b"))
    assert graph.content_requests() == calls


def test_account_scope_produces_separate_resources(pipeline):
    graph = Graph()
    first = sync(pipeline, graph, account="tenant:first")
    second = sync(pipeline, graph, account="tenant:second")
    assert set(first).isdisjoint(second)
    assert all(len(rows(pipeline, table)) == 1 for table in first + second)
    assert graph.content_requests()["pages/p1/content"] == 2


@pytest.mark.parametrize("count", [0, 3, 100, 200])
def test_exact_pagination_boundaries(pipeline, count):
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{i}") for i in range(count)]
    graph.html.update({f"p{i}": f"<p>Body {i}</p>" for i in range(count)})
    (table,) = sync(pipeline, graph, max_requests=1000)
    assert len(rows(pipeline, table)) == count


def test_request_budget_preserves_previous_snapshot(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    sync(pipeline, graph, max_requests=1)
    assert graph.last_stats["failed_budget"] == 1
    assert graph.last_stats["requests"] == 1
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError):
        onenote_source("offline-token", notebook_id="", account_id="account")


def test_confirmed_move_outside_selection_deletes_copy_without_fetch(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    calls = graph.content_requests()
    graph.pages["b"].append(graph.pages["a"].pop())
    graph.html["p1"] = "<p>Excluded update</p>"
    sync(pipeline, graph)
    assert before
    assert rows(pipeline, table) == []
    assert graph.content_requests() == calls


def test_move_between_independently_synced_notebooks(pipeline):
    graph = Graph()
    tables = sync(pipeline, graph, ("a", "b"))
    calls = graph.content_requests()
    graph.pages["b"].append(graph.pages["a"].pop())
    sync(pipeline, graph, ("a", "b"))
    all_rows = [row for table in tables for row in rows(pipeline, table)]
    assert len(all_rows) == 2
    assert any("Original body" in row[2] for row in all_rows)
    assert structures(pipeline, tables[1])["p1"][-1] == {
        "kind": "notebook",
        "id": "b",
        "name": "Excluded",
    }
    assert graph.content_requests()["pages/p1/content"] == calls["pages/p1/content"] + 1


def test_renderer_version_invalidates_unchanged_cache(pipeline, monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    graph = Graph()
    sync(pipeline, graph)
    monkeypatch.setattr(onenote, "_RENDERER_VERSION", onenote._RENDERER_VERSION + 1)
    sync(pipeline, graph)
    assert graph.content_requests()["pages/p1/content"] == 2


def test_pending_load_recovery_then_fresh_extraction(pipeline, monkeypatch):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = "<p>Recover me</p>"
    original_load = pipeline.load

    def failed_load(*args, **kwargs):
        raise RuntimeError("Injected load failure")

    monkeypatch.setattr(pipeline, "load", failed_load)
    with pytest.raises(RuntimeError, match="Injected load failure"):
        sync(pipeline, graph)
    calls = graph.content_requests()
    monkeypatch.setattr(pipeline, "load", original_load)
    with graph.client() as client:
        source = onenote_source(
            "offline-token", notebook_id="a", account_id="tenant:user", http_client=client
        )
        pipeline.run(source)
        assert source.cognee_sync_stats == {}
    assert "Recover me" in rows(pipeline, table)[0][2]
    sync(pipeline, graph)
    assert graph.content_requests() == calls
    assert "Recover me" in rows(pipeline, table)[0][2]


def test_declared_document_and_dataset_scope():
    from cognee.tasks.ingestion.dlt_utils import (
        DOCUMENT_SOURCE_ATTR,
        PIPELINE_SCOPE_ATTR,
        pipeline_name_for_source,
    )

    source = onenote_source("offline-token", notebook_id="a", account_id="tenant:user")
    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "onenote"
    assert getattr(source, PIPELINE_SCOPE_ATTR)
    assert pipeline_name_for_source(source, "first") != pipeline_name_for_source(source, "second")


def test_confirmed_parent_deletion_reconciles_descendant_and_empty_table(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    del graph.notebooks["a"]
    graph.pages["a"] = []
    graph.errors["notebooks/a"] = (404, "20113")
    sync(pipeline, graph)
    assert rows(pipeline, table) == []
    assert any(path == "pages/p1" for path, _ in graph.requests)


@pytest.mark.parametrize("status,code", [(403, "40003"), (404, "20102")])
def test_inaccessible_parent_is_not_treated_as_deletion(pipeline, status, code):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    del graph.notebooks["a"]
    graph.errors["notebooks/a"] = (status, code)
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_malformed_collection_preserves_loaded_table(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    original = graph.handle

    def malformed(request):
        if request.url.path.endswith("/sections/s-a/pages"):
            return httpx.Response(200, json={"value": None})
        return original(request)

    graph.handle = malformed
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_later_pagination_failure_preserves_snapshot_and_cache(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    calls = graph.content_requests()
    original = graph.handle
    graph.pages["a"] = [Graph.page(f"p{i}") for i in range(101)]

    def failed_batch(request):
        if (
            request.url.path.endswith("/sections/s-a/pages")
            and request.url.params.get("$skip") == "100"
        ):
            return httpx.Response(403, json={"error": {"code": "40003"}})
        return original(request)

    graph.handle = failed_batch
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before
    assert graph.content_requests() == calls
    graph.handle = original
    graph.pages["a"] = [Graph.page("p1")]
    sync(pipeline, graph)
    assert rows(pipeline, table) == before
    assert graph.content_requests() == calls


@pytest.fixture
def nested_graph():
    graph = Graph()
    graph.nest_sections()
    return graph


def test_section_group_recursion_records_the_whole_ancestor_chain(pipeline, nested_graph):
    graph = nested_graph
    (table,) = sync(pipeline, graph)
    assert structures(pipeline, table)["p1"] == [
        {"kind": "section", "id": "s-a", "name": "Planning"},
        {"kind": "sectionGroup", "id": "group", "name": "Research"},
        {"kind": "notebook", "id": "a", "name": "Project Alpha"},
    ]


@pytest.mark.parametrize("container_path", ["sections/s-a/pages", "sectionGroups/group/sections"])
def test_confirmed_section_or_group_deletion_reconciles_pages(
    pipeline, nested_graph, container_path
):
    graph = nested_graph
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    graph.errors[container_path] = (404, "20113")
    sync(pipeline, graph)
    assert rows(pipeline, table) == []
    assert any(path == "pages/p1" for path, _ in graph.requests)


@pytest.mark.parametrize("container_path", ["sections/s-a/pages", "sectionGroups/group/sections"])
def test_section_or_group_access_loss_preserves_snapshot(pipeline, nested_graph, container_path):
    graph = nested_graph
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.errors[container_path] = (403, "40003")
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_body_checkpoint_survives_interrupted_run_and_pipeline_reload(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] += [Graph.page("p2"), Graph.page("p3")]
    graph.html.update(p2="<p>Checkpoint body</p>", p3="<p>Recovered body</p>")
    graph.errors["pages/p3/content"] = (403, "40003")
    sync(pipeline, graph)
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["failed"] > 0
    assert rows(pipeline, table) == before
    calls = graph.content_requests()
    assert calls["pages/p2/content"] == 1
    # Read the durable DLT state in a fresh interpreter, not the extraction context.
    persisted = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dlt,json,sys; "
                "p=dlt.attach(sys.argv[1], pipelines_dir=sys.argv[2]); "
                "print(json.dumps(p.state, default=str))"
            ),
            pipeline.pipeline_name,
            pipeline.pipelines_dir,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Checkpoint body" in json.dumps(json.loads(persisted.stdout))
    del graph.errors["pages/p3/content"]
    restored = dlt.attach(pipeline.pipeline_name, pipelines_dir=pipeline.pipelines_dir)
    restored.destination = pipeline.destination
    sync(restored, graph)
    assert graph.last_stats["complete"] == 1
    assert len(rows(restored, table)) == 3
    assert graph.content_requests()["pages/p2/content"] == 1
    assert graph.content_requests()["pages/p3/content"] == calls["pages/p3/content"] + 1


def test_replace_override_rejected_before_requests_and_preserves_rows(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    requests = len(graph.requests)
    with graph.client() as client:
        resource = onenote_source(
            "offline-token", notebook_id="a", account_id="tenant:user", http_client=client
        )
        with pytest.raises((PipelineStepFailed, ValueError), match="merge"):
            pipeline.run(resource, write_disposition="replace")
    assert len(graph.requests) == requests
    assert rows(pipeline, table) == before


def test_connection_resource_name_cannot_be_reused_for_other_notebook(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph, resource_name="stable_connection")
    before = rows(pipeline, table)
    with pytest.raises(PipelineStepFailed, match="(?i)(identity|connection|notebook|scope)"):
        sync(pipeline, graph, selection=("b",), resource_name="stable_connection")
    assert rows(pipeline, table) == before


def test_explicit_ordering_survives_fallback_pagination(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{i}") for i in range(100)]
    graph.html.update({f"p{i}": "<p>Body</p>" for i in range(100)})
    sync(pipeline, graph)
    page_calls = [params for path, params in graph.requests if path == "sections/s-a/pages"]
    assert len(page_calls) == 2
    assert page_calls == [
        {"$orderby": "createdDateTime asc", "$top": "100", "$skip": "0"},
        {"$orderby": "createdDateTime asc", "$top": "100", "$skip": "100"},
    ]


def test_valid_nesting_beyond_thirty_groups(pipeline):
    graph = Graph()
    graph.root_sections_hidden = True
    graph.groups["notebooks/a"] = [("g0", "Level 0")]
    for level in range(35):
        graph.groups[f"sectionGroups/g{level}"] = [(f"g{level + 1}", f"Level {level + 1}")]
    graph.group_sections["g35"] = [("s-a", "Deep planning")]
    (table,) = sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1
    ancestors = structures(pipeline, table)["p1"]
    assert [entry["name"] for entry in ancestors[:2]] == ["Deep planning", "Level 35"]
    assert [entry["name"] for entry in ancestors[-2:]] == ["Level 0", "Project Alpha"]
    assert len(ancestors) == 38


def test_duplicate_after_collection_changes_aborts_without_partial_rows(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = [Graph.page(f"p{i}") for i in range(100)]
    original = graph.handle

    def changed_batch(request):
        if (
            request.url.path.endswith("/sections/s-a/pages")
            and request.url.params.get("$skip") == "100"
        ):
            return httpx.Response(200, json={"value": [Graph.page("p0")]})
        return original(request)

    graph.handle = changed_batch
    sync(pipeline, graph)
    assert graph.last_stats["failed"] > 0
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_account_notebook_identity_tuple_is_unambiguous():
    first = onenote_source("offline-token", account_id="a:b", notebook_id="c")
    second = onenote_source("offline-token", account_id="a", notebook_id="b:c")
    assert first.name != second.name


def test_bulk_deletion_confirmation_resumes_across_fresh_process(pipeline, tmp_path):
    graph = deletion_graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    graph.requests.clear()
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["failed_budget"] == 1
    assert rows(pipeline, table) == before
    first_confirmed = resource_cache(pipeline, table)["confirmed_missing"]
    assert len(first_confirmed) == 4
    assert set(first_confirmed.values()) == {True}
    first_checks = [path for path, _ in graph.requests if path.startswith("pages/")]

    # Execute the next extraction in a genuinely fresh interpreter and Graph fixture.
    script = """
import dlt, json, sys
from runpy import run_path
ns = run_path(sys.argv[1])
pipeline = dlt.attach(sys.argv[2], pipelines_dir=sys.argv[3])
pipeline.destination = dlt.destinations.sqlalchemy(sys.argv[4])
graph = ns['Graph']()
graph.pages['a'] = []
graph.mark_deleted(*[f'deleted-{index:02d}' for index in range(12)])
ns['sync'](pipeline, graph, max_requests=8)
result = {'stats': graph.last_stats, 'cache': ns['resource_cache'](pipeline, sys.argv[5]),
          'requests': graph.requests}
print('RESUMED=' + json.dumps(result, default=str))
"""
    resumed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            fake_onenote.__file__,
            pipeline.pipeline_name,
            pipeline.pipelines_dir,
            f"sqlite:///{tmp_path}/data.db",
            table,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(
        next(
            line.removeprefix("RESUMED=")
            for line in resumed.stdout.splitlines()
            if line.startswith("RESUMED=")
        )
    )
    assert result["stats"]["complete"] == 0
    assert result["stats"]["failed_budget"] == 1
    assert len(result["cache"]["confirmed_missing"]) == 8
    assert set(first_confirmed).issubset(result["cache"]["confirmed_missing"])
    second_checks = [path for path, _ in result["requests"] if path.startswith("pages/")]

    restored = dlt.attach(pipeline.pipeline_name, pipelines_dir=pipeline.pipelines_dir)
    restored.destination = pipeline.destination
    fresh_graph = Graph()
    fresh_graph.pages["a"] = []
    fresh_graph.mark_deleted(*(page_id for page_id, _, _ in before))
    sync(restored, fresh_graph, max_requests=8)
    assert fresh_graph.last_stats["complete"] == 1
    assert fresh_graph.last_stats["deleted"] == 12
    assert fresh_graph.last_stats["requests"] == 8
    assert rows(restored, table) == []
    assert resource_cache(restored, table)["published"] == {}
    assert not resource_cache(restored, table).get("confirmed_missing")
    all_checks = (
        first_checks
        + second_checks
        + [path for path, _ in fresh_graph.requests if path.startswith("pages/")]
    )
    assert len(all_checks) == len(set(all_checks)) == 12


@pytest.mark.parametrize("status,code", [(403, "40003"), (404, "unknown")])
def test_uncertain_deletion_retains_prior_confirmations_and_documents(pipeline, status, code):
    graph = deletion_graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    sync(pipeline, graph, max_requests=8)
    confirmed = deepcopy(resource_cache(pipeline, table)["confirmed_missing"])
    unresolved = sorted({page_id for page_id, _, _ in before} - set(confirmed))
    assert len(confirmed) == 4
    graph.errors[f"pages/{unresolved[0]}"] = (status, code)
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["failed"] == 1
    assert rows(pipeline, table) == before
    assert resource_cache(pipeline, table)["confirmed_missing"] == confirmed
    del graph.errors[f"pages/{unresolved[0]}"]
    for _ in range(2):
        sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 1
    assert rows(pipeline, table) == []


def test_reappearance_invalidates_prior_confirmation_before_later_deletion(pipeline):
    graph = deletion_graph()
    original_pages = deepcopy(graph.pages["a"])
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    sync(pipeline, graph, max_requests=8)
    confirmed = resource_cache(pipeline, table)["confirmed_missing"]
    restored_id = min(confirmed)
    graph.pages["a"] = [next(page for page in original_pages if page["id"] == restored_id)]
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert restored_id not in resource_cache(pipeline, table)["confirmed_missing"]
    assert len(rows(pipeline, table)) == 12
    graph.pages["a"] = []
    graph.errors[f"pages/{restored_id}"] = (403, "40003")
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert restored_id not in resource_cache(pipeline, table)["confirmed_missing"]
    assert len(rows(pipeline, table)) == 12
    del graph.errors[f"pages/{restored_id}"]
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert len(rows(pipeline, table)) == 12
    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 1
    assert rows(pipeline, table) == []
    assert sum(path == f"pages/{restored_id}" for path, _ in graph.requests) >= 3


def test_creation_order_pagination_is_stable_when_an_earlier_page_is_edited(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{index:03d}") for index in range(101)]
    for index, page in enumerate(graph.pages["a"]):
        page["createdDateTime"] = f"2026-01-{index // 24 + 1:02d}T{index % 24:02d}:00:00Z"
        graph.html[page["id"]] = f"<p>Body {index}</p>"
    original = graph.handle

    def edit_between_batches(request):
        if (
            request.url.path.endswith("/sections/s-a/pages")
            and request.url.params.get("$skip") == "100"
        ):
            graph.pages["a"][40]["lastModifiedDateTime"] = "2030-01-01T00:00:00Z"
        return original(request)

    graph.handle = edit_between_batches
    (table,) = sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1
    assert {row[0] for row in rows(pipeline, table)} == {page["id"] for page in graph.pages["a"]}
    page_calls = [params for path, params in graph.requests if path == "sections/s-a/pages"]
    assert [params["$skip"] for params in page_calls] == ["0", "100"]
    assert all(params["$orderby"] == "createdDateTime asc" for params in page_calls)


def test_state_holds_no_page_text_after_a_run_and_only_unpublished_bodies_after_a_cut(pipeline):
    graph = Graph()
    graph.html["p1"] = "<p>UniqueBodyOne_Alpha</p>"
    (table,) = sync(pipeline, graph)
    cache = resource_cache(pipeline, table)
    assert list(cache["published"]) == ["p1"]
    assert "UniqueBodyOne_Alpha" not in json.dumps(cache)
    assert "pending" not in cache
    graph.pages["a"] += [Graph.page("p2"), Graph.page("p3")]
    graph.html.update(p2="<p>UniqueBodyTwo_Beta</p>", p3="<p>Eventually succeeds</p>")
    graph.errors["pages/p3/content"] = (403, "40003")
    sync(pipeline, graph)
    cache = resource_cache(pipeline, table)
    assert graph.last_stats["complete"] == 0
    assert list(cache["published"]) == ["p1"]
    assert list(cache["pending"]) == ["p2"]
    assert json.dumps(cache).count("UniqueBodyTwo_Beta") == 1
    assert "UniqueBodyOne_Alpha" not in json.dumps(cache)
    assert len(rows(pipeline, table)) == 1
    del graph.errors["pages/p3/content"]
    sync(pipeline, graph)
    cache = resource_cache(pipeline, table)
    assert graph.last_stats["complete"] == 1
    assert sorted(cache["published"]) == ["p1", "p2", "p3"]
    assert "pending" not in cache
    assert "UniqueBodyTwo_Beta" not in json.dumps(cache)
    # Each published page costs one short fingerprint, not its text.
    assert all(len(value) == 24 for value in cache["published"].values())


def test_malformed_page_links_checkpoint_bodies_and_recover_after_attach(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = "<p>Healthy edit held until full recovery</p>"
    graph.pages["a"].append(Graph.page("p2"))
    graph.pages["a"][1]["links"] = []
    graph.html["p2"] = "<p>Malformed page body</p>"
    sync(pipeline, graph)
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before
    calls = graph.content_requests()
    assert calls["pages/p1/content"] == 2
    # The malformed page is refused before its body is read; the healthy edit is kept.
    assert "pages/p2/content" not in calls
    assert (
        resource_cache(pipeline, table)["pending"]["p1"]["text"]
        == "Healthy edit held until full recovery"
    )
    restored = dlt.attach(pipeline.pipeline_name, pipelines_dir=pipeline.pipelines_dir)
    restored.destination = pipeline.destination
    graph.pages["a"][1]["links"] = {"oneNoteWebUrl": {"href": "https://onenote.com/p2"}}
    sync(restored, graph)
    assert graph.last_stats["complete"] == 1
    assert graph.content_requests()["pages/p1/content"] == calls["pages/p1/content"]
    assert graph.content_requests()["pages/p2/content"] == 1
    final_content = "\n".join(row[2] for row in rows(restored, table))
    assert "Healthy edit held until full recovery" in final_content
    assert "Malformed page body" in final_content


@pytest.mark.parametrize("name", ["", "   ", 0, False, [], {}])
def test_explicit_invalid_connection_names_rejected_before_requests(name):
    graph = Graph()
    with (
        graph.client() as client,
        pytest.raises((TypeError, ValueError), match="(?i)(name|connection)"),
    ):
        onenote_source(
            "offline-token",
            notebook_id="a",
            account_id="tenant:user",
            resource_name=name,
            http_client=client,
        )
    assert graph.requests == []


def test_normalized_connection_names_collide_before_http():
    from cognee.tasks.ingestion.dlt_utils import PIPELINE_SCOPE_ATTR

    graph = Graph()
    with graph.client() as client:
        first = onenote_source(
            "offline-token",
            notebook_id="a",
            account_id="tenant:user",
            resource_name="Onenote_A",
            http_client=client,
        )
        second = onenote_source(
            "offline-token",
            notebook_id="b",
            account_id="tenant:user",
            resource_name="onenote_a",
            http_client=client,
        )
        assert first.name == second.name == "onenote_a"
        assert (
            getattr(first, PIPELINE_SCOPE_ATTR)
            == getattr(second, PIPELINE_SCOPE_ATTR)
            == "onenote_a"
        )

        @dlt.source(name="duplicate_connections")
        def duplicate_connections():
            return [first, second]

        with pytest.raises(InvalidResourceDataTypeMultiplePipes):
            duplicate_connections()
    assert graph.requests == []


def test_normalized_connection_alias_cannot_rebind_state(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph, resource_name="Onenote_A")
    assert table == "onenote_a"
    before = rows(pipeline, table)
    calls = len(graph.requests)
    with pytest.raises(PipelineStepFailed, match="(?i)(identity|connection|notebook|scope)"):
        sync(pipeline, graph, selection=("b",), resource_name="onenote_a")
    assert rows(pipeline, table) == before
    assert len(graph.requests) == calls


@pytest.mark.parametrize("version", [None, 0, 1, 2, 3])
def test_document_sync_version_guard_accepts_only_supported_versions(monkeypatch, version):
    from cognee.tasks.ingestion import dlt_utils

    if version is None:
        monkeypatch.delattr(dlt_utils, "DOCUMENT_SYNC_VERSION", raising=False)
    else:
        monkeypatch.setattr(dlt_utils, "DOCUMENT_SYNC_VERSION", version)
    if version is None or version < 3:
        with pytest.raises(RuntimeError, match="structure"):
            onenote_source("offline-token", notebook_id="a", account_id="tenant:user")
    else:
        assert onenote_source("offline-token", notebook_id="a", account_id="tenant:user").name


@pytest.mark.parametrize("failure_kind", ["malformed_links", "pagination"])
def test_observed_page_invalidates_removal_confirmation_during_incomplete_scan(
    pipeline, failure_kind
):
    graph = deletion_graph()
    original_pages = deepcopy(graph.pages["a"])
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    sync(pipeline, graph, max_requests=8)
    confirmed = deepcopy(resource_cache(pipeline, table)["confirmed_missing"])
    assert len(confirmed) == 4
    restored_id = min(confirmed)
    restored_page = next(page for page in original_pages if page["id"] == restored_id)
    original_handle = graph.handle
    if failure_kind == "malformed_links":
        malformed_page = Graph.page("zz-malformed")
        malformed_page["links"] = []
        graph.html["zz-malformed"] = "<p>Incomplete scan body</p>"
        graph.pages["a"] = [restored_page, malformed_page]
    else:
        graph.pages["a"] = [restored_page] + [
            Graph.page(f"zz-page-{index:03d}") for index in range(99)
        ]

        def failed_later_batch(request):
            if (
                request.url.path.endswith("/sections/s-a/pages")
                and request.url.params.get("$skip") == "100"
            ):
                graph.requests.append(("sections/s-a/pages", dict(request.url.params)))
                return httpx.Response(403, json={"error": {"code": "40003"}})
            return original_handle(request)

        graph.handle = failed_later_batch

    sync(pipeline, graph, max_requests=8)
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["failed"] == 1
    assert rows(pipeline, table) == before
    if failure_kind == "malformed_links":
        assert "zz-malformed" not in resource_cache(pipeline, table).get("pending", {})
    else:
        assert any(
            path == "sections/s-a/pages" and params.get("$skip") == "100"
            for path, params in graph.requests
        )
    # A positive observation invalidates old deletion evidence immediately, even
    # though later metadata or pagination prevents the traversal from completing.
    assert restored_id not in resource_cache(pipeline, table)["confirmed_missing"]
    assert set(resource_cache(pipeline, table)["confirmed_missing"]) == set(confirmed) - {
        restored_id
    }

    restored = dlt.attach(pipeline.pipeline_name, pipelines_dir=pipeline.pipelines_dir)
    restored.destination = pipeline.destination
    graph.handle = original_handle
    graph.pages["a"] = []
    graph.errors[f"pages/{restored_id}"] = (403, "40003")
    graph.requests.clear()
    sync(restored, graph, max_requests=8)
    assert any(path == f"pages/{restored_id}" for path, _ in graph.requests)
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["failed"] == 1
    assert rows(restored, table) == before
    assert restored_id not in resource_cache(restored, table)["confirmed_missing"]


def test_final_page_deletion_physically_empties_table(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    sync(pipeline, graph)
    assert rows(pipeline, table) == []
    assert graph.last_stats["deleted"] == 1


def test_page_deleted_from_a_live_notebook_is_forgotten_on_20113(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = [Graph.page("p2")]
    sync(pipeline, graph)
    assert graph.last_stats["deleted"] == 1
    assert graph.last_stats["unresolved"] == 0
    assert [row[0] for row in rows(pipeline, table)] == ["p2"]


def test_deleted_parent_confirms_missing_page_on_20113(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    del graph.notebooks["a"]
    graph.errors["notebooks/a"] = (404, "20113")
    sync(pipeline, graph)
    assert rows(pipeline, table) == []
    assert graph.last_stats["deleted"] == 1
    assert any(path == "notebooks/a" for path, _ in graph.requests)
    assert any(path == "pages/p1" for path, _ in graph.requests)


def test_page_answering_20102_is_kept_and_counted_without_blocking_the_notebook(pipeline):
    # 20102 ("does not exist") also fits a page the token can no longer see, so it is never
    # proof of deletion:
    # the page keeps its row and cached body, is counted as unresolved and asked about on every
    # run, while the rest of the notebook publishes normally (complete stays 1, failed stays 0).
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    before = {row[0]: row[2] for row in rows(pipeline, table)}
    graph.pages["a"] = [Graph.page("p2", "2026-10-09T00:00:00Z")]
    graph.html["p2"] = "<p>Edited second body</p>"
    graph.hide_page("p1")
    for attempt in range(1, 4):
        graph.requests.clear()
        sync(pipeline, graph)
        stats = graph.last_stats
        assert (stats["complete"], stats["failed"], stats["deleted"]) == (1, 0, 0)
        assert stats["unresolved"] == 1
        current = {row[0]: row[2] for row in rows(pipeline, table)}
        assert current["p1"] == before["p1"]
        assert "Edited second body" in current["p2"]
        cache = resource_cache(pipeline, table)
        assert sorted(cache["published"]) == ["p1", "p2"]
        assert cache["unresolved"] == ["p1"]
        assert not cache.get("confirmed_missing")
        assert [path for path, _ in graph.requests].count("pages/p1") == 1, attempt
    # Once the page is listed again it is an ordinary page and nothing is unresolved.
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2", "2026-10-09T00:00:00Z")]
    sync(pipeline, graph)
    assert graph.last_stats["unresolved"] == 0
    assert {row[0]: row[2] for row in rows(pipeline, table)}["p1"] == before["p1"]


def test_a_notebook_with_only_unresolved_pages_cleans_up_nothing(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    graph.hide_page("p1")
    sync(pipeline, graph)
    assert graph.last_stats["unresolved"] == 1
    assert graph.last_stats["deleted"] == 0
    assert graph.last_stats["complete"] == 1
    assert rows(pipeline, table) == before
    assert list(resource_cache(pipeline, table)["published"]) == ["p1"]


def test_unresolved_page_is_forgotten_once_graph_answers_20113_for_it(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    graph.hide_page("p1")
    sync(pipeline, graph)
    assert graph.last_stats["unresolved"] == 1
    graph.hidden.clear()
    graph.delete_section()
    sync(pipeline, graph)
    assert graph.last_stats["deleted"] == 1
    assert graph.last_stats["unresolved"] == 0
    assert rows(pipeline, table) == []


def test_page_listed_nowhere_but_resolving_in_the_same_notebook_stops_the_run(pipeline):
    # The listing omitted p1 while GET /pages/p1 still resolves inside this notebook: the
    # traversal is not trustworthy, so nothing is published and nothing is tombstoned.
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = [Graph.page("p2")]
    graph.html["p2"] = "<p>Second body</p>"
    original = graph.handle

    def resolves(request):
        if request.url.path.endswith("/pages/p1"):
            metadata = Graph.page("p1")
            metadata["parentNotebook"] = {"id": "a", "displayName": "Project Alpha"}
            return httpx.Response(200, json=metadata)
        return original(request)

    graph.handle = resolves
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["deleted"] == 0
    assert rows(pipeline, table) == before
    assert list(resource_cache(pipeline, table)["published"]) == ["p1"]


@pytest.mark.parametrize("first_sync", [True, False])
@pytest.mark.parametrize(
    ("status", "code"), [(403, "40003"), (500, "5000"), (404, "20102"), (404, "unknown")]
)
@pytest.mark.parametrize("container_path", ["sections/s-a/pages", "notebooks/a/sections"])
def test_container_listing_errors_other_than_20113_are_never_no_children(
    pipeline, monkeypatch, first_sync, status, code, container_path
):
    from cognee.tasks.ingestion.connectors import onenote

    monkeypatch.setattr(onenote.time, "sleep", lambda _: None)
    graph = Graph()
    if not first_sync:
        (table,) = sync(pipeline, graph)
        before = rows(pipeline, table)
    graph.errors[container_path] = (status, code)
    tables = sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["complete"] == 0
    assert graph.last_stats["deleted"] == 0
    if first_sync:
        assert rows(pipeline, tables[0]) == []
    else:
        assert rows(pipeline, table) == before


def test_page_html_without_a_body_falls_back_to_its_whole_text(pipeline):
    graph = Graph()
    original = graph.handle

    def bare(request):
        if request.url.path.endswith("/pages/p1/content"):
            return httpx.Response(
                200,
                text="<title>Bare</title><p>Fragment without body</p>",
                headers={"content-type": "text/html"},
            )
        return original(request)

    graph.handle = bare
    (table,) = sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1
    assert "Fragment without body" in rows(pipeline, table)[0][2]


class MyCancelled(Exception):
    """A host's own cancellation error, unrelated to Graph."""


def _chain(error):
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


@pytest.mark.parametrize("stage", ["request", "retry", "publication"])
def test_check_active_failure_reaches_the_host_unchanged(pipeline, monkeypatch, stage):
    from cognee.tasks.ingestion.connectors import onenote

    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.requests.clear()
    graph.pages["a"] = []
    cancelled = MyCancelled("connection revoked")
    attempts = []
    original = graph.handle
    monkeypatch.setattr(onenote.time, "sleep", lambda _: None)

    def handler(request):
        attempts.append(request)
        if stage == "retry":
            return httpx.Response(429, headers={"Retry-After": "0"})
        return original(request)

    graph.handle = handler

    def check_active():
        if (
            stage == "request"
            or (stage == "retry" and attempts)
            or (stage == "publication" and any(path == "pages/p1" for path, _ in graph.requests))
        ):
            raise cancelled

    with graph.client() as client:
        resource = onenote_source(
            "offline-token",
            notebook_id="a",
            account_id="tenant:user",
            http_client=client,
            check_active=check_active,
        )
        with pytest.raises(PipelineStepFailed) as captured:
            pipeline.run(resource, write_disposition="merge")
    chain = list(_chain(captured.value))
    assert any(error is cancelled for error in chain)
    # Nothing between dlt and the host's exception: no wrapper type of our own.
    assert not any(type(error) is RuntimeError for error in chain)
    assert not any("Callback" in type(error).__name__ for error in chain)
    assert not any(isinstance(error, OneNoteError) for error in chain)
    assert rows(pipeline, table) == before
    assert resource.cognee_sync_stats["failed"] == 0


def test_check_active_failure_is_not_counted_as_a_graph_failure():
    # Driven without dlt: the exception that comes out of the walk is the host's own object.
    from cognee.tasks.ingestion.connectors.onenote import _Graph

    cancelled = MyCancelled("connection revoked")

    def check_active():
        raise cancelled

    graph = Graph()
    with graph.client() as client:
        walker = _Graph("offline-token", client, check_active=check_active)
        with pytest.raises(MyCancelled) as captured:
            walker.request("https://graph.microsoft.com/v1.0/me/onenote/notebooks")
    assert captured.value is cancelled
    assert graph.requests == []


def test_token_provider_failure_reaches_the_host_unchanged(pipeline):
    graph = Graph()
    failure = MyCancelled("token refresh revoked")

    def token():
        raise failure

    with graph.client() as client:
        resource = onenote_source(
            token, notebook_id="a", account_id="tenant:user", http_client=client
        )
        with pytest.raises(PipelineStepFailed) as captured:
            pipeline.run(resource, write_disposition="merge")
    chain = list(_chain(captured.value))
    assert any(error is failure for error in chain)
    assert not any(type(error) is RuntimeError for error in chain)
    assert not any("Callback" in type(error).__name__ for error in chain)
    assert graph.requests == []


def test_carried_confirmation_is_not_trusted_when_a_listing_flaps(pipeline):
    # Run 1 confirms p1 as deleted and is cut by the budget before p2. p1 is then restored, and
    # run 2's section listing answers 20113 (read as "no children"), so p1 looks missing again.
    # The confirmation carried from run 1 must not stand in for a fresh check of p1.
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    sync(pipeline, graph, max_requests=5)
    assert graph.last_stats["failed_budget"] == 1
    assert list(resource_cache(pipeline, table)["confirmed_missing"]) == ["p1"]

    graph.pages["a"] = [Graph.page("p1")]
    graph.errors["sections/s-a/pages"] = (404, "20113")
    graph.requests.clear()
    sync(pipeline, graph)

    assert [path for path, _ in graph.requests].count("pages/p1") == 1
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["deleted"] == 0
    assert rows(pipeline, table) == before


def test_carried_confirmation_is_not_trusted_when_the_same_section_flaps(pipeline):
    # Same container as the recording run: its listing keeps answering 20113 after the section
    # and its page were restored. The section itself is asked about directly and resolves.
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.delete_section()
    sync(pipeline, graph, max_requests=5)
    assert graph.last_stats["failed_budget"] == 1
    cache = resource_cache(pipeline, table)
    assert list(cache["confirmed_missing"]) == ["p1"]
    assert cache["swallowed"] == ["sections/s-a/pages"]

    graph.pages["a"] = [Graph.page("p1")]
    graph.deleted_sections.clear()
    graph.requests.clear()
    sync(pipeline, graph)

    paths = [path for path, _ in graph.requests]
    assert paths.count("sections/s-a") == 1
    assert paths.count("pages/p1") == 1
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["deleted"] == 0
    assert rows(pipeline, table) == before


def test_a_section_that_is_still_deleted_costs_one_check_and_keeps_the_confirmations(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    graph.delete_section()
    sync(pipeline, graph, max_requests=5)
    assert list(resource_cache(pipeline, table)["confirmed_missing"]) == ["p1"]
    graph.requests.clear()
    sync(pipeline, graph)
    paths = [path for path, _ in graph.requests]
    assert paths.count("sections/s-a") == 1
    assert paths.count("pages/p1") == 0
    assert graph.last_stats["complete"] == 1
    assert graph.last_stats["deleted"] == 2
    assert rows(pipeline, table) == []


def test_carried_confirmations_survive_a_budget_cut_while_a_section_stays_deleted(pipeline):
    graph = deletion_graph()
    (table,) = sync(pipeline, graph)
    graph.delete_section()
    for _ in range(8):
        sync(pipeline, graph, max_requests=8)
        if graph.last_stats["complete"]:
            break
    assert graph.last_stats["complete"] == 1
    assert graph.last_stats["deleted"] == 12
    assert rows(pipeline, table) == []


def test_a_blank_title_is_rendered_untitled_and_syncs(pipeline):
    graph = Graph()
    graph.pages["a"][0]["title"] = ""
    (table,) = sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1
    assert rows(pipeline, table)[0][1] == "Untitled"


def test_page_without_a_title_field_still_fails_the_notebook(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    del graph.pages["a"][0]["title"]
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["complete"] == 0
    assert rows(pipeline, table) == before


def test_page_without_a_web_link_syncs_with_an_empty_url(pipeline):
    graph = Graph()
    graph.pages["a"][0]["links"] = {}
    (table,) = sync(pipeline, graph)
    content = rows(pipeline, table)[0][2]
    assert content.startswith("Plan\n")
    with (
        pipeline.sql_client() as client,
        client.execute_query(f'SELECT url FROM "{table}"') as cursor,
    ):
        assert cursor.fetchall() == [("",)]


def test_rate_limit_and_budget_stop_the_run_with_their_own_counts(pipeline, monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    monkeypatch.setattr(onenote.time, "sleep", lambda _: None)
    graph = Graph()
    original = graph.handle
    graph.handle = lambda request: httpx.Response(429, json={"error": {"code": "10001"}})
    sync(pipeline, graph)
    assert graph.last_stats["failed_rate_limit"] == 1
    assert graph.last_stats["failed_budget"] == 0
    graph.handle = original
    sync(pipeline, graph, max_requests=1)
    assert graph.last_stats["failed_budget"] == 1
    assert graph.last_stats["failed_rate_limit"] == 0


def test_retry_after_beyond_the_wait_budget_counts_as_a_rate_limit(pipeline, monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    delays = []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)
    graph = Graph()
    graph.handle = lambda request: httpx.Response(
        429, headers={"Retry-After": "999"}, json={"error": {"code": "10001"}}
    )
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["failed_rate_limit"] == 1
    assert graph.last_stats["failed_budget"] == 0
    assert delays == []


def test_an_unchanged_page_is_counted_as_skipped(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    sync(pipeline, graph)
    assert (graph.last_stats["scanned"], graph.last_stats["skipped"]) == (2, 0)
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    sync(pipeline, graph)
    assert (graph.last_stats["scanned"], graph.last_stats["skipped"]) == (2, 1)
    sync(pipeline, graph)
    assert (graph.last_stats["scanned"], graph.last_stats["skipped"]) == (2, 2)


def test_the_page_lookup_asks_graph_for_the_parents(pipeline):
    graph = Graph()
    sync(pipeline, graph)
    graph.pages["b"].append(graph.pages["a"].pop())
    graph.requests.clear()
    sync(pipeline, graph)
    assert (graph.last_stats["complete"], graph.last_stats["deleted"]) == (1, 1)
    (lookup,) = [params for path, params in graph.requests if path == "pages/p1"]
    assert set(lookup["$expand"].split(",")) == {"parentNotebook", "parentSection"}


def test_a_page_lookup_answering_without_a_parent_fails_the_run(pipeline):
    # No parentNotebook must never be read as "not moved", so the run stops instead.
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    graph.script("pages/p1", httpx.Response(200, json=Graph.page("p1")))
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert rows(pipeline, table) == before


def test_410_with_20113_confirms_a_page_deletion(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = [Graph.page("p2")]
    graph.errors["pages/p1"] = (410, "20113")
    sync(pipeline, graph)
    assert (graph.last_stats["complete"], graph.last_stats["deleted"]) == (1, 1)
    assert [row[0] for row in rows(pipeline, table)] == ["p2"]


def test_410_with_20113_on_a_container_listing_is_read_as_no_children(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    graph.errors["sections/s-a/pages"] = (410, "20113")
    sync(pipeline, graph)
    assert (graph.last_stats["complete"], graph.last_stats["deleted"]) == (1, 1)
    assert rows(pipeline, table) == []


def test_410_with_20113_on_the_notebook_lookup_confirms_its_pages_are_gone(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = []
    del graph.notebooks["a"]
    graph.errors["notebooks/a"] = (410, "20113")
    sync(pipeline, graph)
    assert (graph.last_stats["complete"], graph.last_stats["deleted"]) == (1, 1)
    assert rows(pipeline, table) == []


def test_410_with_20102_keeps_the_page_and_counts_it_unresolved(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    graph.errors["pages/p1"] = (410, "20102")
    sync(pipeline, graph)
    stats = graph.last_stats
    assert (stats["complete"], stats["failed"], stats["deleted"], stats["unresolved"]) == (
        1,
        0,
        0,
        1,
    )
    assert rows(pipeline, table) == before


def test_a_section_answering_410_is_still_deleted_and_keeps_the_confirmations(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    graph.delete_section()
    for path in ("sections/s-a/pages", "sections/s-a", "pages/p1", "pages/p2"):
        graph.errors[path] = (410, "20113")
    sync(pipeline, graph, max_requests=5)
    assert list(resource_cache(pipeline, table)["confirmed_missing"]) == ["p1"]
    graph.requests.clear()
    sync(pipeline, graph)
    paths = [path for path, _ in graph.requests]
    assert paths.count("sections/s-a") == 1
    assert paths.count("pages/p1") == 0
    assert (graph.last_stats["complete"], graph.last_stats["deleted"]) == (1, 2)
    assert rows(pipeline, table) == []


@pytest.mark.parametrize("content_type", ["application/json", "text/plain", ""])
def test_page_content_that_is_not_html_fails_the_run_and_is_not_cached(
    pipeline, caplog, content_type
):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = "<p>Not what Graph promised</p>"
    graph.content_types["p1"] = content_type
    sync(pipeline, graph)
    assert (graph.last_stats["failed"], graph.last_stats["complete"]) == (1, 0)
    assert "content is not HTML" in caplog.text
    assert rows(pipeline, table) == before
    assert "Not what Graph promised" not in str(resource_cache(pipeline, table))


def test_a_notebook_lookup_answering_with_another_identity_fails_the_run(pipeline, caplog):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    del graph.notebooks["a"]
    graph.script("notebooks/a", httpx.Response(200, json={"id": "b", "displayName": "Excluded"}))
    sync(pipeline, graph)
    assert (graph.last_stats["failed"], graph.last_stats["complete"]) == (1, 0)
    assert "Notebook identity mismatch" in caplog.text
    assert rows(pipeline, table) == before


def test_a_page_listed_in_two_sections_fails_the_run(pipeline, caplog):
    graph = Graph()
    graph.nest_sections()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.root_sections_hidden = False
    sync(pipeline, graph)
    assert (graph.last_stats["failed"], graph.last_stats["complete"]) == (1, 0)
    assert "multiple locations" in caplog.text
    assert rows(pipeline, table) == before


def test_a_page_lookup_answering_with_another_identity_fails_the_run(pipeline, caplog):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.pages["a"] = []
    # Another page, moved to another notebook: its parent would confirm p1 as moved.
    other = {**Graph.page("other"), "parentNotebook": {"id": "b", "displayName": "Excluded"}}
    graph.script("pages/p1", httpx.Response(200, json=other))
    sync(pipeline, graph)
    assert (graph.last_stats["failed"], graph.last_stats["complete"]) == (1, 0)
    assert "Page identity mismatch" in caplog.text
    assert rows(pipeline, table) == before


@pytest.mark.parametrize("cycle", [["g1"], ["g1", "g2"]], ids=["self", "pair"])
def test_a_section_group_cycle_fails_the_run_before_the_request_budget(pipeline, caplog, cycle):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.groups["notebooks/a"] = [(cycle[0], "Loop")]
    for index, group in enumerate(cycle):
        following = cycle[(index + 1) % len(cycle)]
        graph.groups[f"sectionGroups/{group}"] = [(following, "Loop")]
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["failed_budget"] == 0
    assert graph.last_stats["complete"] == 0
    assert "Section group cycle" in caplog.text
    assert rows(pipeline, table) == before


def test_next_link_pagination_is_followed_and_keeps_the_ordering(pipeline):
    graph = Graph()
    graph.next_link_size = 2
    graph.pages["a"] = [Graph.page(f"p{index}") for index in range(5)]
    graph.html.update({f"p{index}": f"<p>Body {index}</p>" for index in range(5)})
    (table,) = sync(pipeline, graph)
    assert graph.last_stats["complete"] == 1
    assert len(rows(pipeline, table)) == 5
    calls = [params for path, params in graph.requests if path == "sections/s-a/pages"]
    assert [call.get("$skiptoken") for call in calls] == [None, "2", "4"]
    assert all("$skip" not in call for call in calls[1:])
    assert all(call["$orderby"] == "createdDateTime asc" for call in calls)


def test_a_server_error_exhausts_five_attempts_without_counting_as_a_rate_limit(
    pipeline, monkeypatch
):
    from cognee.tasks.ingestion.connectors import onenote

    delays = []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)
    graph = Graph()
    graph.script("notebooks", *[(500, {})] * 5)
    sync(pipeline, graph)
    assert graph.last_stats["failed"] == 1
    assert graph.last_stats["failed_rate_limit"] == 0
    assert graph.last_stats["requests"] == 5
    assert delays == [1, 2, 4, 8]


def test_a_throttled_request_waits_for_retry_after_and_the_run_completes(pipeline, monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    delays = []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)
    graph = Graph()
    graph.script("notebooks", (429, {"Retry-After": "3"}), (503, {"Retry-After": "1"}))
    (table,) = sync(pipeline, graph)
    assert (graph.last_stats["complete"], graph.last_stats["failed"]) == (1, 0)
    assert delays == [3, 1]
    assert len(rows(pipeline, table)) == 1


@pytest.mark.parametrize(
    "late_call",
    [1, 2, 3, 4],
    ids=["before_publication", "before_deletion", "before_first_row", "between_rows"],
)
def test_check_active_failure_after_the_walk_publishes_nothing(pipeline, late_call):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    # p1 is deleted, p2 edited and p3 new: the walk ends with its seventh request (the p1
    # lookup), then checks before publishing, before the p1 deletion, and before each of the
    # rows p2, p3.
    graph.pages["a"] = [Graph.page("p2", "2026-10-09T00:00:00Z"), Graph.page("p3")]
    graph.html["p3"] = "<p>Third body</p>"
    graph.requests.clear()
    cancelled = MyCancelled("connection revoked")
    late = 0

    def check_active():
        nonlocal late
        if len(graph.requests) == 7:
            late += 1
            if late == late_call:
                raise cancelled

    with pytest.raises(PipelineStepFailed) as captured:
        sync(pipeline, graph, check_active=check_active)
    chain = list(_chain(captured.value))
    assert any(error is cancelled for error in chain)
    assert not any(isinstance(error, OneNoteError) for error in chain)
    assert len(graph.requests) == 7
    assert rows(pipeline, table) == before
    assert sorted(resource_cache(pipeline, table)["published"]) == ["p1", "p2"]


def test_the_first_sync_publishes_every_page_and_an_edit_only_that_page(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{index}") for index in range(5)]
    graph.html.update({f"p{index}": f"<p>Body {index}</p>" for index in range(5)})
    (table,) = sync(pipeline, graph)
    assert loaded_rows(pipeline, table) == 5
    graph.pages["a"][2]["lastModifiedDateTime"] = "2026-10-05T00:00:00Z"
    graph.html["p2"] = "<p>Body 2 edited</p>"
    sync(pipeline, graph)
    assert loaded_rows(pipeline, table) == 1
    assert graph.last_stats["skipped"] == 4
    assert {row[0]: row[2] for row in rows(pipeline, table)}["p2"] == "Body 2 edited"
    assert len(rows(pipeline, table)) == 5


def test_an_edit_and_a_deletion_publish_one_row_and_one_tombstone(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2"), Graph.page("p3")]
    graph.html.update(p1="<p>One</p>", p2="<p>Two</p>", p3="<p>Three</p>")
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = [Graph.page("p1", "2026-10-05T00:00:00Z"), Graph.page("p3")]
    graph.html["p1"] = "<p>One edited</p>"
    graph.mark_deleted("p2")
    sync(pipeline, graph)
    assert loaded_rows(pipeline, table) == 2
    assert graph.last_stats["deleted"] == 1
    assert {row[0] for row in rows(pipeline, table)} == {"p1", "p3"}


def test_state_costs_one_short_fingerprint_per_page(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page(f"p{index}") for index in range(50)]
    graph.html.update({f"p{index}": "<p>" + "text " * 2000 + "</p>" for index in range(50)})
    (table,) = sync(pipeline, graph)
    assert len(json.dumps(resource_cache(pipeline, table))) < 50 * 100


def test_moving_a_page_into_a_section_group_rewrites_its_structure_only(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.nest_sections()
    sync(pipeline, graph)
    assert loaded_rows(pipeline, table) == 1
    assert rows(pipeline, table) == before
    assert [entry["kind"] for entry in structures(pipeline, table)["p1"]] == [
        "section",
        "sectionGroup",
        "notebook",
    ]


def test_a_rename_costs_one_html_request_per_affected_page_and_no_extraction(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>One</p>", p2="<p>Two</p>")
    (table,) = sync(pipeline, graph)
    before = rows(pipeline, table)
    graph.sections["a"] = "Renamed"
    sync(pipeline, graph)
    assert graph.content_requests() == {"pages/p1/content": 2, "pages/p2/content": 2}
    assert rows(pipeline, table) == before
    assert {entry[0]["name"] for entry in structures(pipeline, table).values()} == {"Renamed"}


def _hidden_pages_graph(count=12):
    """A kept page plus ``count`` pages that vanished from the listing and answer 20102."""
    graph = deletion_graph(count)
    hidden = [page["id"] for page in graph.pages["a"]]
    return graph, hidden


def test_unresolved_pages_outnumbering_the_budget_never_block_the_notebook(pipeline):
    graph, hidden = _hidden_pages_graph()
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = [Graph.page("kept")]
    graph.html["kept"] = "<p>Kept page body</p>"
    for page_id in hidden:
        graph.hide_page(page_id)
    # Finding out which pages are unresolved costs one request each, so the first runs are
    # cut by the budget and checkpoint what they learned.
    for _ in range(10):
        sync(pipeline, graph, max_requests=8)
        if graph.last_stats["complete"]:
            break
    assert graph.last_stats["complete"] == 1
    checked = set()
    for _ in range(6):
        graph.requests.clear()
        sync(pipeline, graph, max_requests=8)
        stats = graph.last_stats
        assert (stats["complete"], stats["failed"], stats["failed_budget"]) == (1, 0, 0)
        assert (stats["unresolved"], stats["deleted"]) == (12, 0)
        assert stats["requests"] == 8
        checked |= {path.removeprefix("pages/") for path, _ in graph.requests if "deleted-" in path}
    # Every unresolved page was asked about again, and none was forgotten.
    assert checked == set(hidden)
    assert {row[0] for row in rows(pipeline, table)} == {"kept", *hidden}


def test_a_page_that_answers_20113_on_a_later_check_is_forgotten(pipeline):
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2"), Graph.page("p3")]
    graph.html.update(p1="<p>One</p>", p2="<p>Two</p>", p3="<p>Three</p>")
    (table,) = sync(pipeline, graph)
    graph.pages["a"] = [Graph.page("p3")]
    graph.hide_page("p1")
    graph.hide_page("p2")
    sync(pipeline, graph)
    assert graph.last_stats["unresolved"] == 2
    graph.hidden.discard("p1")
    graph.mark_deleted("p1")
    sync(pipeline, graph)
    assert (graph.last_stats["deleted"], graph.last_stats["unresolved"]) == (1, 1)
    assert resource_cache(pipeline, table)["unresolved"] == ["p2"]
    assert {row[0] for row in rows(pipeline, table)} == {"p2", "p3"}


def test_a_page_that_is_listed_again_leaves_the_unresolved_queue(pipeline):
    graph = Graph()
    (table,) = sync(pipeline, graph)
    page = graph.pages["a"].pop()
    graph.hide_page("p1")
    sync(pipeline, graph)
    assert resource_cache(pipeline, table)["unresolved"] == ["p1"]
    graph.pages["a"].append(page)
    sync(pipeline, graph)
    assert resource_cache(pipeline, table)["unresolved"] == []
    assert graph.last_stats["unresolved"] == 0


def test_the_retry_wait_budget_is_one_budget_for_the_whole_sync(pipeline, monkeypatch):
    from cognee.tasks.ingestion.connectors import onenote

    delays = []
    monkeypatch.setattr(onenote.time, "sleep", delays.append)
    graph = Graph()
    graph.script("notebooks", (429, {"Retry-After": "40"}))
    graph.script("notebooks/a/sections", (429, {"Retry-After": "40"}))
    (table,) = sync(pipeline, graph)
    assert delays == [40]
    assert graph.last_stats["failed_rate_limit"] == 1
    assert (graph.last_stats["failed"], graph.last_stats["complete"]) == (1, 0)
    assert rows(pipeline, table) == []
