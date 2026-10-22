"""The docs tail inside the code routes: on by default, opt-out, label, snapshot and provenance.

Every code route -- cognify's CODE and CODE_REPO routes and the code graph
pipeline remember(content_type="code") runs -- ends with add_code_graph_docs,
which awaits code_docs_tail once per repository label the run produced, with
the route's ctx. The tail and the enola tasks are stubbed here; what is
checked is the wiring: the tail runs by default and not with
CODE_GRAPH_DOCS=false or code_docs=False, the label is the one enola gave the
repository (the CodeRepository the run produced, or the snapshot's when an
unchanged snapshot was skipped), every call carries the snapshot the route
loaded (``snapshot_dir``, else ``<repo_path>/.enola``; one directory for all
the repositories of a multi-repo run), whose docs.jsonl the tail reads in
place of the stored CodeRepository.path's (stale once a skipped snapshot left
an earlier run's staging directory on the node), the per-file route calls it
while the staged snapshot still exists, the ctx handed on is the route's own,
whose data item owns the code graph, and a tail that fails errors the route.
"""

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.tasks.code_graph.code_files import extract_code_files_graph, get_code_file_tasks
from cognee.tasks.code_graph.code_repo import extract_code_repo_graph, get_code_repo_tasks
from cognee.tasks.code_graph.enola import DOCS_ARTIFACT
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_docs,
    code_graph_repo_labels,
    fact_node_id,
    get_code_graph_tasks,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeRepository
from cognee.tests.utils.enola_snapshot import doc_line, enola_fact_id, write_enola_snapshot

code_docs_package = importlib.import_module("cognee.tasks.code_docs")
code_graph_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
config_module = importlib.import_module("cognee.modules.cognify.config")

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# The label enola gave the fixture snapshot's repository (its facts' "repo"
# field), which is not the scanned directory's basename.
REPO = "acme/shop"


@pytest.fixture(autouse=True)
def fresh_cognify_config(monkeypatch):
    """CODE_GRAPH_DOCS as this test sets it, not what an earlier test left in the cache."""
    monkeypatch.delenv("CODE_GRAPH_DOCS", raising=False)
    config_module.get_cognify_config.cache_clear()
    yield
    config_module.get_cognify_config.cache_clear()


def _disable_docs(monkeypatch):
    monkeypatch.setenv("CODE_GRAPH_DOCS", "false")
    config_module.get_cognify_config.cache_clear()


@pytest.fixture
def tail(monkeypatch):
    """code_docs_tail as the routes import it, from the cognee.tasks.code_docs package."""
    fake = AsyncMock(return_value={"documents": 1, "chunks": 2, "new": 2, "edges": 2, "swept": 0})
    monkeypatch.setattr(code_docs_package, "code_docs_tail", fake)
    return fake


def _repository(repo=REPO, path="/repos/shop"):
    return CodeRepository(id=fact_node_id(repo, "repository", repo), name=repo, path=path)


def _write_snapshot(directory, *repos):
    """An enola snapshot documenting one function per repository label (None: unlabelled).

    The fact and its docs.jsonl line share the fact id, as enola writes them.
    """
    facts, docs = [], []
    for repo in repos:
        name, file = "payments.charge", "payments.py"
        fact = {
            "id": enola_fact_id(repo, "symbol", name, file),
            "kind": "symbol",
            "name": name,
            "file": file,
            "line": 4,
            "props": {"symbol_kind": "function", "doc": "Charge a card."},
        }
        if repo is not None:
            fact["repo"] = repo
        facts.append(fact)
        docs.append(doc_line(repo, "symbol", name, file, "Charge a card.\n\nIdempotent.", line=4))
    return write_enola_snapshot(directory, facts=facts, docs=docs)


def _stub_code_graph(monkeypatch, extract):
    """The enola tasks as the routes call them: ``extract`` yields the run, the loads pass it on."""
    monkeypatch.setattr(code_graph_module, "extract_code_graph", extract)
    for name in ("add_code_graph_data_points", "add_code_graph_edges"):
        monkeypatch.setattr(
            code_graph_module,
            name,
            AsyncMock(side_effect=lambda data_points, **_kwargs: data_points),
        )


def _ctx(data_item):
    return SimpleNamespace(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=data_item,
        pipeline_run_id=uuid4(),
    )


def _repo_manifest_item(tmp_path):
    """A CODE_REPO manifest row for a directory named differently from enola's label."""
    repo_dir = tmp_path / "shop"
    repo_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo_path": str(repo_dir)}))
    item = SimpleNamespace(
        id=uuid4(),
        raw_data_location=str(manifest),
        system_metadata={"source": "code_repo"},
        external_metadata=None,
    )
    return repo_dir, item


def _code_file_item(tmp_path):
    source = tmp_path / "payments.py"
    source.write_text('"""Payments."""\n\n\ndef charge():\n    """Charge a card."""\n')
    return SimpleNamespace(
        id=uuid4(),
        raw_data_location=str(source),
        original_data_location=str(source),
        system_metadata={"source": "code"},
    )


@pytest.mark.asyncio
async def test_repo_route_stores_docstrings_by_default_with_the_route_ctx(
    tmp_path, monkeypatch, tail
):
    repo_dir, item = _repo_manifest_item(tmp_path)
    # The run's CodeRepository carries the scanned directory as its path, as
    # map_facts_to_data_points stores it on the primary repository.
    _stub_code_graph(monkeypatch, AsyncMock(return_value=[_repository(path=str(repo_dir))]))
    ctx = _ctx(item)

    await extract_code_repo_graph([item], ctx=ctx)

    # The label is the one the run's CodeRepository carries, not the directory
    # name ("shop"). The snapshot enola just wrote into the directory goes with
    # it: the docs are read from its docs.jsonl, not from the snapshot under
    # the stored CodeRepository.path. The ctx is the route's own: its data
    # item is the manifest row, so add_data_points attributes the documents
    # and chunks to it and forget(data_id) removes them with the code graph.
    tail.assert_awaited_once_with(repo=REPO, ctx=ctx, snapshot_dir=repo_dir / ".enola")
    assert tail.await_args.kwargs["ctx"].data_item is item


@pytest.mark.asyncio
async def test_failing_tail_errors_the_route(tmp_path, monkeypatch, tail):
    """No usable embedding provider (keyless with COGNEE_SKIP_CONNECTION_TEST set,
    say): the tail's error reaches the route, so the item errors."""
    repo_dir, item = _repo_manifest_item(tmp_path)
    _stub_code_graph(monkeypatch, AsyncMock(return_value=[_repository(path=str(repo_dir))]))
    tail.side_effect = RuntimeError("OpenAIException - Missing credentials")

    with pytest.raises(RuntimeError, match="Missing credentials"):
        await extract_code_repo_graph([item], ctx=_ctx(item))

    tail.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_snapshot_during_label_lookup_keeps_the_existing_graph(tmp_path, tail):
    """An unchanged graph yields no points; its snapshot may disappear before the tail."""
    run = []
    assert await add_code_graph_docs(run, repo_path=tmp_path / "gone") is run
    tail.assert_not_awaited()


@pytest.mark.asyncio
async def test_repo_route_skips_the_tail_when_switched_off(tmp_path, monkeypatch, tail):
    _stub_code_graph(monkeypatch, AsyncMock(return_value=[_repository()]))
    _repo_dir, item = _repo_manifest_item(tmp_path)

    await extract_code_repo_graph([item], code_docs=False)
    tail.assert_not_awaited()

    _disable_docs(monkeypatch)
    await extract_code_repo_graph([item])
    tail.assert_not_awaited()

    # The explicit argument wins over the setting.
    await extract_code_repo_graph([item], code_docs=True)
    tail.assert_awaited_once()


@pytest.mark.asyncio
async def test_repo_route_runs_the_tail_for_an_unchanged_snapshot(tmp_path, monkeypatch, tail):
    """extract_code_graph returns [] for an unchanged snapshot. The label then comes
    from the snapshot, resolved as map_facts_to_data_points resolves it, and the
    tail still runs: it is idempotent, a docstring edit past its first paragraph
    changes docs.jsonl but not the snapshot id, and a repository indexed before
    docstrings were stored gets them on this sync."""
    _stub_code_graph(monkeypatch, AsyncMock(return_value=[]))
    repo_dir, item = _repo_manifest_item(tmp_path)
    shutil.copytree(FIXTURES_DIR, repo_dir / ".enola")
    ctx = _ctx(item)

    await extract_code_repo_graph([item], ctx=ctx)

    # The snapshot still goes with the call: the skipped snapshot left the
    # stored CodeRepository.path as whatever the first load wrote, and only the
    # route knows where this run's snapshot is.
    tail.assert_awaited_once_with(repo=REPO, ctx=ctx, snapshot_dir=repo_dir / ".enola")


@pytest.mark.parametrize("unchanged", [False, True], ids=["new-snapshot", "unchanged-snapshot"])
@pytest.mark.asyncio
async def test_file_route_runs_the_tail_inside_the_staging_directory(
    tmp_path, monkeypatch, tail, unchanged
):
    """The tail reads the docs.jsonl enola wrote into the staging directory, so it
    runs before that directory is removed. The stored CodeRepository.path of a
    re-run of an unchanged file is the previous run's staging directory, gone:
    an unchanged snapshot (extract_code_graph returns []) takes its label from
    the staged snapshot and the tail reads that snapshot all the same."""
    staged = {}

    async def extract(repo_path=None, **_kwargs):
        # What enola leaves in the staged directory: its snapshot, the
        # repository labelled with the directory's name.
        staged["dir"] = Path(repo_path)
        label = staged["dir"].name
        _write_snapshot(staged["dir"] / ".enola", label)
        return [] if unchanged else [_repository(repo=label, path=str(repo_path))]

    async def tail_reads_the_staged_snapshot(repo, ctx=None, snapshot_dir=None):
        staged["docs_present"] = (Path(snapshot_dir) / DOCS_ARTIFACT).is_file()
        return {}

    _stub_code_graph(monkeypatch, extract)
    tail.side_effect = tail_reads_the_staged_snapshot
    item = _code_file_item(tmp_path)
    ctx = _ctx(item)

    await extract_code_files_graph([item], ctx=ctx)

    tail.assert_awaited_once_with(
        repo=staged["dir"].name, ctx=ctx, snapshot_dir=staged["dir"] / ".enola"
    )
    assert staged["docs_present"] is True, "the tail ran after the staged snapshot was removed"
    assert not staged["dir"].exists()


@pytest.mark.asyncio
async def test_file_route_skips_the_tail_when_switched_off(tmp_path, monkeypatch, tail):
    _stub_code_graph(monkeypatch, AsyncMock(return_value=[_repository()]))
    item = _code_file_item(tmp_path)

    await extract_code_files_graph([item], code_docs=False)
    tail.assert_not_awaited()

    _disable_docs(monkeypatch)
    await extract_code_files_graph([item])
    tail.assert_not_awaited()


def test_route_task_lists_bind_the_resolved_setting(monkeypatch):
    """cognify resolves code_docs once; the builders bind it on the task like summary_method."""
    for build in (get_code_repo_tasks, get_code_file_tasks):
        (task,) = build()
        assert task.default_params["kwargs"]["code_docs"] is True
        assert task.needs_llm is False
        assert build(code_docs=False)[0].default_params["kwargs"]["code_docs"] is False

    _disable_docs(monkeypatch)
    for build in (get_code_repo_tasks, get_code_file_tasks):
        assert build()[0].default_params["kwargs"]["code_docs"] is False
        assert build(code_docs=True)[0].default_params["kwargs"]["code_docs"] is True


def test_code_graph_tasks_end_with_the_docs_tail_by_default(monkeypatch):
    tasks = get_code_graph_tasks("/tmp/shop", snapshot_dir=FIXTURES_DIR)

    assert [task.executable.__name__ for task in tasks] == [
        "extract_code_graph",
        "add_code_graph_data_points",
        "add_code_graph_edges",
        "add_code_graph_docs",
    ]
    docs_task = tasks[-1]
    assert docs_task.needs_llm is False
    assert docs_task.accepts_ctx
    assert docs_task.default_params["kwargs"] == {
        "repo_path": "/tmp/shop",
        "snapshot_dir": FIXTURES_DIR,
    }
    assert len(get_code_graph_tasks("/tmp/shop", code_docs=False)) == 3

    _disable_docs(monkeypatch)
    assert len(get_code_graph_tasks("/tmp/shop")) == 3
    assert get_code_graph_tasks("/tmp/shop", code_docs=True)[-1].executable is add_code_graph_docs


@pytest.mark.asyncio
async def test_docs_task_runs_the_tail_once_per_repository_with_one_snapshot(tmp_path, tail):
    """A multi-repository snapshot yields one CodeRepository per repo: the tail runs
    for each, in label order, and every call reads the one snapshot directory the
    run loaded -- enola writes every repository's docs into it, each line naming
    its repository -- whatever path each repository's node carries. The data
    points pass through unchanged."""
    ctx = _ctx(SimpleNamespace(id=uuid4()))
    run = [
        _repository("zeta/api", path="zeta/api"),
        _repository("acme/shop", path="/tmp/shop"),
        SimpleNamespace(name="not a repo"),
    ]

    result = await add_code_graph_docs(run, ctx=ctx, repo_path="/tmp/shop")

    assert result is run
    snapshot = Path("/tmp/shop") / ".enola"
    assert [call.kwargs for call in tail.await_args_list] == [
        {"repo": "acme/shop", "ctx": ctx, "snapshot_dir": snapshot},
        {"repo": "zeta/api", "ctx": ctx, "snapshot_dir": snapshot},
    ]

    # An unchanged multi-repository snapshot produced no data points: the labels
    # come from the snapshot, and every call still reads that one directory.
    tail.reset_mock()
    repo_dir = tmp_path / "shop"
    _write_snapshot(repo_dir / ".enola", "zeta/api", "acme/shop")

    assert await add_code_graph_docs([], ctx=ctx, repo_path=repo_dir) == []
    assert [call.kwargs for call in tail.await_args_list] == [
        {"repo": "acme/shop", "ctx": ctx, "snapshot_dir": repo_dir / ".enola"},
        {"repo": "zeta/api", "ctx": ctx, "snapshot_dir": repo_dir / ".enola"},
    ]


@pytest.mark.asyncio
async def test_docs_task_hands_the_tail_the_snapshot_the_run_loaded(tail):
    """The code graph pipeline's snapshot_dir wins over <repo_path>/.enola, where the
    routes' enola run writes it. A call naming neither hands the tail none: the
    extraction then reads <CodeRepository.path>/.enola, as the memify pipeline does."""
    run = [_repository()]

    await add_code_graph_docs(run, repo_path="/tmp/shop", snapshot_dir=FIXTURES_DIR)
    await add_code_graph_docs(run, snapshot_dir=FIXTURES_DIR)
    await add_code_graph_docs(run, repo_path="/tmp/shop")
    await add_code_graph_docs(run)

    assert [call.kwargs["snapshot_dir"] for call in tail.await_args_list] == [
        FIXTURES_DIR,
        FIXTURES_DIR,
        Path("/tmp/shop") / ".enola",
        None,
    ]
    assert {call.kwargs["repo"] for call in tail.await_args_list} == {REPO}


def test_repo_labels_come_from_the_run_or_else_the_snapshot(tmp_path):
    run = [_repository("b/two", path="b/two"), _repository("a/one", path="/tmp/shop")]
    # The run's CodeRepository points name the labels; no snapshot is read for
    # them (there is none at this path).
    assert code_graph_repo_labels(run, repo_path=tmp_path / "missing") == ["a/one", "b/two"]

    # An unchanged snapshot produced no data points: the labels are resolved
    # from the snapshot -- the pipeline's snapshot_dir, else <repo_path>/.enola --
    # exactly as map_facts_to_data_points resolves them, so they name the same
    # CodeRepository nodes.
    assert code_graph_repo_labels([], repo_path="/tmp/shop", snapshot_dir=FIXTURES_DIR) == [REPO]
    for labels, directory in ((("zeta/api", "acme/shop"), "multi"), ((None,), "unlabelled")):
        repo_dir = tmp_path / directory
        snapshot = _write_snapshot(repo_dir / ".enola", *labels)
        facts = [json.loads(line) for line in (snapshot / "facts.jsonl").read_text().splitlines()]
        mapped = sorted(
            point.name
            for point in map_facts_to_data_points(facts, repo_path=repo_dir)
            if isinstance(point, CodeRepository)
        )
        assert code_graph_repo_labels([], repo_path=repo_dir) == mapped
    # The multi-repo snapshot names both of its repositories; the unlabelled
    # one falls back on the scanned directory's name.
    assert code_graph_repo_labels([], repo_path=tmp_path / "multi") == ["acme/shop", "zeta/api"]
    assert code_graph_repo_labels([], repo_path=tmp_path / "unlabelled") == ["unlabelled"]

    # Nothing to resolve from: no label, so the tail has nothing to run for.
    assert code_graph_repo_labels([]) == []
